import argparse
import math
import logging
import os
import sys
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from model import build_model


def update_class_prototypes(
	prototypes: torch.Tensor,
	prototype_counts: torch.Tensor,
	embeddings: torch.Tensor,
	labels: torch.Tensor,
	momentum: float,
) -> None:
	"""Update running class prototypes using batch embeddings."""
	with torch.no_grad():
		for cls in labels.unique():
			cls_idx = int(cls.item())
			cls_mask = labels == cls_idx
			cls_embeddings = embeddings[cls_mask]
			if cls_embeddings.numel() == 0:
				continue
			batch_mean = cls_embeddings.mean(dim=0)
			if prototype_counts[cls_idx].item() == 0:
				prototypes[cls_idx] = batch_mean
			else:
				prototypes[cls_idx] = momentum * prototypes[cls_idx] + (1.0 - momentum) * batch_mean
			prototype_counts[cls_idx] += cls_mask.sum()


def detect_boundary_samples(
	embeddings: torch.Tensor,
	labels: torch.Tensor,
	prototypes: torch.Tensor,
	prototype_counts: torch.Tensor,
	margin: float,
) -> torch.Tensor:
	"""Mark samples near confusing class borders using prototype distance gaps."""
	valid_proto_mask = prototype_counts > 0
	if valid_proto_mask.sum().item() < 2:
		return torch.zeros(labels.size(0), dtype=torch.bool, device=labels.device)

	dists = torch.cdist(embeddings, prototypes)
	dists[:, ~valid_proto_mask] = float("inf")

	row_idx = torch.arange(labels.size(0), device=labels.device)
	d_true = dists[row_idx, labels]

	d_other = dists.clone()
	d_other[row_idx, labels] = float("inf")
	nearest_other = d_other.min(dim=1).values

	gap = nearest_other - d_true
	return torch.isfinite(gap) & (gap <= margin)


def kl_to_uniform(ambiguity_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
	"""KL(q||u) on boundary samples; low value means high entropy confusion encoding."""
	if not mask.any():
		return ambiguity_probs.new_zeros(())

	selected = ambiguity_probs[mask].clamp_min(1e-8)
	num_classes = selected.size(1)
	return (selected * (selected.log() - math.log(1.0 / num_classes))).sum(dim=1).mean()


def compute_confusion_matrices(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Return multiclass confusion matrix and one-vs-rest multilabel confusion matrices."""
	cm = np.zeros((num_classes, num_classes), dtype=np.int64)
	for true_idx, pred_idx in zip(y_true, y_pred):
		cm[int(true_idx), int(pred_idx)] += 1

	ml_cm = np.zeros((num_classes, 2, 2), dtype=np.int64)
	for cls_idx in range(num_classes):
		true_mask = y_true == cls_idx
		pred_mask = y_pred == cls_idx
		tp = int(np.logical_and(true_mask, pred_mask).sum())
		fn = int(np.logical_and(true_mask, np.logical_not(pred_mask)).sum())
		fp = int(np.logical_and(np.logical_not(true_mask), pred_mask).sum())
		tn = int(np.logical_and(np.logical_not(true_mask), np.logical_not(pred_mask)).sum())
		ml_cm[cls_idx] = np.array([[tn, fp], [fn, tp]], dtype=np.int64)

	return cm, ml_cm


@torch.no_grad()
def collect_predictions(
	model: nn.Module,
	loader: DataLoader,
	device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
	"""Collect predictions and labels for confusion-matrix style reports."""
	model.eval()
	all_preds = []
	all_labels = []

	for images, labels in loader:
		images = images.to(device)
		outputs = model(images)
		logits = outputs["logits"]
		preds = logits.argmax(dim=1)

		all_preds.append(preds.detach().cpu().numpy())
		all_labels.append(labels.detach().cpu().numpy())

	return np.concatenate(all_labels), np.concatenate(all_preds)


class CKExtendedDataset(Dataset):
	def __init__(
		self,
		dataframe: pd.DataFrame,
		label_to_idx: Optional[Dict[int, int]] = None,
		augment: bool = False,
	):
		self.dataframe = dataframe.reset_index(drop=True)
		self.label_to_idx = label_to_idx
		self.augment = augment

	def __len__(self) -> int:
		return len(self.dataframe)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		row = self.dataframe.iloc[idx]
		pixels = np.fromstring(row["pixels"], dtype=np.float32, sep=" ")

		# CK+ grayscale faces are flattened 48x48.
		image = torch.tensor(pixels).view(1, 48, 48) / 255.0
		if self.augment:
			if torch.rand(1).item() < 0.5:
				image = torch.flip(image, dims=[2])
			if torch.rand(1).item() < 0.3:
				image = image + torch.randn_like(image) * 0.02
			if torch.rand(1).item() < 0.3:
				shift_x = int(torch.randint(-2, 3, (1,)).item())
				shift_y = int(torch.randint(-2, 3, (1,)).item())
				image = torch.roll(image, shifts=(shift_y, shift_x), dims=(1, 2))
			image = image.clamp(0.0, 1.0)
		label_raw = int(row["emotion"])
		label_value = self.label_to_idx[label_raw] if self.label_to_idx is not None else label_raw
		label = torch.tensor(label_value, dtype=torch.long)
		return image, label


def dataloader(
	csv_path: str = "ckextended.csv",
	batch_size: int = 64,
	num_workers: int = 0,
	pin_memory: bool = False,
	return_label_map: bool = False,
) -> Union[Dict[str, DataLoader], Tuple[Dict[str, DataLoader], Dict[int, int]]]:
	"""
	Build DataLoaders from CK+ CSV.

	Expects columns: emotion, pixels, Usage
	Usage values are usually: Training, PublicTest, PrivateTest
	"""
	if not os.path.exists(csv_path):
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	df = pd.read_csv(csv_path)
	required_cols = {"emotion", "pixels", "Usage"}
	if not required_cols.issubset(df.columns):
		raise ValueError(f"CSV must contain columns: {required_cols}")

	labels = sorted(df["emotion"].astype(int).unique().tolist())
	label_to_idx: Dict[int, int] = {label: idx for idx, label in enumerate(labels)}

	train_df = df[df["Usage"] == "Training"]
	val_df = df[df["Usage"] == "PublicTest"]
	test_df = df[df["Usage"] == "PrivateTest"]

	# Fallback if test/val split names differ.
	if val_df.empty and test_df.empty:
		raise ValueError(
			"No validation/test rows found. Expected Usage values like "
			"'PublicTest' and 'PrivateTest'."
		)

	loaders: Dict[str, DataLoader] = {}

	if not train_df.empty:
		loaders["train"] = DataLoader(
			CKExtendedDataset(train_df, label_to_idx=label_to_idx, augment=True),
			batch_size=batch_size,
			shuffle=True,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)

	if not val_df.empty:
		loaders["val"] = DataLoader(
			CKExtendedDataset(val_df, label_to_idx=label_to_idx),
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)

	if not test_df.empty:
		loaders["test"] = DataLoader(
			CKExtendedDataset(test_df, label_to_idx=label_to_idx),
			batch_size=batch_size,
			shuffle=False,
			num_workers=num_workers,
			pin_memory=pin_memory,
		)

	if return_label_map:
		return loaders, label_to_idx
	return loaders


def train_one_epoch(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer,
	device: torch.device,
	prototypes: torch.Tensor,
	prototype_counts: torch.Tensor,
	proto_momentum: float,
	boundary_margin: float,
	push_scale: float,
	lambda_ae: float,
	lambda_push: float,
	aux_scale: float,
	epoch: int = 1,
	total_epochs: int = 1,
) -> Tuple[float, float]:
	"""Train model for one epoch and return (loss, accuracy)."""
	model.train()

	running_loss = 0.0
	correct = 0
	total = 0
	num_batches = max(len(loader), 1)
	progress_step = max(1, num_batches // 100)

	for batch_idx, (images, labels) in enumerate(loader, start=1):
		images = images.to(device)
		labels = labels.to(device)

		optimizer.zero_grad()
		outputs = model(images)
		logits_clean = outputs["logits"]
		embeddings = outputs["embedding"]
		gamma = outputs["gamma"]

		if labels.min().item() < 0 or labels.max().item() >= logits_clean.size(1):
			raise ValueError(
				f"Label out of range: min={labels.min().item()} max={labels.max().item()} "
				f"for num_classes={logits_clean.size(1)}"
			)

		update_class_prototypes(
			prototypes,
			prototype_counts,
			embeddings.detach(),
			labels,
			momentum=proto_momentum,
		)

		boundary_mask = detect_boundary_samples(
			embeddings.detach(),
			labels,
			prototypes,
			prototype_counts,
			margin=boundary_margin,
		)

		pushed_embeddings = embeddings
		if boundary_mask.any():
			target_proto = prototypes.detach()[labels]
			step = (target_proto - embeddings) * (gamma.unsqueeze(1) * push_scale)
			pushed_embeddings = embeddings.clone()
			pushed_embeddings[boundary_mask] = embeddings[boundary_mask] + step[boundary_mask]

		pushed_outputs = model.forward_from_embedding(pushed_embeddings)
		logits_pushed = pushed_outputs["logits"]

		loss_clean = criterion(logits_clean, labels)
		if boundary_mask.any():
			loss_push = criterion(logits_pushed[boundary_mask], labels[boundary_mask])
		else:
			loss_push = logits_clean.new_zeros(())

		loss_ae = kl_to_uniform(outputs["ambiguity_probs"], boundary_mask)
		loss = loss_clean + (aux_scale * lambda_push * loss_push) + (aux_scale * lambda_ae * loss_ae)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * images.size(0)
		preds = logits_clean.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

		if batch_idx % progress_step == 0 or batch_idx == num_batches:
			percent = 100.0 * batch_idx / num_batches
			bar_len = 30
			filled = int(bar_len * batch_idx / num_batches)
			bar = "#" * filled + "-" * (bar_len - filled)
			sys.stdout.write(
				f"\rEpoch {epoch:03d}/{total_epochs:03d} "
				f"[{bar}] {percent:6.2f}% ({batch_idx}/{num_batches})"
			)
			sys.stdout.flush()

	print()

	epoch_loss = running_loss / max(total, 1)
	epoch_acc = correct / max(total, 1)
	return epoch_loss, epoch_acc


@torch.no_grad()
def test(
	model: nn.Module,
	loader: DataLoader,
	criterion: Optional[nn.Module],
	device: torch.device,
) -> Tuple[Optional[float], float]:
	"""Evaluate model and return (loss_or_none, accuracy)."""
	model.eval()

	running_loss = 0.0
	correct = 0
	total = 0

	for images, labels in loader:
		images = images.to(device)
		labels = labels.to(device)

		outputs = model(images)
		logits = outputs["logits"]
		if criterion is not None:
			loss = criterion(logits, labels)
			running_loss += loss.item() * images.size(0)

		preds = logits.argmax(dim=1)
		correct += (preds == labels).sum().item()
		total += labels.size(0)

	avg_loss = None if criterion is None else (running_loss / max(total, 1))
	acc = correct / max(total, 1)
	return avg_loss, acc


def setup_file_logger(log_file: str) -> logging.Logger:
	logger = logging.getLogger("ck_train")
	logger.setLevel(logging.INFO)
	logger.handlers.clear()

	os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
	file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
	file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
	logger.addHandler(file_handler)
	logger.propagate = False
	return logger


def run(args: argparse.Namespace) -> None:
	if args.device == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA requested but not available. Check GPU/CUDA setup.")

	device = torch.device(
		"cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
	)

	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True

	logger = setup_file_logger(args.log_file)
	logger.info("start training | device=%s | csv=%s", device, args.csv)

	loaders, label_to_idx = dataloader(
		csv_path=args.csv,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=(device.type == "cuda"),
		return_label_map=True,
	)
	idx_to_label = {idx: raw_label for raw_label, idx in label_to_idx.items()}

	if "train" not in loaders:
		raise ValueError("Training split not found in CSV (Usage == 'Training').")

	inferred_num_classes = len(label_to_idx)
	if args.num_classes != inferred_num_classes:
		print(
			f"[info] overriding --num-classes={args.num_classes} "
			f"with inferred value {inferred_num_classes} from dataset labels."
		)

	model = build_model(
		num_classes=inferred_num_classes,
		embedding_dim=args.embedding_dim,
		dropout=args.dropout,
	).to(device)
	criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		optimizer,
		mode="max",
		factor=args.lr_factor,
		patience=args.lr_patience,
	)
	prototypes = torch.zeros(inferred_num_classes, args.embedding_dim, device=device)
	prototype_counts = torch.zeros(inferred_num_classes, dtype=torch.long, device=device)

	best_val_acc = -1.0
	best_epoch = 0
	no_improve_epochs = 0
	for epoch in range(1, args.epochs + 1):
		if epoch < args.aux_start_epoch:
			aux_scale = 0.0
		else:
			ramp_progress = (epoch - args.aux_start_epoch + 1) / max(args.aux_ramp_epochs, 1)
			aux_scale = min(1.0, max(0.0, ramp_progress))

		train_loss, train_acc = train_one_epoch(
			model,
			loaders["train"],
			criterion,
			optimizer,
			device,
			prototypes,
			prototype_counts,
			proto_momentum=args.proto_momentum,
			boundary_margin=args.boundary_margin,
			push_scale=args.push_scale,
			lambda_ae=args.lambda_ae,
			lambda_push=args.lambda_push,
			aux_scale=aux_scale,
			epoch=epoch,
			total_epochs=args.epochs,
		)

		if "val" in loaders:
			val_loss, val_acc = test(model, loaders["val"], criterion, device)
			scheduler.step(val_acc)
			current_lr = optimizer.param_groups[0]["lr"]
			epoch_msg = (
				f"Epoch {epoch:03d}/{args.epochs} | "
				f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
				f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | lr={current_lr:.6f}"
			)
			print(epoch_msg)
			logger.info(epoch_msg)
			if val_acc > (best_val_acc + args.min_delta):
				best_val_acc = val_acc
				best_epoch = epoch
				no_improve_epochs = 0
				torch.save(model.state_dict(), args.save)
				logger.info("saved best model to %s | best_val_acc=%.4f", args.save, best_val_acc)
			else:
				no_improve_epochs += 1
				if no_improve_epochs >= args.patience:
					stop_msg = (
						f"early stopping at epoch {epoch} | "
						f"best_epoch={best_epoch} best_val_acc={best_val_acc:.4f}"
					)
					print(stop_msg)
					logger.info(stop_msg)
					break
		else:
			current_lr = optimizer.param_groups[0]["lr"]
			epoch_msg = (
				f"Epoch {epoch:03d}/{args.epochs} | "
				f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | lr={current_lr:.6f}"
			)
			print(epoch_msg)
			logger.info(epoch_msg)

	if "val" in loaders and os.path.exists(args.save):
		model.load_state_dict(torch.load(args.save, map_location=device))
		logger.info("loaded best checkpoint for evaluation | file=%s", args.save)

	if "test" in loaders:
		test_loss, test_acc = test(model, loaders["test"], criterion, device)
		test_msg = f"Test  | loss={test_loss:.4f} acc={test_acc:.4f}"
		print(test_msg)
		logger.info(test_msg)

		if args.print_confusion or args.print_multilabel_confusion:
			y_true, y_pred = collect_predictions(model, loaders["test"], device)
			cm, ml_cm = compute_confusion_matrices(y_true, y_pred, inferred_num_classes)

			if args.print_confusion:
				cm_text = np.array2string(cm, separator=", ")
				print("Confusion matrix (rows=true, cols=pred):")
				print(cm_text)
				logger.info("confusion_matrix:\n%s", cm_text)

			if args.print_multilabel_confusion:
				print("Multilabel confusion matrices (one-vs-rest, format [[TN, FP], [FN, TP]]):")
				for cls_idx in range(inferred_num_classes):
					label_name = idx_to_label.get(cls_idx, cls_idx)
					cls_text = np.array2string(ml_cm[cls_idx], separator=", ")
					line = f"class={label_name} idx={cls_idx} matrix={cls_text}"
					print(line)
					logger.info("multilabel_confusion | %s", line)

	logger.info("finish training")


def build_argparser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train emotion model on CK+ CSV")
	parser.add_argument("--csv", type=str, default="ckextended.csv")
	parser.add_argument("--epochs", type=int, default=20)
	parser.add_argument("--batch-size", type=int, default=64)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--weight-decay", type=float, default=1e-4)
	parser.add_argument("--num-workers", type=int, default=0)
	parser.add_argument("--num-classes", type=int, default=7)
	parser.add_argument("--embedding-dim", type=int, default=128)
	parser.add_argument("--dropout", type=float, default=0.3)
	parser.add_argument("--label-smoothing", type=float, default=0.05)
	parser.add_argument("--proto-momentum", type=float, default=0.95)
	parser.add_argument("--boundary-margin", type=float, default=0.15)
	parser.add_argument("--push-scale", type=float, default=0.5)
	parser.add_argument("--lambda-ae", type=float, default=0.02)
	parser.add_argument("--lambda-push", type=float, default=0.25)
	parser.add_argument("--aux-start-epoch", type=int, default=4)
	parser.add_argument("--aux-ramp-epochs", type=int, default=6)
	parser.add_argument("--lr-factor", type=float, default=0.5)
	parser.add_argument("--lr-patience", type=int, default=3)
	parser.add_argument("--patience", type=int, default=8)
	parser.add_argument("--min-delta", type=float, default=1e-3)
	parser.add_argument("--print-confusion", action="store_true")
	parser.add_argument("--print-multilabel-confusion", action="store_true")
	parser.add_argument("--save", type=str, default="best_model.pt")
	parser.add_argument("--log-file", type=str, default="train.log")
	parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
	return parser


if __name__ == "__main__":
	args = build_argparser().parse_args()
	run(args)
