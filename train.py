import argparse
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


class CKExtendedDataset(Dataset):
	def __init__(self, dataframe: pd.DataFrame, label_to_idx: Optional[Dict[int, int]] = None):
		self.dataframe = dataframe.reset_index(drop=True)
		self.label_to_idx = label_to_idx

	def __len__(self) -> int:
		return len(self.dataframe)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		row = self.dataframe.iloc[idx]
		pixels = np.fromstring(row["pixels"], dtype=np.float32, sep=" ")

		# CK+ grayscale faces are flattened 48x48.
		image = torch.tensor(pixels).view(1, 48, 48) / 255.0
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
			CKExtendedDataset(train_df, label_to_idx=label_to_idx),
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
		if labels.min().item() < 0 or labels.max().item() >= outputs.size(1):
			raise ValueError(
				f"Label out of range: min={labels.min().item()} max={labels.max().item()} "
				f"for num_classes={outputs.size(1)}"
			)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		running_loss += loss.item() * images.size(0)
		preds = outputs.argmax(dim=1)
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
		if criterion is not None:
			loss = criterion(outputs, labels)
			running_loss += loss.item() * images.size(0)

		preds = outputs.argmax(dim=1)
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

	if "train" not in loaders:
		raise ValueError("Training split not found in CSV (Usage == 'Training').")

	inferred_num_classes = len(label_to_idx)
	if args.num_classes != inferred_num_classes:
		print(
			f"[info] overriding --num-classes={args.num_classes} "
			f"with inferred value {inferred_num_classes} from dataset labels."
		)

	model = build_model(num_classes=inferred_num_classes).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

	best_val_acc = -1.0
	for epoch in range(1, args.epochs + 1):
		train_loss, train_acc = train_one_epoch(
			model,
			loaders["train"],
			criterion,
			optimizer,
			device,
			epoch=epoch,
			total_epochs=args.epochs,
		)

		if "val" in loaders:
			val_loss, val_acc = test(model, loaders["val"], criterion, device)
			epoch_msg = (
				f"Epoch {epoch:03d}/{args.epochs} | "
				f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
				f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
			)
			print(epoch_msg)
			logger.info(epoch_msg)
			if val_acc > best_val_acc:
				best_val_acc = val_acc
				torch.save(model.state_dict(), args.save)
				logger.info("saved best model to %s | best_val_acc=%.4f", args.save, best_val_acc)
		else:
			epoch_msg = (
				f"Epoch {epoch:03d}/{args.epochs} | "
				f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
			)
			print(epoch_msg)
			logger.info(epoch_msg)

	if "test" in loaders:
		test_loss, test_acc = test(model, loaders["test"], criterion, device)
		test_msg = f"Test  | loss={test_loss:.4f} acc={test_acc:.4f}"
		print(test_msg)
		logger.info(test_msg)

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
	parser.add_argument("--save", type=str, default="best_model.pt")
	parser.add_argument("--log-file", type=str, default="train.log")
	parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto")
	return parser


if __name__ == "__main__":
	args = build_argparser().parse_args()
	run(args)
