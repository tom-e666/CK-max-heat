import torch
from torch import nn
from typing import Dict


class EmotionCNN(nn.Module):
	"""Multi-layer CNN for 48x48 grayscale emotion classification."""

	def __init__(
		self,
		num_classes: int = 7,
		dropout: float = 0.1,
		embedding_dim: int = 128,
		ae_hidden: int = 64,
	):
		super().__init__()
		self.num_classes = num_classes
		self.embedding_dim = embedding_dim

		self.features = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(dropout * 0.5),

			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(dropout * 0.6),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2),
			nn.Dropout2d(dropout * 0.8),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(inplace=True),
			nn.AdaptiveAvgPool2d((1, 1)),
		)

		self.embedding_head = nn.Sequential(
			nn.Flatten(),
			nn.Linear(256, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(128, embedding_dim),
			nn.ReLU(inplace=True),
		)

		self.ambiguity_encoder = nn.Sequential(
			nn.Linear(embedding_dim, ae_hidden),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(ae_hidden, num_classes),
		)

		self.gamma_head = nn.Sequential(
			nn.Linear(embedding_dim, ae_hidden),
			nn.ReLU(inplace=True),
			nn.Linear(ae_hidden, 1),
			nn.Sigmoid(),
		)

		self.classifier = nn.Sequential(
			nn.Linear(embedding_dim + num_classes, 128),
			nn.ReLU(inplace=True),
			nn.Dropout(dropout),
			nn.Linear(128, num_classes),
		)

	def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
		x = self.features(x)
		z = self.embedding_head(x)
		return z

	def forward_from_embedding(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
		ambiguity_logits = self.ambiguity_encoder(embedding)
		ambiguity_probs = torch.softmax(ambiguity_logits, dim=1)
		gamma = self.gamma_head(embedding).squeeze(1)
		fused = torch.cat([embedding, ambiguity_probs], dim=1)
		logits = self.classifier(fused)
		return {
			"logits": logits,
			"embedding": embedding,
			"ambiguity_logits": ambiguity_logits,
			"ambiguity_probs": ambiguity_probs,
			"gamma": gamma,
		}

	def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
		embedding = self.extract_embedding(x)
		return self.forward_from_embedding(embedding)


def build_model(num_classes: int = 7, embedding_dim: int = 128) -> nn.Module:
	"""Convenience factory for training scripts."""
	return EmotionCNN(num_classes=num_classes, embedding_dim=embedding_dim)
