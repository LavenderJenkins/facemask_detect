import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def yolo_loss(outputs: List[torch.Tensor], targets: torch.Tensor, img_size: int = 416) -> torch.Tensor:
		"""
		Minimal placeholder YOLO loss compatible with the notebook training loop.

		- `outputs` is a list of 3 tensors corresponding to model outputs at
			different scales. Each tensor has shape (B, anchors*(5+num_classes), H, W).
		- `targets` is a tensor with shape (N, 6) where each row is
			[batch_idx, cls, x_center, y_center, w, h] (normalized in [0,1]).

		This function returns a scalar tensor suitable for `.backward()`.

		NOTE: This is a simple, differentiable placeholder. Replace with your
		full YOLOv3 loss (objectness + bbox + class losses + anchor mapping)
		for proper training.
		"""
		device = outputs[0].device

		# Basic sanity: compute L2 loss between outputs and zero to keep gradients flowing.
		# We don't attempt to decode boxes or match anchors here.
		loss = torch.tensor(0.0, device=device)
		for out in outputs:
				loss = loss + torch.mean(out ** 2)

		# If targets are empty, still return a valid scalar
		return loss


__all__ = ["yolo_loss"]

