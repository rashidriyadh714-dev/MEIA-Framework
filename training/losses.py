"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Multi-Task Loss Engines
Author: Imad Gohar and Rashid Riyadh, et al.
Institution: Sunway University, Malaysia

This module implements the custom loss functions for the MEIA framework,
including a weighted multi-label focal loss for handling severe class 
imbalances in the intention and action distributions.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.nn as nn

class WeightedMultiLabelFocalLoss(nn.Module):
    """
    Fuses dynamic inverse class weights with Focal Loss modulating factors.
    Designed to prevent rare classes from vanishing into zero-gradients
    during multi-label optimization.
    """
    def __init__(self, pos_weight: torch.Tensor | None = None, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 1. Calculate standard BCE with dynamic inverse weights
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, 
            targets.float(), 
            pos_weight=self.pos_weight, 
            reduction="none"
        )
        
        # 2. Calculate the probability of the true class
        pt = torch.exp(-bce_loss)
        
        # 3. Apply Focal Modulator: alpha * (1 - pt)^gamma
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class TriTaskLossEngine(nn.Module):
    """
    Centralized Loss Engine for the MEIA pipeline.
    Combines Smoothed Cross-Entropy for single-label Emotion classification
    with Weighted Focal Loss for multi-label Intention and Action tasks.
    """
    def __init__(
        self,
        pos_weight_intent: torch.Tensor | None = None,
        pos_weight_action: torch.Tensor | None = None,
        emotion_weight: float = 1.0,
        intention_weight: float = 2.0, 
        action_weight: float = 2.0, 
    ):
        super().__init__()
        self.emotion_weight = emotion_weight
        self.intention_weight = intention_weight
        self.action_weight = action_weight

        # Label Smoothing mitigates overconfidence on unambiguous text/vision samples
        self.emotion_loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Weighted Focal Loss handles severe long-tail distribution imbalances
        self.intention_loss_fn = WeightedMultiLabelFocalLoss(pos_weight=pos_weight_intent, gamma=2.0)
        self.action_loss_fn = WeightedMultiLabelFocalLoss(pos_weight=pos_weight_action, gamma=2.0)

    def forward(
        self,
        emotion_logits: torch.Tensor,
        intention_logits: torch.Tensor,
        action_logits: torch.Tensor,
        emotion_labels: torch.Tensor,
        intention_labels: torch.Tensor,
        action_labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        emotion_loss = self.emotion_loss_fn(emotion_logits, emotion_labels)
        intention_loss = self.intention_loss_fn(intention_logits, intention_labels)
        action_loss = self.action_loss_fn(action_logits, action_labels)

        total_loss = (
            self.emotion_weight * emotion_loss
            + self.intention_weight * intention_loss
            + self.action_weight * action_loss
        )

        return {
            "total_loss": total_loss,
            "emotion_loss": emotion_loss,
            "intention_loss": intention_loss,
            "action_loss": action_loss,
        }
