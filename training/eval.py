"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Multi-Task Evaluation Metrics Engine
Author: Rashid, et al.
Institution: Sunway University, Malaysia

This module provides standard evaluation protocols for single-label
and multi-label tasks across the emotion, intention, and action spaces.
"""

from __future__ import annotations

import warnings
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    average_precision_score,
)

# Suppress warnings triggered when rare structural classes are absent from validation slices
warnings.filterwarnings("ignore", message="No positive class found in y_true.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics")


def evaluate_tritask(
    emotion_preds: torch.Tensor,
    intention_preds: torch.Tensor,
    action_preds: torch.Tensor,
    emotion_labels: torch.Tensor,
    intention_labels: torch.Tensor,
    action_labels: torch.Tensor,
    threshold: float = 0.4,
) -> dict[str, float]:
    """
    Computes rigorous multi-task performance metrics for joint model validation and testing.

    Args:
        emotion_preds: Categorical logits or predicted labels for emotion recognition.
        intention_preds: Binary logits or probabilities for multi-label intention mapping.
        action_preds: Binary logits or probabilities for multi-label action categorization.
        emotion_labels: Ground truth categorical indices for emotion.
        intention_labels: Ground truth binary target vectors for intention.
        action_labels: Ground truth binary target vectors for action.
        threshold: Positive classification threshold boundary for multi-label tasks.

    Returns:
        A dictionary tracking evaluation scores across all three target domains.
    """
    # Defensive validations to ensure tensor batch constraints match across tasks
    assert emotion_preds.size(0) == emotion_labels.size(0), "Emotion prediction and label batch sizes must align."
    assert intention_preds.size(0) == intention_labels.size(0), "Intention prediction and label batch sizes must align."
    assert action_preds.size(0) == action_labels.size(0), "Action prediction and label batch sizes must align."

    metrics: dict[str, float] = {}
    
    # -------------------------------------------------------------------------
    # 1. Emotion (Single-Label Classification)
    # -------------------------------------------------------------------------
    if emotion_preds.dim() > 1 and emotion_preds.size(1) > 1:
        emotion_preds = torch.argmax(emotion_preds, dim=1)
        
    emotion_preds_np = emotion_preds.cpu().numpy()
    emotion_labels_np = emotion_labels.cpu().numpy()
    
    metrics["emotion_accuracy"] = float(accuracy_score(emotion_labels_np, emotion_preds_np))
    metrics["emotion_macro_f1"] = float(f1_score(emotion_labels_np, emotion_preds_np, average="macro", zero_division=0))
    
    # -------------------------------------------------------------------------
    # 2. Intention (Multi-Label Classification)
    # -------------------------------------------------------------------------
    if intention_preds.min() < 0 or intention_preds.max() > 1:
        intention_probs = torch.sigmoid(intention_preds)
    else:
        intention_probs = intention_preds
        
    intention_probs_np = intention_probs.cpu().numpy()
    intention_binary_np = (intention_probs_np > threshold).astype(int)
    intention_labels_np = intention_labels.cpu().numpy()
    
    metrics["intention_macro_f1"] = float(f1_score(intention_labels_np, intention_binary_np, average="macro", zero_division=0))
    
    # Evaluate Precision-Recall curves via mean Average Precision (mAP) independent of operational threshold
    try:
        metrics["intention_mAP"] = float(average_precision_score(intention_labels_np, intention_probs_np, average="macro"))
    except ValueError:
        metrics["intention_mAP"] = 0.0
        
    # -------------------------------------------------------------------------
    # 3. Action (Multi-Label Classification)
    # -------------------------------------------------------------------------
    if action_preds.min() < 0 or action_preds.max() > 1:
        action_probs = torch.sigmoid(action_preds)
    else:
        action_probs = action_preds
        
    action_probs_np = action_probs.cpu().numpy()
    action_binary_np = (action_probs_np > threshold).astype(int)
    action_labels_np = action_labels.cpu().numpy()
    
    metrics["action_macro_f1"] = float(f1_score(action_labels_np, action_binary_np, average="macro", zero_division=0))
    
    try:
        metrics["action_mAP"] = float(average_precision_score(action_labels_np, action_probs_np, average="macro"))
    except ValueError:
        metrics["action_mAP"] = 0.0
        
    return metrics
