#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Class-Wise Performance Analytics


This script utilizes scikit-learn to generate comprehensive classification 
reports across the tri-task taxonomy. It isolates boundary cases (best and 
worst performing classes) to inform ablation discussions.
"""

import sys
import logging
import torch
import warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.meia_architecture import MEIAModel
from data.multimodal_dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("MetricsReport")

EMOTION_NAMES = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Confused", "Shy"]
INTENTION_NAMES = ["Informing", "Seeking_Info", "Req_Help", "Complaining", "Agreeing", "Disagreeing", "Warning", "Greeting", "Apologizing", "Suggesting", "Gratitude", "Confusion"]
ACTION_NAMES = ["Still", "Standing", "Sitting", "Walking", "Running", "Pointing", "Typing", "Shouting", "Crying", "Smiling", "Holding", "Looking_Away", "Gesturing", "Waving", "Reading"]

def generate_report() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing Metrics Generator on device: {device}")

    logger.info("Mounting Evaluation Dataloader...")
    _, _, test_loader = get_dataloaders(
        batch_size=8, eval_batch_size=32, num_workers=4, distributed=False
    )

    logger.info("Mounting Optimal MEIA Checkpoint (Seed 42)...")
    model = MEIAModel(hidden_dim=1024).to(device)
    model_path = project_root / "checkpoints" / "seed_42" / "best_model.pt"

    if not model_path.exists():
        logger.error(f"Failed to locate checkpoint at {model_path}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    logger.info("Extracting Tri-Task Per-Class Metrics... (Approx. 60 seconds)")
    
    all_emo_preds, all_emo_labels = [], []
    all_int_preds, all_int_labels = [], []
    all_act_preds, all_act_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            images = batch.get("images")
            if images is not None: 
                images = images.to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                out = model(input_ids, attention_mask, images=images)

            emo_preds = torch.argmax(out["emotion_logits"], dim=1).cpu().numpy()
            int_preds = (torch.sigmoid(out["intention_logits"]).cpu().numpy() > 0.4).astype(int)
            act_preds = (torch.sigmoid(out["action_logits"]).cpu().numpy() > 0.4).astype(int)

            all_emo_preds.extend(emo_preds)
            all_emo_labels.extend(batch["emotion_labels"].cpu().numpy())
            
            all_int_preds.extend(int_preds)
            all_int_labels.extend(batch["intention_labels"].cpu().numpy())
            
            all_act_preds.extend(act_preds)
            all_act_labels.extend(batch["action_labels"].cpu().numpy())

    emo_labels_np = np.array(all_emo_labels)
    emo_preds_np = np.array(all_emo_preds)
    int_labels_np = np.array(all_int_labels)
    int_preds_np = np.array(all_int_preds)
    act_labels_np = np.array(all_act_labels)
    act_preds_np = np.array(all_act_preds)

    def print_extreme_cases(report_dict: dict, class_names: list, task_name: str) -> None:
        scores = []
        for cls in class_names:
            if cls in report_dict:
                scores.append((cls, report_dict[cls]['f1-score']))
                
        scores.sort(key=lambda x: x[1])
        
        logger.info(f"--- {task_name.upper()} BOUNDARY ANALYSIS ---")
        logger.info("  Bottom Performing Classes (Bottlenecks):")
        for name, f1 in scores[:4]:
            logger.info(f"    - {name:<15}: F1 = {f1:.4f}")
            
        logger.info("  Top Performing Classes (Robust):")
        for name, f1 in reversed(scores[-4:]):
            logger.info(f"    - {name:<15}: F1 = {f1:.4f}")
        logger.info("--------------------------------------------------\n")

    logger.info("\n======================================================================")
    logger.info(" MEIA CLASS-WISE PERFORMANCE REPORT")
    logger.info("======================================================================\n")

    emo_report = classification_report(
        emo_labels_np, emo_preds_np, 
        labels=list(range(len(EMOTION_NAMES))), target_names=EMOTION_NAMES, output_dict=True, zero_division=0
    )
    print_extreme_cases(emo_report, EMOTION_NAMES, "Emotion Recognition")

    int_report = classification_report(
        int_labels_np, int_preds_np, 
        labels=list(range(len(INTENTION_NAMES))), target_names=INTENTION_NAMES, output_dict=True, zero_division=0
    )
    print_extreme_cases(int_report, INTENTION_NAMES, "Intention Detection")

    act_report = classification_report(
        act_labels_np, act_preds_np, 
        labels=list(range(len(ACTION_NAMES))), target_names=ACTION_NAMES, output_dict=True, zero_division=0
    )
    print_extreme_cases(act_report, ACTION_NAMES, "Action Prediction")

if __name__ == "__main__":
    generate_report()
