#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Core Test Set Evaluation Engine
Author: Rashid, et al.
Institution: Sunway University, Malaysia

This script executes a strict inference pass on the unseen test set using 
pre-trained weights. It utilizes the standardized `evaluate_tritask` logic 
to ensure metric parity with the training pipeline.
"""

import sys
import json
import logging
import torch
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.meia_architecture import MEIAModel
from data.multimodal_dataset import get_dataloaders
from training.eval import evaluate_tritask

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TestEvaluator")

def evaluate_safely() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Allocating computation to: {device}")

    # Enforce relative pathing for environment portability
    config_path = project_root / "configs" / "meia_config.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except FileNotFoundError:
        logger.warning(f"Configuration not found at {config_path}. Falling back to default hyper-parameters.")
        config = {"batch_size": 8, "eval_batch_size": 32, "num_workers": 4, "hidden_dim": 1024}

    logger.info("Initializing Test Set Dataloader...")
    _, _, test_loader = get_dataloaders(
        batch_size=config.get("batch_size", 8),
        eval_batch_size=config.get("eval_batch_size", 32),
        num_workers=config.get("num_workers", 4),
        distributed=False
    )

    logger.info("Mounting MEIA Architecture...")
    model = MEIAModel(hidden_dim=config.get("hidden_dim", 1024)).to(device)

    weights_path = project_root / "checkpoints" / "seed_42" / "best_model.pt"
    
    if not weights_path.exists():
        logger.error(f"Failed to locate trained checkpoint at {weights_path}. Exiting.")
        sys.exit(1)

    logger.info(f"Injecting learned weights from: {weights_path.name}")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval() 
    
    all_emo_preds, all_emo_labels = [], []
    all_int_preds, all_int_labels = [], []
    all_act_preds, all_act_labels = [], []

    logger.info("Executing Gradient-Free Forward Pass Protocol...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing Inference Loop"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            emotion_labels = batch["emotion_labels"].to(device)
            intention_labels = batch["intention_labels"].to(device)
            action_labels = batch["action_labels"].to(device)
            
            images = batch.get("images")
            if images is not None: 
                images = images.to(device)

            with torch.amp.autocast('cuda'):
                outputs = model(input_ids, attention_mask, images=images)
            
            all_emo_preds.append(outputs["emotion_logits"].cpu())
            all_emo_labels.append(emotion_labels.cpu())
            all_int_preds.append(outputs["intention_logits"].cpu())
            all_int_labels.append(intention_labels.cpu())
            all_act_preds.append(outputs["action_logits"].cpu())
            all_act_labels.append(action_labels.cpu())

    # Aggregate and evaluate using the standardized evaluation logic
    metrics = evaluate_tritask(
        emotion_preds=torch.cat(all_emo_preds, dim=0), 
        intention_preds=torch.cat(all_int_preds, dim=0), 
        action_preds=torch.cat(all_act_preds, dim=0),
        emotion_labels=torch.cat(all_emo_labels, dim=0), 
        intention_labels=torch.cat(all_int_labels, dim=0), 
        action_labels=torch.cat(all_act_labels, dim=0),
    )

    logger.info("======================================================================")
    logger.info(" 🏆 BMVC 2026: FINAL TEST SET METRICS (SEED 42)")
    logger.info("======================================================================")
    logger.info(f"EMOTION  -> Accuracy: {metrics['emotion_accuracy'] * 100:.2f}% | Macro F1: {metrics['emotion_macro_f1'] * 100:.2f}%")
    logger.info(f"INTENTION-> mAP:      {metrics['intention_mAP'] * 100:.2f}%")
    logger.info(f"ACTION   -> mAP:      {metrics['action_mAP'] * 100:.2f}%")
    logger.info("======================================================================")

if __name__ == "__main__":
    evaluate_safely()
