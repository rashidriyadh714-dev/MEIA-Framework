#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Modality Dependency Ablation Engine
Author: Imad,Rashid, et al.
Institution: Sunway University, Malaysia

This script conducts quantitative ablation studies on the MEIA architecture 
by selectively "blinding" the vision or text encoders. This isolates and 
measures the exact mathematical contribution of each modality to the final 
fused tri-task prediction.
"""

import sys
import torch
import warnings
import logging
from pathlib import Path
from transformers import AutoTokenizer
from sklearn.metrics import average_precision_score, accuracy_score, f1_score
from tqdm import tqdm

# Silence zero-division warnings for empty batch classes
warnings.filterwarnings("ignore")

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.meia_architecture import MEIAModel
from data.multimodal_dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AblationEngine")

def run_quantitative_ablation() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initialized Modality Ablation Engine on device: {device}")

    logger.info("Instantiating Evaluation Dataloader...")
    _, _, test_loader = get_dataloaders(
        batch_size=8, 
        eval_batch_size=32, 
        num_workers=4, 
        distributed=False
    )

    logger.info("Caching RoBERTa Tokenizer for Text Blinding...")
    hf_cache = project_root / "models" / "hf_hub"
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", cache_dir=str(hf_cache))
    
    # Pre-compute an empty text tensor mask to "blind" the RoBERTa pipeline
    blank_text = tokenizer("", return_tensors="pt").to(device)

    logger.info("Mounting Optimal MEIA Checkpoint (Seed 42)...")
    model = MEIAModel(hidden_dim=1024).to(device)
    model_path = project_root / "checkpoints" / "seed_42" / "best_model.pt"
    
    if not model_path.exists():
        logger.error(f"Optimal checkpoint not found at {model_path}. Exiting.")
        sys.exit(1)

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    def evaluate_condition(modality_name: str, blind_text: bool = False, blind_vision: bool = False) -> None:
        logger.info(f"Executing Forward Pass Condition: {modality_name}")
        
        all_emo_preds, all_emo_labels = [], []
        all_int_preds, all_int_probs, all_int_labels = [], [], []
        all_act_preds, all_act_probs, all_act_labels = [], [], []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {modality_name}", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                images = batch.get("images")
                if images is not None: 
                    images = images.to(device)

                # ==============================================================
                # MODALITY BLINDING LOGIC
                # ==============================================================
                if blind_text:
                    b_size = input_ids.size(0)
                    input_ids = blank_text["input_ids"].expand(b_size, -1).to(device)
                    attention_mask = blank_text["attention_mask"].expand(b_size, -1).to(device)
                if blind_vision:
                    images = None

                with torch.amp.autocast('cuda'):
                    out = model(input_ids, attention_mask, images=images)

                # 1. Emotion Metrics
                emo_preds = torch.argmax(out["emotion_logits"], dim=1).cpu().numpy()
                all_emo_preds.extend(emo_preds)
                all_emo_labels.extend(batch["emotion_labels"].cpu().numpy())

                # 2. Intention Metrics
                int_probs = torch.sigmoid(out["intention_logits"]).cpu().numpy()
                int_preds = (int_probs > 0.4).astype(int)
                all_int_probs.extend(int_probs)
                all_int_preds.extend(int_preds)
                all_int_labels.extend(batch["intention_labels"].cpu().numpy())

                # 3. Action Metrics
                act_probs = torch.sigmoid(out["action_logits"]).cpu().numpy()
                act_preds = (act_probs > 0.4).astype(int)
                all_act_probs.extend(act_probs)
                all_act_preds.extend(act_preds)
                all_act_labels.extend(batch["action_labels"].cpu().numpy())

        # ==============================================================
        # METRIC AGGREGATION
        # ==============================================================
        emo_acc = accuracy_score(all_emo_labels, all_emo_preds) * 100
        int_f1 = f1_score(all_int_labels, all_int_preds, average="macro", zero_division=0) * 100
        int_map = average_precision_score(all_int_labels, all_int_probs, average="macro") * 100
        act_f1 = f1_score(all_act_labels, all_act_preds, average="macro", zero_division=0) * 100
        act_map = average_precision_score(all_act_labels, all_act_probs, average="macro") * 100

        logger.info(f"--- Ablation Results: {modality_name} ---")
        logger.info(f"Emotion Task  -> Accuracy: {emo_acc:.2f}%")
        logger.info(f"Intention Task-> Macro F1: {int_f1:.2f}% | mAP: {int_map:.2f}%")
        logger.info(f"Action Task   -> Macro F1: {act_f1:.2f}% | mAP: {act_map:.2f}%")
        logger.info("-" * 50)

    # Execute isolated pipeline studies
    logger.info("Commencing Modality Dependency Study...")
    evaluate_condition("VISION ONLY (DINOv2 isolated, Text Blinded)", blind_text=True, blind_vision=False)
    evaluate_condition("TEXT ONLY (RoBERTa isolated, Vision Blinded)", blind_text=False, blind_vision=True)
    logger.info("Ablation Study Complete.")

if __name__ == "__main__":
    run_quantitative_ablation()
