#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Qualitative Success Extractor

This script dynamically traverses the evaluation subset to extract physical 
image instances representing True Positive (perfect) classifications across 
all three taxonomies for manuscript visualization.
"""

import sys
import logging
import torch
import torchvision
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.meia_architecture import MEIAModel
from data.multimodal_dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SuccessExtractor")

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Confused", "Shy"]
INTENTIONS = ["Informing", "Seeking_Info", "Req_Help", "Complaining", "Agreeing", "Disagreeing", "Warning", "Greeting", "Apologizing", "Suggesting", "Gratitude", "Confusion"]
ACTIONS = ["Still", "Standing", "Sitting", "Walking", "Running", "Pointing", "Typing", "Shouting", "Crying", "Smiling", "Holding", "Looking_Away", "Gesturing", "Waving", "Reading"]

def sanitize(name: str) -> str:
    return str(name).replace("/", "-").replace(" ", "_")

def save_success_images() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing Qualitative Success Extractor on: {device}")

    output_dir = project_root / "outputs" / "success_analysis"
    for category in ["Emotions", "Intentions", "Actions"]:
        (output_dir / category).mkdir(parents=True, exist_ok=True)

    logger.info("Mounting Evaluation Dataloader (Batch Size: 1)...")
    _, _, test_loader = get_dataloaders(
        batch_size=1, eval_batch_size=1, num_workers=2, distributed=False
    )

    logger.info("Mounting Pre-Trained MEIA Architecture...")
    model = MEIAModel(hidden_dim=1024).to(device)
    model_path = project_root / "checkpoints" / "seed_42" / "best_model.pt"
    
    if not model_path.exists():
        logger.error(f"Checkpoint not found at {model_path}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    MAX_SAVES = 5
    saved_counts = {
        "Emotions": {name: 0 for name in EMOTIONS},
        "Intentions": {name: 0 for name in INTENTIONS},
        "Actions": {name: 0 for name in ACTIONS}
    }

    logger.info("Commencing True Positive extraction protocol...")

    total_saved = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            images = batch.get("images")
            if images is None:
                continue 

            images = images.to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            emo_lbl = batch["emotion_labels"].item()
            int_lbls = batch["intention_labels"][0].cpu().numpy()
            act_lbls = batch["action_labels"][0].cpu().numpy()

            with torch.amp.autocast('cuda'):
                out = model(input_ids, attention_mask, images=images)

            emo_pred = torch.argmax(out["emotion_logits"], dim=1).item()
            int_preds = (torch.sigmoid(out["intention_logits"]) > 0.4)[0].cpu().numpy()
            act_preds = (torch.sigmoid(out["action_logits"]) > 0.4)[0].cpu().numpy()

            if emo_pred == emo_lbl and emo_lbl < len(EMOTIONS):
                true_name = EMOTIONS[emo_lbl]
                if saved_counts["Emotions"][true_name] < MAX_SAVES:
                    filename = output_dir / "Emotions" / f"TruePositive_{sanitize(true_name)}_{batch_idx}.png"
                    torchvision.utils.save_image(images[0], filename, normalize=True)
                    saved_counts["Emotions"][true_name] += 1
                    total_saved += 1

            for i, (true_val, pred_val) in enumerate(zip(int_lbls, int_preds)):
                if true_val == 1 and pred_val == 1 and i < len(INTENTIONS):
                    name = INTENTIONS[i]
                    if saved_counts["Intentions"][name] < MAX_SAVES:
                        filename = output_dir / "Intentions" / f"TruePositive_{sanitize(name)}_{batch_idx}.png"
                        torchvision.utils.save_image(images[0], filename, normalize=True)
                        saved_counts["Intentions"][name] += 1
                        total_saved += 1

            for i, (true_val, pred_val) in enumerate(zip(act_lbls, act_preds)):
                if true_val == 1 and pred_val == 1 and i < len(ACTIONS):
                    name = ACTIONS[i]
                    if saved_counts["Actions"][name] < MAX_SAVES:
                        filename = output_dir / "Actions" / f"TruePositive_{sanitize(name)}_{batch_idx}.png"
                        torchvision.utils.save_image(images[0], filename, normalize=True)
                        saved_counts["Actions"][name] += 1
                        total_saved += 1

            if batch_idx > 1500:
                break
                
            if batch_idx % 100 == 0 and batch_idx > 0:
                logger.info(f"Processed {batch_idx} iterations. Captured {total_saved} instances.")

    logger.info(f"Extraction execution finalized. Isolated {total_saved} True Positive spatial tensors.")

if __name__ == "__main__":
    save_success_images()
