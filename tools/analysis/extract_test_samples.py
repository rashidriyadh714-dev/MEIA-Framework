#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Qualitative Modality Ablation Extraction


Extracts sample inferences from the test set under normal and blinded 
modality conditions. Generates formatted markdown tables for qualitative 
manuscript analysis.
"""

import sys
import logging
import torch
import torchvision
from pathlib import Path
from transformers import AutoTokenizer

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.meia_architecture import MEIAModel
from data.multimodal_dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QualitativeAblation")

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Confused", "Shy"]
INTENTIONS = ["Informing", "Seeking_Info", "Req_Help", "Complaining", "Agreeing", "Disagreeing", "Warning", "Greeting", "Apologizing", "Suggesting", "Gratitude", "Confusion"]
ACTIONS = ["Still", "Standing", "Sitting", "Walking", "Running", "Pointing", "Typing", "Shouting", "Crying", "Smiling", "Holding", "Looking_Away", "Gesturing", "Waving", "Reading"]

def get_predictions(out: dict) -> tuple:
    emo_probs = torch.softmax(out["emotion_logits"], dim=1)[0]
    int_probs = torch.sigmoid(out["intention_logits"])[0]
    act_probs = torch.sigmoid(out["action_logits"])[0]

    best_emo_idx = torch.argmax(emo_probs).item()
    pred_emo = f"{EMOTIONS[best_emo_idx]} ({emo_probs[best_emo_idx].item() * 100:.1f}%)"

    pred_ints = [f"{INTENTIONS[i]} ({int_probs[i].item() * 100:.1f}%)" for i in range(len(INTENTIONS)) if int_probs[i] > 0.4]
    pred_acts = [f"{ACTIONS[i]} ({act_probs[i].item() * 100:.1f}%)" for i in range(len(ACTIONS)) if act_probs[i] > 0.4]

    int_str = ", ".join(pred_ints) if pred_ints else "None"
    act_str = ", ".join(pred_acts) if pred_acts else "None"

    return pred_emo, int_str, act_str

def generate_ablation_samples() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initializing Qualitative Ablation Engine on device: {device}")

    output_dir = project_root / "outputs" / "qualitative"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Mounting Evaluation Dataloader...")
    _, _, test_loader = get_dataloaders(
        batch_size=1, eval_batch_size=1, num_workers=2, distributed=False
    )

    logger.info("Loading Semantic Tokenizer...")
    hf_cache = project_root / "models" / "hf_hub"
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", cache_dir=str(hf_cache))
    blank_text = tokenizer("", return_tensors="pt").to(device)

    logger.info("Loading Model Checkpoint...")
    model = MEIAModel(hidden_dim=1024).to(device)
    model_path = project_root / "checkpoints" / "seed_42" / "best_model.pt"
    
    if not model_path.exists():
        logger.error(f"Checkpoint not found at {model_path}.")
        return

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    logger.info("Processing test samples for ablation extraction...")

    doc_path = output_dir / "Ablation_Evaluation_Document.md"
    
    with open(doc_path, "w", encoding="utf-8") as doc, torch.no_grad():
        doc.write("# Multimodal Ablation Study (Test Set)\n")
        doc.write("Comparative analysis of full multimodal fusion against blinded single-modality baselines.\n\n")

        samples_saved = 0
        
        for batch_idx, batch in enumerate(test_loader):
            images = batch.get("images")
            if images is None:
                continue 

            images = images.to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            
            input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)

            emo_lbl = EMOTIONS[batch["emotion_labels"].item()]
            int_lbls = [INTENTIONS[i] for i, val in enumerate(batch["intention_labels"][0]) if val == 1]
            act_lbls = [ACTIONS[i] for i, val in enumerate(batch["action_labels"][0]) if val == 1]

            # Modality Testing
            out_full = model(input_ids, attention_mask, images=images)
            emo_full, int_full, act_full = get_predictions(out_full)

            out_img = model(blank_text["input_ids"], blank_text["attention_mask"], images=images)
            emo_img, int_img, act_img = get_predictions(out_img)

            out_txt = model(input_ids, attention_mask, images=None)
            emo_txt, int_txt, act_txt = get_predictions(out_txt)

            img_filename = f"ablation_sample_{samples_saved + 1}.png"
            torchvision.utils.save_image(images[0], output_dir / img_filename, normalize=True)

            # Generate formal markdown output
            doc.write(f"## Sample {samples_saved + 1} (Reference: `{img_filename}`)\n")
            doc.write(f"**Semantic Input:** \"{input_text}\"\n\n")
            
            doc.write(f"- **Ground Truth Emotion:** {emo_lbl}\n")
            doc.write(f"- **Ground Truth Intentions:** {', '.join(int_lbls) if int_lbls else 'None'}\n")
            doc.write(f"- **Ground Truth Actions:** {', '.join(act_lbls) if act_lbls else 'None'}\n\n")

            doc.write("| Setup | Emotion Prediction | Intention Predictions | Action Predictions |\n")
            doc.write("| :--- | :--- | :--- | :--- |\n")
            doc.write(f"| Fusion (Image + Text) | {emo_full} | {int_full} | {act_full} |\n")
            doc.write(f"| Vision-Only (Blinded) | {emo_img} | {int_img} | {act_img} |\n")
            doc.write(f"| Text-Only (Blinded) | {emo_txt} | {int_txt} | {act_txt} |\n\n")
            doc.write("---\n\n")

            samples_saved += 1
            if samples_saved >= 8:
                break

    logger.info(f"Extraction complete. Results mapped to: {output_dir}")
