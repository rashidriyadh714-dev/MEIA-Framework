#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Step-by-Step Tensor Walkthrough

This diagnostic script executes a singular forward pass on the CPU, outputting 
the exact dimensional transformations and human-readable interpretations of the 
multimodal fusion process. Useful for methodology diagram validation.
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

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("WalkthroughEngine")

# Taxonomies
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise", "Confused", "Shy"]
INTENTIONS = ["Informing", "Seeking_Info", "Req_Help", "Complaining", "Agreeing", "Disagreeing", "Warning", "Greeting", "Apologizing", "Suggesting", "Gratitude", "Confusion"]
ACTIONS = ["Still", "Standing", "Sitting", "Walking", "Running", "Pointing", "Typing", "Shouting", "Crying", "Smiling", "Holding", "Looking_Away", "Gesturing", "Waving", "Reading"]

def formal_log_step(step_id: str, description: str, data_shape: str, interpretation: str) -> None:
    logger.info("----------------------------------------------------------------------")
    logger.info(f" STEP {step_id}: {description}")
    logger.info("----------------------------------------------------------------------")
    logger.info(f" Tensor Dimension : {data_shape}")
    logger.info(f" Interpretation   : {interpretation}")
    logger.info("")

def step_by_step_walkthrough() -> None:
    device = torch.device("cpu")
    logger.info(f"Initializing MEIA Walkthrough Engine (Device: {device})...")
    
    output_dir = project_root / "outputs" / "diagnostics"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Mounting Evaluation Dataloader (Single Sample)...")
    _, _, test_loader = get_dataloaders(
        batch_size=1, eval_batch_size=1, num_workers=0, distributed=False
    )
    
    logger.info("Mounting RoBERTa Tokenizer...")
    hf_cache = project_root / "models" / "hf_hub"
    tokenizer = AutoTokenizer.from_pretrained("roberta-large", cache_dir=str(hf_cache))

    logger.info("Mounting Pre-Trained MEIA Architecture...")
    model = MEIAModel(hidden_dim=1024).to(device)
    model_path = project_root / "checkpoints" / "seed_42" / "best_model.pt"
    
    if not model_path.exists():
        logger.error(f"Checkpoint not found at {model_path}.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    batch = next(iter(test_loader))
    images = batch.get("images")
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    if images is None: 
        logger.error("No image found in sample.")
        return

    logger.info("\n======================================================================")
    logger.info(" FORWARD PASS DIAGNOSTIC: LAYER-BY-LAYER TRACE")
    logger.info("======================================================================")

    # Phase 1: Inputs
    input_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    img_filename = output_dir / "walkthrough_input.png"
    torchvision.utils.save_image(images[0], img_filename, normalize=True)
    
    logger.info("RAW MULTIMODAL INPUTS:")
    logger.info(f" Text Semantic : \"{input_text}\"")
    logger.info(f" Visual Tensor : Saved to {img_filename}\n")

    formal_log_step("1", "Multimodal Input Loading", str(list(images.shape)), 
                    "Initial image tensor consisting of 1 batch, 3 color channels, scaled to 224x224 pixels.")

    # Phase 2: Extractors
    with torch.no_grad():
        text_out = model.text_backbone.roberta(input_ids=input_ids, attention_mask=attention_mask)
        text_cls = text_out.last_hidden_state[:, 0, :] 
        
        formal_log_step("2A", "RoBERTa-Large Extraction", str(list(text_cls.shape)),
                        "RoBERTa computes a 1024-dimensional CLS token summarizing semantic structure.")

        img_feats = model.vision_backbone(images)
        
        formal_log_step("2B", "DINOv2 ViT Extraction", str(list(img_feats.shape)),
                        "DINOv2 computes a 768-dimensional CLS token summarizing structural visual patterns.")

    # Phase 3: Alignment
    with torch.no_grad():
        text_embed = model.text_encoder(text_cls)
        image_embed = model.image_encoder(img_feats)
        
        formal_log_step("3", "Dimensionality Alignment", str(list(text_embed.shape)),
                        "Linear projection aligns both modalities into a shared 1024-dimensional latent space.")

    # Phase 4: Fusion
    with torch.no_grad():
        modalities = [text_embed, image_embed, torch.zeros_like(text_embed), torch.zeros_like(text_embed)]
        reliability_scores, uncertainties = model.reliability_module(modalities)
        
        text_rel = reliability_scores[0, 0].item() * 100
        image_rel = reliability_scores[0, 1].item() * 100
        
        formal_log_step("4A", "Reliability Scoring Module", str(list(reliability_scores.shape)),
                        f"Dynamic weight allocation computed: Text ({text_rel:.1f}%), Image ({image_rel:.1f}%).")

        fused_embed = model.fusion(modalities, reliability_scores)
        
        formal_log_step("4B", "Dual-Layer Cross-Attention", str(list(fused_embed.shape)),
                        "Modalities are synthesized into a unified 1024-dimensional representation based on reliability weights.")

    # Phase 5: Task Splitting
    with torch.no_grad():
        shared = model.task_heads.shared_encoder(fused_embed)
        emo_logits = model.task_heads.emotion_head(shared)
        int_logits = model.task_heads.intention_head(shared)
        act_logits = model.task_heads.action_head(shared)
        
        shapes = f"Emo: {list(emo_logits.shape)}, Int: {list(int_logits.shape)}, Act: {list(act_logits.shape)}"
        formal_log_step("5", "Tri-Task Output Logits", shapes,
                        "The shared representation is projected across three independent classification heads.")

    # Phase 6: Final Decoding
    with torch.no_grad():
        temperature = torch.clamp(model.temperature, min=1e-3)
        emo_scaled = emo_logits / temperature
        
        emo_probs = torch.softmax(emo_scaled, dim=1)[0]
        int_probs = torch.sigmoid(int_logits)[0]
        act_probs = torch.sigmoid(act_logits)[0]

        best_emo_idx = torch.argmax(emo_probs).item()
        pred_emo = EMOTIONS[best_emo_idx]
        emo_conf = emo_probs[best_emo_idx].item() * 100

        pred_ints = [INTENTIONS[i] for i in range(len(INTENTIONS)) if int_probs[i] > 0.4]
        pred_acts = [ACTIONS[i] for i in range(len(ACTIONS)) if act_probs[i] > 0.4]

    logger.info("======================================================================")
    logger.info(" FORWARD PASS COMPLETE: FINAL CLASSIFICATION OUTPUT")
    logger.info("======================================================================")
    logger.info(f" Emotion   : {pred_emo} (Confidence: {emo_conf:.1f}%)")
    logger.info(f" Intention : {', '.join(pred_ints) if pred_ints else 'None'}")
    logger.info(f" Action    : {', '.join(pred_acts) if pred_acts else 'None'}")
    logger.info("======================================================================\n")

if __name__ == "__main__":
    step_by_step_walkthrough()
