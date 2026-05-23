#!/usr/bin/env python3
#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Distributed Cloud Training Pipeline (Scaling Baseline)
Author: Imad Gohar and Rashid Riyadh, et al.
Institution: Sunway University, Malaysia

This script handles the high-throughput distributed training loop across 
cloud instances. It serves as the unconstrained baseline to complement 
the VRAM-optimized main pipeline (train_meia.py).

Usage:
    python scripts/train_distributed.py \
        --output-dir checkpoints/distributed-training \
        --epochs 6 --batch-size 16 --seeds 41 42 43
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

from sklearn.metrics import accuracy_score, f1_score

from transformers import get_cosine_schedule_with_warmup
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()

sys.path.insert(0, str(Path(__file__).parent.parent))

# Updated imports to match the new repository structure
from data.multimodal_dataset import get_meia_dataloaders
from models.meia_architecture import MEIAModel

logger = logging.getLogger(__name__)

# ==============================================================================
# Dynamic Loss Engine
# ==============================================================================
class TriTaskLossEngine(nn.Module):
    """
    Multi-task loss engine implementing Dynamic Inverse Weighting for BCE to 
    mitigate long-tail imbalance on rare intentions and actions.
    """
    def __init__(
        self, 
        pos_weight_intent: torch.Tensor, 
        pos_weight_action: torch.Tensor, 
        emo_w: float = 1.0, 
        int_w: float = 2.0, 
        act_w: float = 2.0
    ):
        super().__init__()
        self.emo_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.int_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_intent)
        self.act_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_action)
        self.weights = (emo_w, int_w, act_w)

    def forward(
        self, 
        emotion_logits: torch.Tensor, 
        intention_logits: torch.Tensor, 
        action_logits: torch.Tensor, 
        emotion_labels: torch.Tensor, 
        intention_labels: torch.Tensor, 
        action_labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        
        l_emo = self.emo_loss(emotion_logits, emotion_labels)
        l_int = self.int_loss(intention_logits, intention_labels)
        l_act = self.act_loss(action_logits, action_labels)
        
        total = (self.weights[0] * l_emo) + (self.weights[1] * l_int) + (self.weights[2] * l_act)
        return {"total_loss": total, "emo_loss": l_emo, "int_loss": l_int, "act_loss": l_act}


def compute_dynamic_pos_weights(loader, device: torch.device, num_intent: int = 12, num_action: int = 15) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates inverse class weights over the dataset to balance BCE loss."""
    int_pos = torch.zeros(num_intent, device=device)
    act_pos = torch.zeros(num_action, device=device)
    total_samples = 0
    
    for batch in loader:
        int_pos += batch["intention_labels"].to(device).sum(dim=0)
        act_pos += batch["action_labels"].to(device).sum(dim=0)
        total_samples += batch["intention_labels"].size(0)
        
    int_neg = total_samples - int_pos
    act_neg = total_samples - act_pos
    
    pw_int = torch.clamp(int_neg / (int_pos + 1e-5), min=1.0, max=50.0)
    pw_act = torch.clamp(act_neg / (act_pos + 1e-5), min=1.0, max=50.0)
    return pw_int, pw_act

# ==============================================================================
# Infrastructure & Environment Configuration
# ==============================================================================
def setup_local_cache(repo_root: Path) -> None:
    """Enforce local repository paths for model weights to prevent environment leaks."""
    hf_hub_dir = (repo_root / "models" / "hf_hub").resolve()
    torch_hub_dir = (repo_root / "models" / "torch_hub").resolve()

    hf_hub_dir.mkdir(parents=True, exist_ok=True)
    torch_hub_dir.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(hf_hub_dir)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_hub_dir)
    os.environ["HF_DATASETS_CACHE"] = str(hf_hub_dir)
    os.environ["TORCH_HOME"] = str(torch_hub_dir)


def validate_dataset_paths(repo_root: Path) -> None:
    """Pre-flight check to ensure the curated dataset splits are available."""
    mine_curated_root = repo_root / "data" / "mine_curated"
    fane_root = repo_root / "data" / "fane"
    
    assert mine_curated_root.exists(), f"Missing curated MINE dataset at {mine_curated_root}"
    assert fane_root.exists(), f"Missing distilled FANE dataset at {fane_root}"


def setup_distributed() -> Tuple[int, int, torch.device]:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    if "RANK" not in os.environ:
        return 0, 1, torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    torch.cuda.set_device(local_rank)
    return rank, world_size, torch.device(f"cuda:{local_rank}")


def setup_logging(rank: int, output_dir: Path) -> None:
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(output_dir / "training.log"), logging.StreamHandler()],
        )
    else:
        logging.basicConfig(level=logging.WARNING)

# ==============================================================================
# Training & Evaluation Loops
# ==============================================================================
def train_one_epoch(
    model: nn.Module, 
    train_loader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer, 
    scheduler, 
    device: torch.device, 
    epoch: int, 
    fp16: bool = True, 
    grad_accum_steps: int = 4
) -> float:
    
    model.train()
    total_loss, num_batches = 0, 0
    scaler = torch.amp.GradScaler('cuda') if fp16 else None
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        emotion_labels = batch["emotion_labels"].to(device, non_blocking=True)
        intention_labels = batch["intention_labels"].to(device, non_blocking=True)
        action_labels = batch["action_labels"].to(device, non_blocking=True)
        
        images = batch.get("images")
        if images is not None: 
            images = images.to(device, non_blocking=True)
        
        if fp16:
            with torch.amp.autocast('cuda'):
                model_output = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
                loss_dict = criterion(
                    model_output["emotion_logits"], model_output["intention_logits"], model_output["action_logits"],
                    emotion_labels, intention_labels, action_labels
                )
                loss = loss_dict["total_loss"] / grad_accum_steps
                
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        else:
            model_output = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
            loss_dict = criterion(
                model_output["emotion_logits"], model_output["intention_logits"], model_output["action_logits"],
                emotion_labels, intention_labels, action_labels
            )
            loss = loss_dict["total_loss"] / grad_accum_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        
        total_loss += loss.item() * grad_accum_steps
        num_batches += 1
        
        if batch_idx % 100 == 0:
            logger.info(f"Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item() * grad_accum_steps:.4f}")
            
    return total_loss / max(num_batches, 1)


def evaluate_one_epoch(
    model: nn.Module, 
    val_loader, 
    criterion: nn.Module, 
    device: torch.device, 
    epoch: int
) -> tuple[float, dict]:
    
    model.eval()
    total_loss, num_batches = 0, 0
    all_emo_logits, all_int_logits, all_act_logits = [], [], []
    all_emo_labels, all_int_labels, all_act_labels = [], [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            emotion_labels = batch["emotion_labels"].to(device, non_blocking=True)
            intention_labels = batch["intention_labels"].to(device, non_blocking=True)
            action_labels = batch["action_labels"].to(device, non_blocking=True)
            
            images = batch.get("images")
            if images is not None: 
                images = images.to(device, non_blocking=True)
            
            model_output = model(input_ids=input_ids, attention_mask=attention_mask, images=images)
            loss_dict = criterion(
                model_output["emotion_logits"], model_output["intention_logits"], model_output["action_logits"],
                emotion_labels, intention_labels, action_labels
            )
            
            total_loss += loss_dict["total_loss"].item()
            num_batches += 1
            
            all_emo_logits.append(model_output["emotion_logits"].cpu())
            all_int_logits.append(model_output["intention_logits"].cpu())
            all_act_logits.append(model_output["action_logits"].cpu())
            
            all_emo_labels.append(emotion_labels.cpu())
            all_int_labels.append(intention_labels.cpu())
            all_act_labels.append(action_labels.cpu())
    
    avg_loss = total_loss / max(num_batches, 1)
    
    emotion_preds = torch.argmax(torch.cat(all_emo_logits), dim=1).numpy()
    emotion_targets = torch.cat(all_emo_labels).numpy()
    
    intention_probs = torch.sigmoid(torch.cat(all_int_logits)).numpy()
    action_probs = torch.sigmoid(torch.cat(all_act_logits)).numpy()
    intention_targets = torch.cat(all_int_labels).numpy()
    action_targets = torch.cat(all_act_labels).numpy()
    
    # Validation multi-label thresholding
    intention_preds = (intention_probs > 0.4).astype(int)
    action_preds = (action_probs > 0.4).astype(int)
    
    metrics = {
        "emotion_accuracy": accuracy_score(emotion_targets, emotion_preds),
        "intention_macro_f1": f1_score(intention_targets, intention_preds, average='macro', zero_division=0),
        "action_macro_f1": f1_score(action_targets, action_preds, average='macro', zero_division=0),
    }
    
    logger.info(
        f"Epoch {epoch} Validation Loss: {avg_loss:.4f} | "
        f"Emo Acc: {metrics['emotion_accuracy']:.4f} | "
        f"Int F1: {metrics['intention_macro_f1']:.4f} | "
        f"Act F1: {metrics['action_macro_f1']:.4f}"
    )
    return avg_loss, metrics

# ==============================================================================
# Seed Execution
# ==============================================================================
def run_seed(seed: int, config: dict, output_dir: Path, rank: int, world_size: int, device: torch.device) -> dict:
    logger.info(f"\n{'='*80}\nInitializing MEIA Pipeline: Seed {seed}\n{'='*80}")
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(seed)
    
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader, test_loader = get_meia_dataloaders(
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        distributed=(world_size > 1)
    )

    if rank == 0: 
        logger.info("Computing Inverse Class Weights for BCE Loss Engine...")
        
    pw_int, pw_act = compute_dynamic_pos_weights(train_loader, device)

    criterion = TriTaskLossEngine(
        pos_weight_intent=pw_int, pos_weight_action=pw_act,
        emo_w=1.0, int_w=2.0, act_w=2.0
    )

    model = MEIAModel(hidden_dim=config.get("hidden_dim", 1024)).to(device)
    if world_size > 1: 
        model = DDP(model, device_ids=[device.index])
    
    optimizer = AdamW(model.parameters(), lr=config.get("learning_rate", 3e-5), weight_decay=config.get("weight_decay", 0.05))
    total_steps = len(train_loader) * config.get("epochs", 6)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.15), num_training_steps=total_steps)
    
    best_val_loss, best_epoch, patience_counter = float("inf"), -1, 0
    patience = config.get("early_stopping_patience", 2)
    
    seed_metrics = {"seed": seed, "train_losses": [], "val_losses": [], "emotion_accuracies": [], "intention_f1s": [], "action_f1s": []}
    
    for epoch in range(1, config.get("epochs", 6) + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch, 
            config.get("fp16", True), config.get("grad_accum_steps", 4)
        )
        val_loss, metrics = evaluate_one_epoch(model, val_loader, criterion, device, epoch)
        
        seed_metrics["train_losses"].append(train_loss)
        seed_metrics["val_losses"].append(val_loss)
        seed_metrics["emotion_accuracies"].append(metrics["emotion_accuracy"])
        seed_metrics["intention_f1s"].append(metrics["intention_macro_f1"])
        seed_metrics["action_f1s"].append(metrics["action_macro_f1"])
        
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            if rank == 0:
                torch.save((model.module if isinstance(model, DDP) else model).state_dict(), seed_dir / "best_model.pt")
                logger.info(f"Saved optimized model weights at epoch {epoch}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if rank == 0: 
                    logger.info("Early stopping threshold reached. Terminating training.")
                break

    if (seed_dir / "best_model.pt").exists():
        (model.module if isinstance(model, DDP) else model).load_state_dict(torch.load(seed_dir / "best_model.pt", map_location=device))

    logger.info(f"Evaluating Model Weights on Unseen Test Partition...")
    test_loss, test_metrics = evaluate_one_epoch(model, test_loader, criterion, device, best_epoch)
    
    seed_metrics.update({
        "test_loss": test_loss, 
        "test_emotion_accuracy": test_metrics["emotion_accuracy"],
        "test_intention_f1": test_metrics["intention_macro_f1"], 
        "test_action_f1": test_metrics["action_macro_f1"],
        "best_epoch": best_epoch, 
        "best_val_loss": best_val_loss
    })
    
    if rank == 0:
        with open(seed_dir / "metrics.json", "w") as f: 
            json.dump(seed_metrics, f, indent=2)
            
        logger.info(f"\n[Seed {seed}] Final Test Evaluation (Weights from Epoch {best_epoch}):")
        logger.info(
            f"  Emotion Acc: {test_metrics['emotion_accuracy']:.4f} | "
            f"Intention F1: {test_metrics['intention_macro_f1']:.4f} | "
            f"Action F1: {test_metrics['action_macro_f1']:.4f}\n"
        )
    
    return seed_metrics

# ==============================================================================
# Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MEIA Framework Distributed Training")
    parser.add_argument("--config", type=str, default="configs/meia_training.json")
    parser.add_argument("--output-dir", type=str, default="checkpoints/meia-training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()
    
    rank, world_size, device = setup_distributed()
    output_dir = Path(args.output_dir)
    config = json.load(open(args.config)) if args.config and Path(args.config).exists() else {}
    
    config.update({k: v for k, v in vars(args).items() if v is not None and k not in ["config", "output_dir"]})
    config.setdefault("epochs", 6)
    config.setdefault("batch_size", 16)
    config.setdefault("seeds", [41, 42, 43])
    config.setdefault("learning_rate", 3e-5)
    config.setdefault("grad_accum_steps", 4)
    config.setdefault("early_stopping_patience", 2)

    repo_root = Path.cwd().resolve()
    setup_local_cache(repo_root)
    validate_dataset_paths(repo_root)
    setup_logging(rank, output_dir)
    
    if rank == 0:
        logger.info(f"\n{'='*80}\nMEIA Framework Training Pipeline Initialization\n{'='*80}")

    all_seed_metrics = {str(seed): run_seed(seed, config, output_dir, rank, world_size, device) for seed in config["seeds"]}
    
    if rank == 0:
        test_emo = [m["test_emotion_accuracy"] for m in all_seed_metrics.values()]
        test_int = [m["test_intention_f1"] for m in all_seed_metrics.values()]
        test_act = [m["test_action_f1"] for m in all_seed_metrics.values()]
        
        summary = {
            "config": config,
            "test_emotion_accuracy_mean": float(np.mean(test_emo)), "test_emotion_accuracy_std": float(np.std(test_emo)),
            "test_intention_f1_mean": float(np.mean(test_int)), "test_intention_f1_std": float(np.std(test_int)),
            "test_action_f1_mean": float(np.mean(test_act)), "test_action_f1_std": float(np.std(test_act))
        }
        with open(output_dir / "summary.json", "w") as f: 
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*80}\nTraining Execution Complete: Aggregated Results\n{'='*80}")
        logger.info(f"Test Emo Acc:  {summary['test_emotion_accuracy_mean']:.4f} ± {summary['test_emotion_accuracy_std']:.4f}")
        logger.info(f"Test Int F1:   {summary['test_intention_f1_mean']:.4f} ± {summary['test_intention_f1_std']:.4f}")
        logger.info(f"Test Act F1:   {summary['test_action_f1_mean']:.4f} ± {summary['test_action_f1_std']:.4f}")
        logger.info(f"{'='*80}\n")

    if world_size > 1: 
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
