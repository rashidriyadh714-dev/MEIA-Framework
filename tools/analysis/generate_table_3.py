#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Manuscript Table Generator (Class Distribution)


Automates the extraction of absolute class-wise volumetric distributions 
across the combined dataset to generate manuscript-ready tables.
"""

import sys
import logging
import numpy as np
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.multimodal_dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("TableGenerator")

EMOTION_CLASSES = [
    "Angry", "Disgust", "Fear", "Happy", "Neutral", 
    "Sad", "Surprise", "Confused", "Shy"
]

INTENTION_CLASSES = [
    "Informing/Stating", "Seeking Information", "Requesting Help", "Complaining", 
    "Agreeing", "Disagreeing", "Warning", "Greeting", 
    "Apologizing", "Suggesting", "Expressing Gratitude", "Expressing Confusion"
]

ACTION_CLASSES = [
    "No Action/Still", "Standing", "Sitting", "Walking", "Running", 
    "Pointing", "Typing/Texting", "Shouting/Yelling", "Crying", "Smiling/Laughing", 
    "Holding an Object", "Looking Away", "Gesturing", "Waving", "Reading/Examining"
]

def get_dataloader_totals(dataloader) -> tuple:
    emo_counts = np.zeros(len(EMOTION_CLASSES), dtype=int)
    int_counts = np.zeros(len(INTENTION_CLASSES), dtype=int)
    act_counts = np.zeros(len(ACTION_CLASSES), dtype=int)

    for batch in dataloader:
        emos = batch["emotion_labels"].cpu().numpy()
        for e in emos:
            if 0 <= e < len(EMOTION_CLASSES):
                emo_counts[e] += 1

        ints = batch["intention_labels"].cpu().numpy()
        int_counts += ints.sum(axis=0).astype(int)

        acts = batch["action_labels"].cpu().numpy()
        act_counts += acts.sum(axis=0).astype(int)

    return emo_counts, int_counts, act_counts

def generate_perfect_table_3() -> None:
    logger.info("Initializing dataset distribution scanner...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=32, eval_batch_size=32, num_workers=4, distributed=False
    )

    logger.info("Aggregating Training Data...")
    tr_emo, tr_int, tr_act = get_dataloader_totals(train_loader)
    
    logger.info("Aggregating Validation Data...")
    val_emo, val_int, val_act = get_dataloader_totals(val_loader)
    
    logger.info("Aggregating Testing Data...")
    te_emo, te_int, te_act = get_dataloader_totals(test_loader)

    total_emo = tr_emo + val_emo + te_emo
    total_int = tr_int + val_int + te_int
    total_act = tr_act + val_act + te_act

    logger.info("\n============================================================")
    logger.info(" TABLE 3: CLASS-WISE DISTRIBUTION (GRAND TOTALS)")
    logger.info("============================================================")
    logger.info(f"{'Task':<15} {'ID':<5} {'Class':<25} {'Samples':>10}")
    logger.info("-" * 60)

    logger.info("Emotion")
    for i, name in enumerate(EMOTION_CLASSES):
        logger.info(f"{'':<15} {i+1:<5} {name:<25} {total_emo[i]:>10,}")
    logger.info("-" * 60)

    logger.info("Intention")
    for i, name in enumerate(INTENTION_CLASSES):
        logger.info(f"{'':<15} {i+1:<5} {name:<25} {total_int[i]:>10,}")
    logger.info("-" * 60)

    logger.info("Action")
    for i, name in enumerate(ACTION_CLASSES):
        logger.info(f"{'':<15} {i+1:<5} {name:<25} {total_act[i]:>10,}")
    logger.info("============================================================\n")

if __name__ == "__main__":
    generate_perfect_table_3()
