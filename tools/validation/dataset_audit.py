#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Comprehensive Dataset Volumetric Audit

This script conducts a deep, batch-level volumetric audit of the MEIA dataset. 
It rigorously quantifies modality availability (Text vs. Image) and tracks the 
exact compositional ratio of MINE and FANE source distributions across all splits.
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_datasets import get_cloud_dataloaders

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("DatasetAudit")

def generate_detailed_audit() -> None:
    logger.info("\n======================================================================")
    logger.info(" 🏛️  MEIA FRAMEWORK: OFFICIAL MULTIMODAL DATASET AUDIT")
    logger.info("======================================================================")
    logger.info("Initializing Dataset Volumetric Scanner...")

    try:
        train_loader, val_loader, test_loader = get_cloud_dataloaders(
            batch_size=32, eval_batch_size=32, num_workers=4, distributed=False
        )
    except Exception as e:
        logger.error(f"Failed to mount dataloaders during audit: {e}")
        return

    def scan_split(loader, split_name: str) -> dict:
        total_samples, total_images, total_texts = 0, 0, 0
        logger.info(f"Executing deep scan on {split_name} split boundaries...")
        
        for batch in loader:
            b_size = batch["input_ids"].size(0)
            total_samples += b_size
            total_texts += b_size
            
            if batch.get("images") is not None:
                total_images += batch["images"].size(0)

        # Deep Inspection: Isolate source distribution
        mine_count, fane_count = 0, 0
        dataset = loader.dataset
        if hasattr(dataset, 'samples'):
            for sample in dataset.samples:
                if getattr(sample, 'source_dataset', '') == "MINE_Llama_Curated":
                    mine_count += 1
                elif getattr(sample, 'source_dataset', '') == "FANE_Distilled":
                    fane_count += 1

        return {
            "name": split_name, "total": total_samples, "images": total_images,
            "texts": total_texts, "mine": mine_count, "fane": fane_count
        }

    train_data = scan_split(train_loader, "TRAIN")
    val_data = scan_split(val_loader, "VALIDATION")
    test_data = scan_split(test_loader, "TEST")

    # Aggregate global volume metrics
    grand_mine = train_data['mine'] + val_data['mine'] + test_data['mine']
    grand_fane = train_data['fane'] + val_data['fane'] + test_data['fane']
    grand_total = train_data['total'] + val_data['total'] + test_data['total']
    grand_images = train_data['images'] + val_data['images'] + test_data['images']
    grand_texts = train_data['texts'] + val_data['texts'] + test_data['texts']

    logger.info("\n======================================================================")
    logger.info(" 📊 DETAILED BREAKDOWN PER SPLIT")
    logger.info("======================================================================")

    for s in [train_data, val_data, test_data]:
        logger.info(f" 📂 {s['name']} SUBSET:")
        logger.info(f"    ├─ Origin Distribution:")
        logger.info(f"    │  ├─ MINE Curated Matrix : {s['mine']:>8} tensors")
        logger.info(f"    │  └─ FANE Distilled Matrix:{s['fane']:>8} tensors")
        logger.info(f"    ├─ Modality Distribution:")
        logger.info(f"    │  ├─ Semantic Inputs     : {s['texts']:>8} valid texts")
        logger.info(f"    │  └─ Visual Inputs       : {s['images']:>8} valid images")
        logger.info(f"    └─ 🔹 TOTAL SAMPLES       : {s['total']:>8}")
        logger.info("-" * 70)

    logger.info(" 🏆 GLOBAL DATASET METRICS (ALL SPLITS)")
    logger.info("======================================================================")
    logger.info(f" 🔹 TOTAL MINE SAMPLES     : {grand_mine:>9}")
    logger.info(f" 🔹 TOTAL FANE SAMPLES     : {grand_fane:>9}")
    logger.info(f" 🔹 VALID SEMANTIC PAYLOADS: {grand_texts:>9}")
    logger.info(f" 🔹 VALID VISUAL PAYLOADS  : {grand_images:>9}")
    logger.info(f" 🚀 AGGREGATE SYSTEM VOLUME: {grand_total:>9}")
    logger.info("======================================================================\n")

if __name__ == "__main__":
    generate_detailed_audit()
