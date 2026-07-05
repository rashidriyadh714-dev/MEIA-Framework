#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Cross-Split Data Leakage Detection

This script enforces dataset integrity by scanning for identical image instances 
and subject-level sequence overlaps (flipbook leaks) across the training and 
testing boundaries.
"""

import sys
import re
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.cloud_datasets import get_cloud_dataloaders

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("LeakageAudit")


def extract_image_paths(dataloader) -> list:
    """Extracts all image paths directly from the MultimodalSample objects."""
    paths = []
    dataset = dataloader.dataset
    
    for sample in dataset.samples:
        if sample.image_path:
            paths.append(str(sample.image_path))
            
    return paths


def run_leakage_check() -> None:
    logger.info("Initializing dataloaders for cross-split leakage analysis...")
    train_loader, _, test_loader = get_cloud_dataloaders(
        batch_size=1, eval_batch_size=1, num_workers=4, distributed=False, sources=["mine_curated", "fane"]
    )

    logger.info("Extracting file identifiers from the Training Split...")
    train_paths = extract_image_paths(train_loader)
    
    logger.info("Extracting file identifiers from the Testing Split...")
    test_paths = extract_image_paths(test_loader)

    if not train_paths or not test_paths:
        logger.error("Failed to extract image paths. Ensure dataset.samples is correctly instantiated.")
        sys.exit(1)

    train_set = set(train_paths)
    test_set = set(test_paths)

    # 1. Strict Instance Overlap Check
    exact_duplicates = train_set.intersection(test_set)
    
    logger.info("======================================================================")
    logger.info(" DATA LEAKAGE REPORT")
    logger.info("======================================================================")
    logger.info(f"Total Unique Training Instances : {len(train_set)}")
    logger.info(f"Total Unique Testing Instances  : {len(test_set)}")
    
    if len(exact_duplicates) > 0:
        logger.critical(f"FATAL LEAK: Detected {len(exact_duplicates)} identical image files spanning Train and Test splits.")
        logger.info("First 5 overlapping instances:")
        for dup in list(exact_duplicates)[:5]:
            logger.info(f"  - {dup}")
    else:
        logger.info("Instance Check: PASS. No exact filename duplicates detected across boundaries.")

    # 2. Heuristic Subject-Level Sequence Overlap Check
    logger.info("Executing heuristic subject-level boundary validation (Sequence Overlap)...")
    
    def get_prefix(filename: str) -> str:
        name = Path(filename).stem
        prefix = re.sub(r'[0-9]+$', '', name)
        return prefix.rstrip('_')

    train_prefixes = set(get_prefix(p) for p in train_set)
    test_prefixes = set(get_prefix(p) for p in test_set)
    
    prefix_overlap = train_prefixes.intersection(test_prefixes)
    
    if len(prefix_overlap) > 0:
        logger.warning(f"WARNING: Detected {len(prefix_overlap)} potential subject/sequence overlaps.")
        logger.warning("Subjects present in training data may appear in the unseen test set.")
        logger.info("First 5 overlapping sequences:")
        for prefix in list(prefix_overlap)[:5]:
            logger.info(f"  - {prefix}")
    else:
        logger.info("Sequence Check: PASS. No subject-level prefix overlaps detected.")
        logger.info("======================================================================")


if __name__ == "__main__":
    run_leakage_check()
