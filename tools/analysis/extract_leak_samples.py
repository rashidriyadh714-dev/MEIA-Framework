#!/usr/bin/env python3
"""
MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
Module: Visual Leakage Extractor

This script isolates and extracts potential heuristic subject-level data 
leaks (flipbook sequences) by mapping prefix identifiers across the 
training and testing splits. Outputs are saved for manual visual inspection.
"""

import sys
import re
import shutil
import logging
from pathlib import Path
from collections import defaultdict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from data.multimodal_dataset import get_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VisualLeakExtractor")

def extract_image_paths(dataloader) -> list:
    paths = []
    dataset = dataloader.dataset
    for sample in dataset.samples:
        if getattr(sample, 'image_path', None):
            paths.append(str(sample.image_path))
    return paths

def get_prefix(filename: str) -> str:
    name = Path(filename).stem
    prefix = re.sub(r'[0-9]+$', '', name)
    return prefix.rstrip('_')

def run_visual_extraction() -> None:
    logger.info("Initializing Visual Leakage Extraction Engine...")
    
    train_loader, _, test_loader = get_dataloaders(
        batch_size=1, eval_batch_size=1, num_workers=4, distributed=False
    )

    logger.info("Extracting spatial pathways...")
    train_paths = extract_image_paths(train_loader)
    test_paths = extract_image_paths(test_loader)

    train_dict = defaultdict(list)
    test_dict = defaultdict(list)

    for p in train_paths:
        train_dict[get_prefix(p)].append(p)
        
    for p in test_paths:
        test_dict[get_prefix(p)].append(p)

    train_prefixes = set(train_dict.keys())
    test_prefixes = set(test_dict.keys())
    
    prefix_overlap = train_prefixes.intersection(test_prefixes)

    if not prefix_overlap:
        logger.info("Data Integrity Confirmed: Zero subject overlap detected.")
        return

    output_dir = project_root / "outputs" / "leak_inspection"
    train_out_dir = output_dir / "TRAINING_SAMPLES"
    test_out_dir = output_dir / "TESTING_SAMPLES"

    if output_dir.exists():
        shutil.rmtree(output_dir)

    train_out_dir.mkdir(parents=True, exist_ok=True)
    test_out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("======================================================================")
    logger.info(f" DETECTED POTENTIAL LEAKS: {len(prefix_overlap)} Identities")
    logger.info("======================================================================")
    logger.info(f"Cloning heuristic overlap samples to: {output_dir}")
    
    for person in list(prefix_overlap):
        logger.info(f"Cloning identity instances for subset: '{person}'")
        
        person_train_dir = train_out_dir / person
        person_test_dir = test_out_dir / person
        
        person_train_dir.mkdir(exist_ok=True)
        person_test_dir.mkdir(exist_ok=True)
        
        for path_str in train_dict[person][:10]:
            src_path = Path(path_str)
            if src_path.exists():
                shutil.copy2(src_path, person_train_dir / src_path.name)
            
        for path_str in test_dict[person][:10]:
            src_path = Path(path_str)
            if src_path.exists():
                shutil.copy2(src_path, person_test_dir / src_path.name)

    logger.info("Visual extraction complete. Please manually inspect the output directory.")

if __name__ == "__main__":
    run_visual_extraction()
