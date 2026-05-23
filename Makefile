# Save this file exactly as: Makefile
SHELL := /bin/bash
.DEFAULT_GOAL := help

# =============================================================================
# MEIA Framework: Multimodal Emotion, Intention, and Action Recognition
# Author: Imad Gohar and Rashid Riyadh, et al.
# Institution: Sunway University, Malaysia
# =============================================================================

PYTHON ?= python
TORCHRUN ?= torchrun

CONFIG ?= configs/meia_training.json

OUTPUT_DIR ?= checkpoints/meia-results-final
SMOKE_OUTPUT_DIR ?= checkpoints/meia-smoke
REPRODUCE_OUTPUT_DIR ?= checkpoints/meia-reproduce-paper

# VRAM Optimized Defaults (DINOv2 + RoBERTa)
BATCH_SIZE ?= 8
EPOCHS ?= 6
NUM_WORKERS ?= 4
SEEDS ?= 41 42 43

# Strict Local Vault Architecture
MODELS_DIR ?= $(PWD)/models
HF_HOME ?= $(MODELS_DIR)/hf_hub
TRANSFORMERS_CACHE ?= $(MODELS_DIR)/hf_hub
HF_DATASETS_CACHE ?= $(MODELS_DIR)/hf_hub
TORCH_HOME ?= $(MODELS_DIR)/torch_hub
MINE_CURATED_ROOT ?= $(PWD)/data/mine_curated

GPU_COUNT := $(shell $(PYTHON) -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 0)

# -----------------------------------------------------------------------------
# Utility Helpers
# -----------------------------------------------------------------------------
define ensure_cache_dirs
	mkdir -p data/mine_curated/images data/fane models/hf_hub models/torch_hub checkpoints
endef

define export_cache_env
	export HF_HOME="$(HF_HOME)"; \
	export TRANSFORMERS_CACHE="$(TRANSFORMERS_CACHE)"; \
	export HF_DATASETS_CACHE="$(HF_DATASETS_CACHE)"; \
	export TORCH_HOME="$(TORCH_HOME)"; \
	export MINE_CURATED_ROOT="$(MINE_CURATED_ROOT)";
endef

# -----------------------------------------------------------------------------
# Help Menu
# -----------------------------------------------------------------------------
.PHONY: help
help:
	@echo "MEIA FRAMEWORK TRAINING PIPELINE"
	@echo "================================================="
	@echo "Main targets:"
	@echo "  make install           - Install system dependencies"
	@echo "  make preflight         - Create vaults and verify Curated MINE/FANE data"
	@echo "  make predownload       - Cache DINOv2 and RoBERTa foundation models"
	@echo "  make smoke-test        - Quick 1-epoch architecture compilation test"
	@echo "  make train-single-gpu  - Full training on a single GPU"
	@echo "  make train-multi-gpu   - Full training using all detected GPUs (DDP)"
	@echo "  make train-auto        - Auto-selects single or multi-GPU training based on hardware"
	@echo "  make reproduce-paper   - Balances data and runs reproducible BMVC defense run"
	@echo "  make organize-paper    - Compile raw output logs into paper-ready formats"
	@echo "  make clean             - Remove Python cache artifacts"
	@echo ""

# -----------------------------------------------------------------------------
# Environment & Preflight
# -----------------------------------------------------------------------------
.PHONY: install
install:
	pip install -r requirements.txt
	@echo "[INFO] Dependencies installed successfully."

.PHONY: preflight
preflight:
	@$(ensure_cache_dirs)
	@echo "[INFO] Running dataset readiness verification..."
	@$(export_cache_env) \
	$(PYTHON) tools/validation/check_cloud_dataset_ready.py \
		--sources "mine_curated,fane" \
		--train-rows 2 \
		--val-rows 1 \
		--report-path data/source_availability_report.json \
		--output-json data/dataset_check.json
	@echo "[INFO] Preflight complete. Local dataset paths secured."

.PHONY: predownload
predownload:
	@$(ensure_cache_dirs)
	@echo "[INFO] Caching DINOv2 and RoBERTa weights locally..."
	@$(export_cache_env) \
	$(PYTHON) tools/data_prep/predownload_assets.py
	@echo "[INFO] Foundation models cached."

# -----------------------------------------------------------------------------
# MEIA Core Training
# -----------------------------------------------------------------------------
.PHONY: smoke-test
smoke-test:
	@$(ensure_cache_dirs)
	@echo "[INFO] Initiating architecture smoke test (1 Epoch)..."
	@$(export_cache_env) \
	$(PYTHON) scripts/train_meia.py \
		--config $(CONFIG) \
		--output-dir $(SMOKE_OUTPUT_DIR) \
		--epochs 1 \
		--batch-size 4 \
		--seeds 41 \
		--num-workers 0
	@echo "[INFO] Smoke test passed. Architecture compiled successfully."

.PHONY: train-single-gpu
train-single-gpu:
	@$(ensure_cache_dirs)
	@echo "[INFO] Initiating MEIA training (Single GPU)..."
	@$(export_cache_env) \
	$(PYTHON) scripts/train_meia.py \
		--config $(CONFIG) \
		--output-dir $(OUTPUT_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--seeds $(SEEDS)

.PHONY: train-multi-gpu
train-multi-gpu:
	@$(ensure_cache_dirs)
	@if [ "$(GPU_COUNT)" -lt 2 ]; then \
		echo "[ERROR] train-multi-gpu requires at least 2 GPUs, found $(GPU_COUNT)."; \
		exit 1; \
	fi
	@echo "[INFO] Initiating MEIA distributed training on $(GPU_COUNT) GPUs..."
	@$(export_cache_env) \
	$(TORCHRUN) --nproc_per_node=$(GPU_COUNT) scripts/train_meia.py \
		--config $(CONFIG) \
		--output-dir $(OUTPUT_DIR) \
		--epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--num-workers $(NUM_WORKERS) \
		--seeds $(SEEDS)

.PHONY: train-auto
train-auto:
	@$(ensure_cache_dirs)
	@echo "[INFO] Detected $(GPU_COUNT) GPU(s)."
	@if [ "$(GPU_COUNT)" -gt 1 ]; then \
		$(MAKE) train-multi-gpu CONFIG="$(CONFIG)" OUTPUT_DIR="$(OUTPUT_DIR)" EPOCHS="$(EPOCHS)" BATCH_SIZE="$(BATCH_SIZE)" SEEDS="$(SEEDS)"; \
	else \
		$(MAKE) train-single-gpu CONFIG="$(CONFIG)" OUTPUT_DIR="$(OUTPUT_DIR)" EPOCHS="$(EPOCHS)" BATCH_SIZE="$(BATCH_SIZE)" NUM_WORKERS="$(NUM_WORKERS)" SEEDS="$(SEEDS)"; \
	fi

# -----------------------------------------------------------------------------
# Academic Reproducibility
# -----------------------------------------------------------------------------
.PHONY: balance-fane
balance-fane:
	@echo "[INFO] Balancing FANE dataset partitions..."
	@$(PYTHON) tools/data_prep/balance_fane.py

.PHONY: reproduce-paper
reproduce-paper: balance-fane
	@$(ensure_cache_dirs)
	@mkdir -p $(REPRODUCE_OUTPUT_DIR)
	@$(PYTHON) -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" \
	|| (echo "[ERROR] CUDA is not available. Install CUDA-enabled PyTorch." && exit 1)
	@$(export_cache_env) \
	if [ ! -d "$$MINE_CURATED_ROOT" ]; then \
		echo "[WARNING] Curated dataset not found at $$MINE_CURATED_ROOT"; \
	else \
		echo "[INFO] Using Pure MINE dataset from: $$MINE_CURATED_ROOT"; \
	fi; \
	if [ "$(GPU_COUNT)" -gt 1 ]; then \
		$(TORCHRUN) --nproc_per_node=$(GPU_COUNT) scripts/train_meia.py \
			--config $(CONFIG) \
			--batch-size $(BATCH_SIZE) \
			--epochs $(EPOCHS) \
			--output-dir $(REPRODUCE_OUTPUT_DIR); \
	else \
		$(PYTHON) scripts/train_meia.py \
			--config $(CONFIG) \
			--batch-size $(BATCH_SIZE) \
			--epochs $(EPOCHS) \
			--output-dir $(REPRODUCE_OUTPUT_DIR); \
	fi
	@echo "[INFO] Reproducibility run complete: $(REPRODUCE_OUTPUT_DIR)"

.PHONY: organize-paper
organize-paper:
	@echo "[INFO] Organizing research paper artifacts..."
	@$(PYTHON) tools/data_prep/organize_paper_data.py $(REPRODUCE_OUTPUT_DIR) research_paper_data
	@echo "[INFO] Paper folder organized successfully."

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete || true
	rm -rf .pytest_cache .mypy_cache
	@echo "[INFO] Workspace cleaned."
