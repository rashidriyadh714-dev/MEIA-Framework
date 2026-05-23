# Makefile for MEIA-Framework

.PHONY: help install train eval clean

help:
	@echo "MEIA-Framework Commands"
	@echo "====================="
	@echo "make install    - Install dependencies"
	@echo "make train      - Train the model"
	@echo "make eval       - Evaluate the model"
	@echo "make clean      - Clean up temporary files"

install:
	pip install -r requirements.txt

train:
	python scripts/train_advanced_multimodal.py

eval:
	python scripts/eval_only.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
