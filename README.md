# MEIA-Framework

A multimodal framework for enhanced image analysis and understanding.

## Project Structure

```
├── Makefile
├── requirements.txt
├── requirements-cu126.txt
├── README.md
│
├── data/
│   └── cloud_datasets.py
│
├── models/
│   └── advanced_multimodal_bear.py
│
├── training/
│   ├── eval.py
│   └── losses.py
│
├── scripts/
│   ├── train_advanced_multimodal.py
│   ├── train_multimodal_cloud.py
│   ├── eval_only.py
│   └── run_quantitative_ablation.py
│
├── tools/
│   ├── data_prep/
│   │   ├── distill_llama_annotations.py
│   │   ├── balance_fane.py
│   │   ├── predownload_assets.py
│   │   └── organize_paper_data.py
│   │
│   ├── validation/
│   │   ├── check_cloud_dataset_ready.py
│   │   ├── check_data_splits.py
│   │   ├── check_data_leakage.py
│   │   ├── dataset_audit.py
│   │   ├── verify_dataloaders.py
│   │   └── verify_roberta_text.py
│   │
│   └── analysis/
│       ├── generate_table_3.py
│       ├── plot_graphs.py
│       ├── plot_loss.py
│       ├── extract_good_cases.py
│       ├── extract_bad_cases.py
│       ├── extract_leak_samples.py
│       ├── extract_test_samples.py
│       ├── export_architecture.py
│       ├── detector.py
│       ├── report.py
│       └── walkthrough.py
```

## Installation

```bash
make install
```

## Training

```bash
make train
```

## Evaluation

```bash
make eval
```

## Cleanup

```bash
make clean
```
