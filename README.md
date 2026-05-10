# Sequential Instruction Tuning

Assignment 3 — LLM and Agentic Systems, UTSA

## Overview
Two-stage QLoRA fine-tuning of Phi-3.5 Mini Instruct:
- Stage 1: General instruction following (Alpaca dataset)
- Stage 2: Structured JSON output (teacher-generated via Llama 3.3 70B)

## Requirements
- Python 3.10+
- CUDA GPU (trained on UTSA HPC)
- UTSA VPN (for teacher and judge model API calls)

## Setup
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt

## How to Run

### 1. Prepare Data
python src/data_prep/prepare_alpaca.py
python src/data_prep/generate_json_instruct.py

### 2. Train on HPC
sbatch hpc_scripts/stage1_train.sh
sbatch hpc_scripts/stage2_train.sh

### 3. Evaluate
python src/evaluation/generate_outputs.py
python src/evaluation/compute_metrics.py
python src/judge/judge_eval.py

## Repository Structure
- configs/ — hyperparameter configuration
- data/ — training and evaluation datasets
- hpc_scripts/ — SLURM batch scripts for HPC
- logs/ — training logs from HPC
- outputs/ — model checkpoints and evaluation results
- prompts/ — prompt templates
- src/ — all source code
- REPORT.md — blog post with full results and analysis

## Results
See REPORT.md for full results and analysis.
