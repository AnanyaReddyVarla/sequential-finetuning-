#!/bin/bash
#SBATCH --job-name=stage2_json
#SBATCH --partition=gpu2v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/work/roj023/sequential-finetuning/logs/stage2_%j.out
#SBATCH --error=/work/roj023/sequential-finetuning/logs/stage2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=roj023@arc.utsa.edu
source /work/roj023/envs/llm_env/bin/activate
cd /work/roj023/sequential-finetuning
echo "Starting Stage 2..."
python src/training/train.py --stage 2
echo "Stage 2 complete!"
