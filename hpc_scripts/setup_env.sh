#!/bin/bash
python3 -m venv ~/envs/llm_env
source ~/envs/llm_env/bin/activate
pip install --upgrade pip
pip install -r /work/roj023/sequential-finetuning/requirements.txt
echo "Environment ready!"
