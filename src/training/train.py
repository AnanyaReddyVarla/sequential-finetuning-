import argparse, json, yaml, os, sys
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch
sys.path.append(".")

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

def load_data(path):
    with open(path) as f:
        data = json.load(f)
    def format_prompt(ex):
        inp = f"\n### Input:\n{ex['input']}" if ex.get("input") else ""
        return f"### Instruction:\n{ex['instruction']}{inp}\n### Response:\n{ex['output']}"
    return Dataset.from_list([{"text": format_prompt(ex)} for ex in data])

def main(args):
    cfg = load_config()
    stage = args.stage
    if stage == 1:
        base_model = cfg["student_model"]
        data_path  = cfg["data"]["alpaca_train"]
        output_dir = cfg["outputs"]["checkpoint1"]
        epochs     = cfg["training"]["stage1_epochs"]
    else:
        base_model = cfg["outputs"]["checkpoint1"]
        data_path  = cfg["data"]["json_train"]
        output_dir = cfg["outputs"]["checkpoint2"]
        epochs     = cfg["training"]["stage2_epochs"]

    print(f"\n{'='*50}")
    print(f"  Stage {stage} Training")
    print(f"  Base model : {base_model}")
    print(f"  Data       : {data_path}")
    print(f"  Output     : {output_dir}")
    print(f"  Epochs     : {epochs}")
    print(f"{'='*50}\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    dataset = load_data(data_path)
    print(f"Training on {len(dataset)} examples...")

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=cfg["training"]["batch_size"],
        gradient_accumulation_steps=cfg["training"]["gradient_accumulation_steps"],
        learning_rate=cfg["training"]["learning_rate"],
        fp16=False, bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        max_length=cfg["training"]["max_seq_length"],
        dataset_text_field="text",
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nStage {stage} complete. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2])
    main(parser.parse_args())
