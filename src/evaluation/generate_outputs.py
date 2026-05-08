import json, yaml, os, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
from tqdm import tqdm
sys.path.append(".")

def load_config():
    with open("configs/config.yaml") as f:
        return yaml.safe_load(f)

def load_model(checkpoint_id, cfg):
    base = cfg["student_model"]
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, quantization_config=bnb, device_map="auto", trust_remote_code=True
    )
    if checkpoint_id == 1:
        model = PeftModel.from_pretrained(model, cfg["outputs"]["checkpoint1"])
    elif checkpoint_id == 2:
        model = PeftModel.from_pretrained(model, cfg["outputs"]["checkpoint2"])
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, instruction, input_text="", max_new=256):
    inp = f"\n### Input:\n{input_text}" if input_text else ""
    prompt = f"### Instruction:\n{instruction}{inp}\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new,
            do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

def run_checkpoint(checkpoint_id, cfg):
    print(f"\n--- Checkpoint {checkpoint_id} ---")
    model, tokenizer = load_model(checkpoint_id, cfg)
    for eval_type in ["alpaca", "json"]:
        data_file = cfg["data"][f"{eval_type}_eval"]
        out_file = f"outputs/checkpoint{checkpoint_id}/{eval_type}_outputs.json"
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        with open(data_file) as f:
            examples = json.load(f)
        results = []
        for ex in tqdm(examples, desc=f"  {eval_type}"):
            response = generate_response(model, tokenizer, ex["instruction"], ex.get("input", ""))
            results.append({
                "instruction": ex["instruction"],
                "input": ex.get("input", ""),
                "expected_output": ex["output"],
                "model_output": response,
                "checkpoint": checkpoint_id
            })
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved {len(results)} outputs to {out_file}")

if __name__ == "__main__":
    cfg = load_config()
    for ckpt in [0, 1, 2]:
        run_checkpoint(ckpt, cfg)
