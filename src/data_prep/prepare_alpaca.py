from datasets import load_dataset
import json, random, os

def prepare_alpaca(output_dir="data/alpaca", eval_size=100, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    print("Downloading Alpaca-Cleaned dataset...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")

    def format_example(ex):
        return {
            "instruction": ex["instruction"],
            "input": ex.get("input", ""),
            "output": ex["output"]
        }

    data = [format_example(ex) for ex in ds]
    random.shuffle(data)
    eval_data = data[:eval_size]
    train_data = data[eval_size:]

    with open(f"{output_dir}/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open(f"{output_dir}/eval.json", "w") as f:
        json.dump(eval_data, f, indent=2)

    print(f"Done! {len(train_data)} train, {len(eval_data)} eval")

if __name__ == "__main__":
    prepare_alpaca()
