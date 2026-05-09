import json, os, sys
from tqdm import tqdm
sys.path.append(".")
from src.utils.api_client import call_model

def call_judge(instruction, response_a, response_b):
    prompt = f"""You are an expert evaluator of language model outputs.

Task: {instruction}

Response A:
{response_a}

Response B:
{response_b}

Score each response on these dimensions from 1 (worst) to 5 (best):
- instruction_following
- correctness
- clarity
- completeness
- structured_output_validity
- hallucination_risk

Return ONLY valid JSON with no extra text in this exact format:
{{"response_a_scores": {{"instruction_following": N, "correctness": N, "clarity": N, "completeness": N, "structured_output_validity": N, "hallucination_risk": N}}, "response_b_scores": {{"instruction_following": N, "correctness": N, "clarity": N, "completeness": N, "structured_output_validity": N, "hallucination_risk": N}}, "winner": "A or B or TIE", "justification": "one sentence reason"}}"""

    raw = call_model(prompt, role="judge", max_tokens=512, temperature=0.0)
    try:
        return json.loads(raw)
    except:
        return {"raw_response": raw, "parse_error": True}

def run_pairwise(ckpt_a, ckpt_b, eval_type="alpaca"):
    file_a = f"outputs/checkpoint{ckpt_a}/{eval_type}_outputs.json"
    file_b = f"outputs/checkpoint{ckpt_b}/{eval_type}_outputs.json"
    if not os.path.exists(file_a) or not os.path.exists(file_b):
        print(f"  Missing files for ckpt {ckpt_a} or {ckpt_b}, skipping.")
        return
    with open(file_a) as f: data_a = json.load(f)
    with open(file_b) as f: data_b = json.load(f)
    results = []
    wins_a = wins_b = ties = 0
    for a, b in tqdm(zip(data_a, data_b), total=len(data_a),
                     desc=f"  ckpt{ckpt_a} vs ckpt{ckpt_b} [{eval_type}]"):
        judgment = call_judge(a["instruction"], a["model_output"], b["model_output"])
        winner = judgment.get("winner", "?")
        if winner == "A":   wins_a += 1
        elif winner == "B": wins_b += 1
        else:               ties   += 1
        results.append({
            "instruction": a["instruction"],
            "checkpoint_a": ckpt_a, "checkpoint_b": ckpt_b,
            "response_a": a["model_output"],
            "response_b": b["model_output"],
            "judgment": judgment
        })
    n = len(results)
    summary = {
        "checkpoint_a": ckpt_a, "checkpoint_b": ckpt_b,
        "eval_type": eval_type,
        "win_rate_a": round(wins_a/n, 3),
        "win_rate_b": round(wins_b/n, 3),
        "tie_rate":   round(ties/n, 3),
        "total": n
    }
    print(f"\n  Summary: {summary}")
    out_file = f"outputs/judge_{eval_type}_ckpt{ckpt_a}_vs_ckpt{ckpt_b}.json"
    with open(out_file, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"  Saved to {out_file}")

if __name__ == "__main__":
    print("Running judge evaluation...")
    for a, b in [(0,1), (1,2), (0,2)]:
        for eval_type in ["alpaca", "json"]:
            run_pairwise(a, b, eval_type)
