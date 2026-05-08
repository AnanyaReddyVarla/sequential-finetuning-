import json, os
from rouge_score import rouge_scorer
import evaluate

def json_metrics(outputs):
    valid = schema_ok = exact = 0
    for ex in outputs:
        try:
            parsed = json.loads(ex["model_output"])
            valid += 1
            try:
                expected = json.loads(ex["expected_output"])
                if set(parsed.keys()) == set(expected.keys()):
                    schema_ok += 1
                if parsed == expected:
                    exact += 1
            except:
                pass
        except:
            pass
    n = len(outputs)
    return {
        "json_validity":     round(valid/n, 3),
        "schema_compliance": round(schema_ok/n, 3),
        "exact_match":       round(exact/n, 3),
        "total":             n
    }

def text_metrics(outputs):
    scorer = rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
    bertscore = evaluate.load("bertscore")
    preds = [ex["model_output"]    for ex in outputs]
    refs  = [ex["expected_output"] for ex in outputs]
    rouge = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for p, r in zip(preds, refs):
        s = scorer.score(r, p)
        for k in rouge:
            rouge[k] += s[k].fmeasure
    n = len(outputs)
    rouge = {k: round(v/n, 3) for k, v in rouge.items()}
    bs = bertscore.compute(predictions=preds, references=refs, lang="en")
    rouge["bertscore_f1"] = round(sum(bs["f1"]) / len(bs["f1"]), 3)
    return rouge

def main():
    all_results = {}
    for ckpt in [0, 1, 2]:
        all_results[f"checkpoint_{ckpt}"] = {}
        for eval_type in ["alpaca", "json"]:
            path = f"outputs/checkpoint{ckpt}/{eval_type}_outputs.json"
            if not os.path.exists(path):
                print(f"Missing: {path} - skipping")
                continue
            with open(path) as f:
                data = json.load(f)
            metrics = text_metrics(data)
            if eval_type == "json":
                metrics.update(json_metrics(data))
            all_results[f"checkpoint_{ckpt}"][eval_type] = metrics
            print(f"\nCheckpoint {ckpt} | {eval_type}: {metrics}")
    with open("outputs/all_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nAll metrics saved to outputs/all_metrics.json")

if __name__ == "__main__":
    main()
