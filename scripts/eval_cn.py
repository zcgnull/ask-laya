"""在中文测试集上评估任意 laya 检查点（准确率 / ECE / Brier / 延迟）。

用法：
  python scripts/eval_cn.py --model convaiinnovations/laya --subfolder multilingual --tag ml-base
  python scripts/eval_cn.py --model models/laya-zh --tag zh-finetuned
输出 results/eval_<tag>.json
"""
import argparse
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def load_rows(name):
    path = os.path.join(ROOT, "data", name)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(agent, rows, max_rows=None):
    from laya.common import ece_score

    if max_rows:
        rows = rows[:max_rows]

    per_type = {"choice": [], "score": [], "noul": []}
    t0 = time.perf_counter()
    n_tokens = 0
    for row in rows:
        res = agent.predict(row["state"], row["questions"])
        n_tokens += res["usage"]["input_tokens"]
        gold = row["gold"]
        for qid, ans in res["answers"].items():
            g = gold[qid]["probabilities"]
            if ans["type"] == "choice":
                keys = list(row["questions"][qid]["criteria"].keys())
                gold_label = max(keys, key=lambda k: g[k])
                p = np.array([ans["probabilities"][k] for k in keys])
                gvec = np.array([g[k] for k in keys])
                per_type["choice"].append({
                    "conf": ans["confidence"],
                    "correct": float(ans["choice"] == gold_label),
                    "brier": float(((p - gvec) ** 2).sum()),
                    "top2": float(gold_label in [k for k, _ in sorted(zip(keys, p), key=lambda kv: -kv[1])[:2]]),
                })
            elif ans["type"] == "score":
                gold_level = max(range(len(g)), key=lambda i: g[str(i)])
                exp = ans["score"]
                per_type["score"].append({
                    "conf": ans["confidence"],
                    "correct": float(round(exp) == gold_level),
                    "mae": abs(exp - gold_level),
                    "brier": float(sum((ans["probabilities"][str(i)] - g[str(i)]) ** 2 for i in range(len(g)))),
                })
            else:
                pt = ans["noul"]
                gt = g["true"]
                per_type["noul"].append({
                    "conf": max(pt, 1 - pt),
                    "correct": float((pt > 0.5) == (gt > 0.5)),
                    "brier": (pt - gt) ** 2,
                })

    dt = time.perf_counter() - t0
    summary = {}
    for qt, recs in per_type.items():
        if not recs:
            continue
        conf = np.array([r["conf"] for r in recs])
        corr = np.array([r["correct"] for r in recs])
        entry = {
            "n": len(recs),
            "accuracy": round(float(corr.mean()), 4),
            "ece": round(ece_score(conf, corr), 4),
            "mean_confidence": round(float(conf.mean()), 4),
            "brier": round(float(np.mean([r["brier"] for r in recs])), 4),
        }
        if qt == "choice":
            entry["top2_accuracy"] = round(float(np.mean([r["top2"] for r in recs])), 4)
        if qt == "score":
            entry["mae"] = round(float(np.mean([r["mae"] for r in recs])), 4)
        summary[qt] = entry

    summary["overall"] = {
        "n_decisions": sum(len(v) for v in per_type.values()),
        "accuracy": round(float(np.mean([
            r["correct"] for v in per_type.values() for r in v])), 4),
        "ece": round(ece_score(
            np.array([r["conf"] for v in per_type.values() for r in v]),
            np.array([r["correct"] for v in per_type.values() for r in v])), 4),
    }
    summary["latency"] = {
        "total_s": round(dt, 2),
        "ms_per_case": round(dt * 1000 / len(rows), 1),
        "ms_per_decision": round(dt * 1000 / max(1, summary["overall"]["n_decisions"]), 1),
        "input_tokens_per_case": round(n_tokens / len(rows), 1),
    }
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import laya
    agent = laya.load(args.model, subfolder=args.subfolder, device=args.device)
    print("device:", agent.device, "| temperature:", dict(zip(["choice", "score", "noul"], agent.temperature)))

    results = {"tag": args.tag, "model": args.model, "subfolder": args.subfolder, "device": str(agent.device)}
    for name, key in [("cn_test.jsonl", "template"), ("cn_test_hard.jsonl", "hard")]:
        rows = load_rows(name)
        print("evaluating %s (%d rows) ..." % (name, len(rows)))
        results[key] = evaluate(agent, rows, args.max_rows)
        print(json.dumps(results[key], ensure_ascii=False, indent=2))

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.path.join(ROOT, "results", "eval_%s.json" % args.tag)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("saved ->", out)


if __name__ == "__main__":
    main()
