"""用 HTTP 调用 /v1/systemone 后端（Piccolo llama-server 等）跑官方 feishu_zh 基准。

用法：python scripts/piccolo_http_feishu_eval.py --url http://127.0.0.1:8200 --tag piccolo-4b [repeats]
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research" / "feishu_zh"))

import urllib.request  # noqa: E402

from prompts import interpret, requests_for  # noqa: E402
from audit import load_cases  # noqa: E402
from metrics import summarize  # noqa: E402


def post(url, payload, timeout=300):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.loads(resp.read())
    return result, (time.perf_counter() - t0) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True, help="如 http://127.0.0.1:8200（/v1/systemone 所在基址）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    endpoint = args.url.rstrip("/") + "/v1/systemone"
    cases, _manifest = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    jobs = [(r, case, mode) for r in range(args.repeats) for case in cases
            for mode in ("choice", "four_noul")]
    out_rows = []
    t0 = time.time()
    for i, (repeat, case, mode) in enumerate(jobs):
        req = requests_for(case)[mode]
        row = {"id": case["id"], "family": case["family"], "expected": case["expected"],
               "mode": mode, "repeat": repeat}
        try:
            response, elapsed = post(endpoint, {"state": req["state"], "questions": req["questions"]})
            answers = response["answers"]
            row.update(status="ok", elapsed_ms=round(elapsed, 1),
                       **interpret(mode, answers))
            if mode == "choice":
                row["probabilities"] = answers["category"]["probabilities"]
        except Exception as e:  # noqa: BLE001
            row.update(status=f"error: {type(e).__name__}: {e}")
        out_rows.append(row)
        if (i + 1) % 16 == 0:
            ok = sum(1 for r in out_rows if r["status"] == "ok")
            print(f"  {i+1}/{len(jobs)} | ok {ok} | {time.time()-t0:.0f}s", flush=True)

    summary = {}
    for mode in ("choice", "four_noul"):
        m = summarize([r for r in out_rows if r["mode"] == mode])
        summary[mode] = {k: m[k] for k in ("n", "accuracy", "macro_f1", "false_action_count",
                                           "nonaction_denominator", "missed_action_count",
                                           "action_denominator", "urgent_recalled", "urgent_denominator")}
        summary[mode]["p50_ms"] = round(m["timing"]["p50_ms"], 1)
    print(json.dumps({"tag": args.tag, **summary}, ensure_ascii=False, indent=1))

    out_dir = ROOT / "results" / f"feishu_{args.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
