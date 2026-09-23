#!/bin/bash
# 用官方协议评测检查点：eval_feishu.sh <checkpoint_dir> <tag> [repeats] [device]
set -e
CKPT="$1"; TAG="$2"; REPEATS="${3:-3}"; DEVICE="${4:-mps}"
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$HERE/../research/feishu_zh"
OUT="$HERE/../results/feishu_$TAG"
cd "$BENCH"
../../.venv/bin/python run.py --backend laya --checkpoint "$CKPT" --device "$DEVICE" \
  --output "$OUT" --repeats "$REPEATS" 2>&1 | tail -1
../../.venv/bin/python - << PYEOF
import json, sys
sys.path.insert(0, ".")
from metrics import summarize
rows = [json.loads(l) for l in open("$OUT/raw.jsonl")]
out = {}
for mode in ("choice", "four_noul"):
    m = summarize([r for r in rows if r["mode"] == mode])
    out[mode] = {"accuracy": m["accuracy"], "macro_f1": round(m["macro_f1"], 3),
                 "false_action": f'{m["false_action_count"]}/{m["nonaction_denominator"]}',
                 "missed_action": f'{m["missed_action_count"]}/{m["action_denominator"]}',
                 "p50_ms": round(m["timing"]["p50_ms"])}
print(json.dumps({"tag": "$TAG", **out}, ensure_ascii=False, indent=1))
json.dump(out, open("../../results/feishu_$TAG-summary.json", "w"), ensure_ascii=False)
PYEOF
