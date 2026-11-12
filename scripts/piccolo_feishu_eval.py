"""用 Piccolo 参考推理跑官方 feishu_zh 基准（64 场景 × choice/four_noul × repeats）。

复用上游参考实现（FlexCompiler/Layout）+ 官方冻结 prompts/audit/metrics，
输出与 laya 侧完全一致的指标（accuracy/macro_f1/误报动作/漏动作/延迟）。

用法：python scripts/piccolo_feishu_eval.py <model_dir> <tag> [repeats] [device] [dtype]
"""
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research" / "feishu_zh"))
sys.path.insert(0, str(ROOT / "scripts" / "piccolo_ref"))

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import AutoModelForImageTextToText, AutoTokenizer  # noqa: E402
from piccolo_layout import Layout, render_question  # noqa: E402
from piccolo_inference import FlexCompiler  # noqa: E402
from prompts import interpret, requests_for  # noqa: E402
from audit import load_cases  # noqa: E402
from metrics import summarize  # noqa: E402


def main():
    model_dir = sys.argv[1] if len(sys.argv) > 1 else "models/piccolo-4b"
    tag = sys.argv[2] if len(sys.argv) > 2 else "piccolo-v0"
    repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    device = sys.argv[4] if len(sys.argv) > 4 else "mps"
    dtype_name = sys.argv[5] if len(sys.argv) > 5 else "bfloat16"

    print(f"[piccolo-eval] loading {model_dir} on {device}/{dtype_name} ...", flush=True)
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    compiler = FlexCompiler(tokenizer, SimpleNamespace(Layout=Layout, render_question=render_question))
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, dtype=getattr(torch, dtype_name), device_map={"": device},
        attn_implementation="sdpa", local_files_only=True).eval()
    print(f"[piccolo-eval] loaded in {time.time()-t0:.0f}s", flush=True)

    decoder = model.model.language_model
    head = model.get_output_embeddings()
    attention_state = {"bidirectional": True, "calls": 0}

    def full_attention_mask(module, positional, kwargs):
        if attention_state["bidirectional"]:
            hidden = kwargs["hidden_states"]
            length = hidden.shape[1]
            kwargs["attention_mask"] = torch.zeros(
                (hidden.shape[0], 1, length, length), dtype=hidden.dtype, device=hidden.device)
            attention_state["calls"] += 1
        return positional, kwargs

    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    hooks = [m.register_forward_pre_hook(full_attention_mask, with_kwargs=True)
             for m in decoder.modules() if isinstance(m, Qwen3_5Attention)]
    print(f"[piccolo-eval] bidirectional hooks on {len(hooks)} full-attention layers", flush=True)

    cases, _manifest = load_cases()
    label_tokens = {}
    jobs = [(r, case, mode) for r in range(repeats) for case in cases
            for mode in ("choice", "four_noul")]
    out_rows = []
    t0 = time.time()
    for i, (repeat, case, mode) in enumerate(jobs):
        req = requests_for(case)[mode]
        row_status, row_pred, row_probs, elapsed = "ok", None, None, 0.0
        try:
            compiled, _layout = compiler.compile(req)
            answers = {}
            t0 = time.perf_counter()
            for row in compiled:
                ids = torch.tensor([row.input_ids], device=device)
                attention_state["bidirectional"] = True
                with torch.inference_mode():
                    out = decoder(input_ids=ids, attention_mask=None, use_cache=False)
                    label_rows = head.weight[torch.tensor(row.candidate_ids, device=device)].float()
                    logits = torch.mv(label_rows, out.last_hidden_state[0, -1].float())
                    probs = torch.softmax(logits, dim=-1).cpu().tolist()
                pm = {key: p for key, p in zip(row.option_keys, probs)}
                q = req["questions"][row.question_id]
                if q["type"] == "choice":
                    best = max(pm, key=pm.get)
                    answers[row.question_id] = {"type": "choice", "choice": best,
                                                "probabilities": pm, "confidence": max(pm.values())}
                else:
                    answers[row.question_id] = {"type": "noul", "noul": pm.get(True, 0.0)}
            elapsed = (time.perf_counter() - t0) * 1000
            row_pred = interpret(mode, answers)["predicted"]
            if mode == "choice":
                row_probs = answers["category"]["probabilities"]
        except Exception as e:  # noqa: BLE001
            row_status = f"error: {type(e).__name__}: {e}"
        out_rows.append({"id": case["id"], "family": case["family"], "expected": case["expected"],
                         "mode": mode, "repeat": repeat, "status": row_status,
                         "predicted": row_pred if row_status == "ok" else None,
                         "probabilities": row_probs, "elapsed_ms": round(elapsed, 1)})
        if (i + 1) % 16 == 0:
            done = sum(1 for r in out_rows if r["status"] == "ok")
            print(f"  {i+1}/{len(jobs)} requests | ok {done} | {time.time()-t0:.0f}s", flush=True)

    summary = {}
    for mode in ("choice", "four_noul"):
        m = summarize([r for r in out_rows if r["mode"] == mode])
        summary[mode] = {k: m[k] for k in ("n", "accuracy", "macro_f1", "false_action_count",
                                           "nonaction_denominator", "missed_action_count",
                                           "action_denominator", "urgent_recalled", "urgent_denominator")}
        summary[mode]["p50_ms"] = round(m["timing"]["p50_ms"], 1)
    print(json.dumps({"tag": tag, **{k: v for k, v in summary.items()}}, ensure_ascii=False, indent=1))

    out_dir = ROOT / "results" / f"feishu_{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "raw.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out_rows) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("saved ->", out_dir)


if __name__ == "__main__":
    main()
