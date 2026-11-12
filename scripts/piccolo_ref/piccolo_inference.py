"""Full-checkpoint Piccolo reference: native linear layers, noncausal full attention."""
import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from piccolo_layout import Layout, render_question

def prepare_question(question):
    """Explicit benchmark extension; official DI choice/noul remain unchanged."""
    if any(k in question for k in ("gold", "answers", "expected", "labels")):
        raise ValueError("Supervision is not an inference input")
    q = dict(question)
    kind = q["type"]
    criteria = q.get("criteria")
    if kind == "score":
        if not isinstance(criteria, list) or not 2 <= len(criteria) <= 255:
            raise ValueError("score requires an ordered list of 2..255 described levels")
        if any(not isinstance(x, str) or not x.strip() for x in criteria):
            raise ValueError("score levels must be nonempty descriptions")
        q.update(type="choice", criteria={str(i): text for i, text in enumerate(criteria)})
    elif kind == "noul":
        if isinstance(criteria, dict) and set(criteria) == {"yes", "no"}:
            q["criteria"] = {"false": criteria["no"], "true": criteria["yes"]}
    elif kind != "choice":
        raise ValueError("Only choice, noul and ordered score are supported")
    order = q.get("option_order")
    if order is not None:
        if kind == "score":
            raise ValueError("score level order must remain the original numeric order")
        c = q.get("criteria")
        if not isinstance(c, dict) or len(order) != len(set(order)) or set(order) != set(c):
            raise ValueError("option_order must contain every criteria key exactly once")
        q["criteria"] = {key: c[key] for key in order}
    return q


@dataclass(frozen=True)
class PromptRow:
    question_id: str
    kind: str
    option_keys: tuple
    candidate_ids: tuple
    input_ids: tuple
    repeated: bool


class FlexCompiler:
    def __init__(self, tokenizer, upstream, *, variant="flex_twice", max_context=65536, twice_max_tokens=3000):
        if variant not in ("flex_decider", "flex_header", "flex_twice"):
            raise ValueError("Unknown FlexDM input layout")
        if max_context < 2 or twice_max_tokens < 0:
            raise ValueError("Invalid context or repeat budget")
        self.layout = upstream.Layout(tokenizer)
        self.render_question = upstream.render_question
        self.variant = variant
        self.max_row_tokens = max_context - 1
        self.twice_max_tokens = twice_max_tokens
        # Validate every available candidate at the actual answer-token boundary.
        for name, token in zip(self.layout.labels, self.layout.label_ids):
            if self.layout._enc("Answer: (" + name) != self.layout.answer + [token]:
                raise ValueError("Candidate is not one token at the answer boundary: " + name)

    def compile(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Inference request must be an object")
        if any(k in payload for k in ("gold", "answers", "expected", "labels")):
            raise ValueError("Supervision is not an inference input")
        if payload.get("enable_thinking", False) is not False:
            raise ValueError("FlexDM probability readout requires thinking=false")
        questions = payload.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise ValueError("Expected a complete nonempty request, not a pre-split question")
        rendered = []
        for key, question in questions.items():
            text, options, keys = self.render_question(prepare_question(question))
            rendered.append((key, question["type"], text, options, tuple(keys)))
        texts = [r[2] for r in rendered] if self.variant != "flex_decider" else None
        ctx = self.layout.context(payload["state"], texts)
        again = self.layout.again(payload["state"], texts) if self.variant == "flex_twice" else None
        pieces = []
        for key, kind, text, options, keys in rendered:
            once = self.layout.question(text, options)
            twice = self.layout.question(text, options, again) if again is not None else None
            pieces.append((key, kind, keys, once, twice))
        once_tokens = len(ctx) + sum(len(row[3]) for row in pieces)
        repeated = (
            again is not None
            and all(len(ctx) + len(row[4]) <= self.max_row_tokens for row in pieces)
            and (self.twice_max_tokens == 0 or once_tokens <= self.twice_max_tokens)
        )
        rows = []
        for key, kind, keys, once, twice in pieces:
            ids = tuple(ctx + (twice if repeated else once))
            if len(ids) > self.max_row_tokens:
                raise ValueError("Complete prompt exceeds context; no truncation permitted")
            rows.append(PromptRow(key, kind, keys, tuple(self.layout.label_ids[:len(keys)]), ids, repeated))
        return rows, {
            "variant": self.variant,
            "request_questions": len(rows),
            "request_once_tokens": once_tokens,
            "request_repeated": repeated,
            "model_input_tokens": sum(len(row.input_ids) for row in rows),
            "twice_max_tokens": self.twice_max_tokens,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--output")
    parser.add_argument("--compare-causal", action="store_true")
    args = parser.parse_args()
    import torch
    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    descriptor = json.loads((Path(args.model) / "decision-inference.json").read_text())
    if descriptor.get("attention_mode") != "bidirectional_prompt" or descriptor.get("prompt_variant") != "flex_twice":
        raise ValueError("Unsupported Piccolo inference contract")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    compiler = FlexCompiler(tokenizer, SimpleNamespace(Layout=Layout, render_question=render_question))
    payload = json.loads(Path(args.request).read_text())
    rows, layout = compiler.compile(payload)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map={"": args.device},
        attn_implementation="sdpa", local_files_only=True,
    ).eval()
    decoder = model.model.language_model
    head = model.get_output_embeddings()
    attention_state = {"bidirectional": True, "calls": 0}
    def full_attention_mask(module, positional, kwargs):
        if attention_state["bidirectional"]:
            hidden = kwargs["hidden_states"]
            length = hidden.shape[1]
            kwargs["attention_mask"] = torch.zeros((hidden.shape[0], 1, length, length), dtype=hidden.dtype, device=hidden.device)
            attention_state["calls"] += 1
        return positional, kwargs
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention
    hooks = [module.register_forward_pre_hook(full_attention_mask, with_kwargs=True)
             for module in decoder.modules() if isinstance(module, Qwen3_5Attention)]
    expected_full_layers = decoder.config.layer_types.count("full_attention")
    if len(hooks) != expected_full_layers:
        raise ValueError("Full-attention layer count mismatch")
    results = []
    for row in rows:
        ids = torch.tensor([row.input_ids], device=args.device)
        n = ids.shape[1]
        entry = {"question_id": row.question_id, "kind": row.kind, "option_keys": row.option_keys,
                 "candidate_ids": row.candidate_ids, "input_ids": row.input_ids, "repeated": row.repeated}
        for mode in (["bidirectional_prompt", "causal"] if args.compare_causal else ["bidirectional_prompt"]):
            attention_state["bidirectional"] = mode == "bidirectional_prompt"
            with torch.inference_mode():
                out = decoder(input_ids=ids, attention_mask=None, use_cache=False)
                label_rows = head.weight[torch.tensor(row.candidate_ids, device=args.device)].float()
                logits = torch.mv(label_rows, out.last_hidden_state[0, -1].float())
                probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            if not all(math.isfinite(x) for x in probabilities):
                raise ValueError("Non-finite candidate probabilities")
            entry[mode] = {"candidate_logits": logits.cpu().tolist(), "probabilities": probabilities}
        results.append(entry)
    output = {"model": Path(args.model).name, "dtype": args.dtype, "torch": torch.__version__, "transformers": transformers.__version__,
              "attention_mode": "bidirectional_prompt", "full_attention_mask": "all visible within each complete prompt",
              "linear_attention": "unchanged", "prefix_cache": False, "chunked_prefill": False,
              "full_attention_layers": len(hooks), "full_attention_hook_calls": attention_state["calls"],
              "probability_temperature": 1.0, "layout": layout, "results": results}
    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    else:
        print(text)

if __name__ == "__main__":
    main()
