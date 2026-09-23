"""探查 laya 决策模型的完整结构：顶层模块、编码器逐层、决策头、前向形状追踪。

用法：python scripts/inspect_model.py [模型路径，默认 models/laya-zh]
"""
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def fmt(n):
    return "%.2fM" % (n / 1e6) if n >= 1e6 else "%d" % n


def dump_module(m, indent=0, max_depth=2):
    lines = []
    for name, child in m.named_children():
        extra = ""
        if isinstance(child, torch.nn.Linear):
            extra = "(in=%d, out=%d, bias=%s)" % (child.in_features, child.out_features, child.bias is not None)
        elif isinstance(child, torch.nn.Embedding):
            extra = "(num=%d, dim=%d)" % child.num_embeddings, child.embedding_dim
            extra = "(num=%d, dim=%d)" % (child.num_embeddings, child.embedding_dim)
        elif isinstance(child, torch.nn.LayerNorm):
            extra = "(dim=%d, eps=%g, bias=%s)" % (child.normalized_shape[0], child.eps, child.bias is not None)
        elif isinstance(child, torch.nn.TransformerEncoderLayer):
            sa = child.self_attn
            extra = "(d_model=%d, nhead=%d, ffn=%d)" % (sa.embed_dim, sa.num_heads, child.linear1.out_features)
        lines.append("%s%s %s %s [%s params]" % (" " * indent, type(child).__name__, name, extra, fmt(n_params(child))))
        if indent // 2 < max_depth:
            lines.extend(dump_module(child, indent + 2, max_depth))
    return lines


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "laya-zh")
    from laya.common import build_model, build_sequence, QTYPES
    from laya.agent import _fix_tokenizer_config
    from transformers import AutoTokenizer

    _fix_tokenizer_config(path)
    with open(os.path.join(path, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    model = build_model(cfg, encoder_dir=os.path.join(path, "encoder"))
    from safetensors.torch import load_file
    weights = load_file(os.path.join(path, "model.safetensors"))
    missing, unexpected = model.load_state_dict(weights, strict=False)
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    tok = AutoTokenizer.from_pretrained(os.path.join(path, "tokenizer"))
    model.eval()

    print("=" * 30, "顶层 DecisionModel", "=" * 30)
    print("总参数量:", fmt(n_params(model)))
    for name, child in model.named_children():
        print("  %-10s %-24s %s params" % (name, type(child).__name__, fmt(n_params(child))))
    print("\n-- 决策头各子模块 --")
    for name, mod in model.named_children():
        if name == "encoder":
            continue
        print("[%s] %s" % (name, type(mod).__name__))
        for line in dump_module(mod, 4, max_depth=1):
            print(line)

    print("\n" + "=" * 30, "编码器（mmBERT-base / ModernBERT 架构）", "=" * 30)
    enc = model.encoder
    print("config: hidden=%d layers=%d heads=%d ffn=%d vocab=%d max_pos=%d" % (
        enc.config.hidden_size, enc.config.num_hidden_layers, enc.config.num_attention_heads,
        enc.config.intermediate_size, enc.config.vocab_size, enc.config.max_position_embeddings))
    print("attention: global every %d layers, local window %d, rope_theta=%s" % (
        enc.config.global_attn_every_n_layers, enc.config.local_attention,
        enc.config.rope_parameters.get("full_attention", {}).get("rope_theta")))
    print("\n-- embeddings --")
    for name, child in enc.embeddings.named_children():
        print("  %-20s %-22s %s" % (name, type(child).__name__, fmt(n_params(child))))
    print("  (token emb 权重与 MLM decoder tied: %s)" % enc.config.tie_word_embeddings)

    types = enc.config.layer_types
    print("\n-- 22 层注意力类型 --")
    for i, t in enumerate(types):
        print("  layer %2d: %s" % (i, t), end="\n" if (i + 1) % 3 == 0 else "  |")
    print()

    for kind in (0, 1):
        layer = enc.layers[kind]
        print("\n-- 一个 %s 层的内部结构 (layer %d, %s params) --" % (
            types[kind], kind, fmt(n_params(layer))))
        for name, child in layer.named_children():
            print("  [%s] %s" % (name, type(child).__name__))
            for ln, c in child.named_children():
                extra = ""
                if isinstance(c, torch.nn.Linear):
                    extra = "(in=%d out=%d bias=%s)" % (c.in_features, c.out_features, c.bias is not None)
                elif "Norm" in type(c).__name__:
                    extra = "(dim=%d bias=%s)" % (c.normalized_shape[0], c.bias is not None)
                print("    %-28s %-24s %s %s" % (ln, type(c).__name__, extra, fmt(n_params(c))))
        attn = layer.attn
        print("    heads=%d×dim=%d (QKV 融合 Wqkv), window=%s" % (
            enc.config.num_attention_heads, enc.config.hidden_size // enc.config.num_attention_heads,
            enc.config.local_attention if "sliding" in types[kind] else None))
        print("    attn_norm=%s, mlp_norm=%s (残差前归一化)" % (
            type(layer.attn_norm).__name__, type(layer.mlp_norm).__name__))
        print("    mlp: Wi 768→%d（gate+value 融合，GiLU 分块），Wo %d→768（有效中间维 %d）" % (
            layer.mlp.Wi.out_features, layer.mlp.Wi.out_features // 2 * 0 + layer.mlp.Wo.in_features,
            layer.mlp.Wi.out_features // 2))

    print("\n-- 编码器参数分布 --")
    print("  embeddings: %s" % fmt(n_params(enc.embeddings)))
    print("  22 layers : %s (平均 %s/层)" % (fmt(n_params(enc.layers)), fmt(n_params(enc.layers) / 22)))
    print("  final_norm: %s" % fmt(n_params(enc.final_norm)))

    print("\n" + "=" * 30, "前向传播形状追踪（真实中文样本）", "=" * 30)
    state = "快递显示派送了但没送到，放错了驿站，我腿脚不方便，请帮我重新安排派送。"
    q = {"t": "choice", "ins": "判断这位用户的主要诉求属于哪一类。",
         "crit": {"refund": "申请退款、退货或换货", "logistics": "查询物流", "product": "产品咨询", "complaint": "投诉", "other": "其他"}}
    ids, markers = build_sequence(tok, state, q, cfg.get("max_len", 512), cfg.get("head_max_len", 256))
    print("序列文本预览:", tok.decode(ids[:60]).replace("\n", " ")[:150], "...")
    print("总 token 数=%d, 选项 marker 位置=%s" % (len(ids), markers))
    print("marker 位置对应的 token:", [tok.decode([ids[m]]) for m in markers])

    x = torch.tensor([ids])
    with torch.no_grad():
        h = model.encoder(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
        print("\n[1] encoder 输出 last_hidden_state:", tuple(h.shape))
        h2 = h + model.type_emb(torch.tensor([QTYPES["choice"]]))[:, None, :]
        print("[2] + type_emb(3×%d 查表广播到每个 token):" % model.type_emb.embedding_dim, tuple(h2.shape))
        out = h2
        for i, layer in enumerate(model.head.layers):
            out = layer(out, src_key_padding_mask=None)
            print("[3.%d] 决策头 transformer layer %d ->" % (i, i), tuple(out.shape))
        m = torch.gather(out, 1, torch.tensor(markers)[None, :, None].expand(-1, -1, out.size(-1)))
        print("[4] gather 出 %d 个选项 marker 向量:" % len(markers), tuple(m.shape))
        logits = model.scorer(m).squeeze(-1)
        print("[5] scorer(LayerNorm→Linear(%d,%d)→GELU→Linear(%d,1)) ->" % (
            model.scorer[1].in_features, model.scorer[1].out_features, model.scorer[3].out_features),
            tuple(logits.shape), "=", [round(float(v), 3) for v in logits[0]])
        p = torch.softmax(logits, -1)
        print("[6] softmax over 选项 ->", {k: round(float(v), 3) for k, v in zip(q["crit"], p[0])})
        feats = torch.stack([torch.tensor(0.9), torch.tensor(0.8), torch.tensor(0.1), torch.tensor(5 / 255.0)])
        act_in = torch.cat([out[:, 0], feats[None].float()], -1)
        act = model.act_head(act_in)
        print("[7] act_head(Linear(%d,256)→GELU→Linear(256,2), 输入=[CLS]池化+4统计特征) ->" % model.act_head[0].in_features,
            tuple(act.shape), "(2类: act / escalate)")
    print("\ntemperature buffer (3种题型):", model.temperature.tolist())


if __name__ == "__main__":
    main()
