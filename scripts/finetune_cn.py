"""在自建中文数据上用 RLCD（GRPO 式策略梯度 + proper scoring rule 奖励）微调 laya-multilingual。

忠实复现官方 notebook（laya_finetune_typed_decisions_2xT4_kaggle.ipynb）的训练循环，
针对单卡 MPS/CPU（Apple M4 16GB）做了如下适配：
  1. 无 DDP / 无 fp16 GradScaler（MPS 用 fp32）；
  2. 默认冻结编码器 embedding 层（mmBERT 词表 256k，冻结可省约 2.4GB 优化器内存）；
  3. 编码器开梯度检查点；
  4. 批量缩小（MICRO_BATCH=8 × GROUP_SIZE=4 = 32 序列/步）。

用法：
  python scripts/finetune_cn.py [--epochs 3] [--freeze-embeddings] [--freeze-encoder]
输出 models/laya-zh/（含温度校准后的 rl_agent_config.json）
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from safetensors.torch import load_file, save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def collate_train_batch(items, pad_id):
    n, L = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos,
        "marker_mask": mmask, "target": target,
        "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def fit_one_temp(sel):
    """官方温度拟合：对单题型做 LBFGS 温度缩放（最小化 NLL）。"""
    if len(sel) < 10:
        return 1.0
    kmax = max(len(z) for z, _ in sel)
    Z = torch.full((len(sel), kmax), -1e4)
    T = torch.zeros((len(sel), kmax))
    for i, (z, t) in enumerate(sel):
        Z[i, :len(z)] = torch.tensor(z)
        T[i, :len(t)] = torch.tensor(t, dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)
    def closure():
        opt.zero_grad()
        loss = -(T * torch.log_softmax(Z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss
    opt.step(closure)
    return float(torch.clamp(log_t.exp(), 0.5, 5.0).item())


def build_items(rows, tok, cfg):
    """把中文 jsonl 行转成训练序列（与官方 build_training_item 相同的逻辑）。"""
    from laya.common import build_sequence, render_options, QTYPES

    items = []
    for row in rows:
        state, questions, gold = row["state"], row["questions"], row["gold"]
        for qid, q in questions.items():
            t = q["type"]
            crit = q.get("criteria", {})
            g = gold[qid]["probabilities"]
            if t == "choice":
                keys = list(crit.keys())
                target = [g.get(k, 0.0) for k in keys]
            elif t == "noul":
                target = [g.get("false", 0.5), g.get("true", 0.5)]
            else:
                n = len(crit) if isinstance(crit, list) else 4
                target = [g.get(str(i), 0.0) for i in range(n)]
            s = sum(target)
            target = [v / s for v in target] if s > 0 else [1.0 / len(target)] * len(target)
            internal = {"t": t, "ins": q["instructions"], "crit": crit if t != "choice" else {k: crit[k] for k in keys}}
            k = len(render_options(internal))
            seq, markers = build_sequence(tok, state, internal, cfg.get("max_len", 512), cfg.get("head_max_len", 256))
            if len(markers) != k:
                continue
            items.append({"ids": seq, "markers": markers, "qtype": QTYPES[t],
                          "target": target, "label": int(np.argmax(target))})
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="convaiinnovations/laya")
    ap.add_argument("--subfolder", default="multilingual")
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "laya-zh"))
    ap.add_argument("--train", default=os.path.join(ROOT, "data", "cn_train.jsonl"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--lr-encoder", type=float, default=2.5e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--freeze-embeddings", action="store_true", default=False)
    ap.add_argument("--freeze-encoder", action="store_true", default=False)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from huggingface_hub import snapshot_download
    from laya.agent import _fix_tokenizer_config
    from laya.common import build_model, proper_reward

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("device:", device)

    print("resolving base checkpoint...")
    model_dir = snapshot_download(args.base, allow_patterns=[
        (args.subfolder + "/" if args.subfolder else "") + n
        for n in ("rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*")])
    if args.subfolder:
        model_dir = os.path.join(model_dir, args.subfolder)
    _fix_tokenizer_config(model_dir)

    with open(os.path.join(model_dir, "rl_agent_config.json")) as f:
        cfg = json.load(f)
    print("base cfg:", {k: cfg.get(k) for k in ("encoder", "max_len", "head_max_len", "temperature")})

    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    weights = load_file(os.path.join(model_dir, "model.safetensors"))
    model.load_state_dict(weights, strict=True)

    # 内存优化：编码器梯度检查点
    has_gc = hasattr(model.encoder, "gradient_checkpointing_enable")
    if has_gc:
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # 冻结策略（Mac 显存适配）
    enc = model.encoder.get_decoder() if hasattr(model.encoder, "get_decoder") else model.encoder
    n_frozen = 0
    if args.freeze_embeddings or args.freeze_encoder:
        emb = getattr(enc, "embeddings", None)
        if emb is not None:
            for p in emb.parameters():
                p.requires_grad_(False)
                n_frozen += p.numel()
    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad_(False)

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print("params: total=%.1fM trainable=%.1fM (frozen=%.1fM)" % (n_total / 1e6, n_train / 1e6, n_frozen / 1e6))

    model.to(device)
    model.train()

    print("loading train rows...")
    with open(args.train, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    items = build_items(rows, tok, cfg)
    print("built %d training items from %d rows" % (len(items), len(rows)))

    enc_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("encoder.")]
    groups = []
    if enc_params:
        groups.append({"params": enc_params, "lr": args.lr_encoder})
    if head_params:
        groups.append({"params": head_params, "lr": args.lr_head})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.01)

    MICRO_BATCH, GRAD_ACCUM, GROUP_SIZE = args.micro_batch, args.grad_accum, args.group_size
    SIGMA_START, SIGMA_END = 0.4, 0.1
    steps_per_epoch = len(items) // MICRO_BATCH
    total_updates = max(1, (steps_per_epoch // GRAD_ACCUM) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=1e-6)
    print("steps/epoch=%d, updates=%d, epochs=%d" % (steps_per_epoch, total_updates, args.epochs))

    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        random.shuffle(items)
        epoch_loss, n_batches, accum_step = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, args.epochs - 1)
        t0 = time.time()

        for b_idx in range(0, len(items), MICRO_BATCH):
            chunk = items[b_idx:b_idx + MICRO_BATCH]
            if not chunk:
                continue
            batch = collate_train_batch(chunk, tok.pad_token_id)
            logits, act = model(
                batch["input_ids"].to(device), batch["attention_mask"].to(device),
                batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                batch["qtype"].to(device))
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)

            # RLCD：G 组零均值高斯探索
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)

            with torch.no_grad():
                r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask, w_sph=0.75, w_rps=1.0)
                adv = r - r.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)

            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + 1.0 * loss_ce) / GRAD_ACCUM + 0.0 * act.sum()

            loss.backward()
            accum_step += 1
            if accum_step % GRAD_ACCUM == 0 or (b_idx + MICRO_BATCH) >= len(items):
                torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * GRAD_ACCUM
            n_batches += 1
            if n_batches % 25 == 0:
                print("  epoch %d | step %d/%d | loss %.4f | reward %.3f | lr %.2e | %.1fs" % (
                    epoch + 1, n_batches, steps_per_epoch, loss.item() * GRAD_ACCUM,
                    r.mean().item(), scheduler.get_last_lr()[0], time.time() - t0), flush=True)

        avg = epoch_loss / max(1, n_batches)
        history.append({"epoch": epoch + 1, "loss": avg, "sigma": sigma, "sec": round(time.time() - t0, 1)})
        print("=== epoch %d/%d done | avg loss %.4f | %.1fs ===" % (epoch + 1, args.epochs, avg, time.time() - t0), flush=True)

        # 滚动 checkpoint，防中断丢失
        ckpt = os.path.join(args.out, "checkpoint_latest")
        os.makedirs(ckpt, exist_ok=True)
        save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(ckpt, "model.safetensors"))

    print("training done in %.1fs, fitting calibration temperatures..." % (time.time() - t_start))
    model.eval()

    # 温度校准（训练集每 5 条取 1，最多 300 条；测试集完全不参与）
    calib_items = items[::max(1, len(items) // 300)]
    calib_preds = []
    with torch.no_grad():
        for c in range(0, len(calib_items), 16):
            cb = collate_train_batch(calib_items[c:c + 16], tok.pad_token_id)
            l_sub, _ = model(cb["input_ids"].to(device), cb["attention_mask"].to(device),
                             cb["marker_pos"].to(device), cb["marker_mask"].to(device),
                             cb["qtype"].to(device))
            l_np = l_sub.float().cpu().numpy()
            for r_i, it in enumerate(calib_items[c:c + 16]):
                k = len(it["markers"])
                calib_preds.append((it["qtype"], l_np[r_i, :k], it["target"]))

    fitted = [1.0, 1.0, 1.0]
    for qt in range(3):
        sel = [(z, t) for q_type, z, t in calib_preds if q_type == qt]
        if sel:
            fitted[qt] = fit_one_temp(sel)
    print("fitted temperatures (choice, score, noul):", [round(t, 3) for t in fitted])

    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(args.out, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(args.out, "encoder"))
    tok.save_pretrained(os.path.join(args.out, "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-zh"
    cfg["base_checkpoint"] = "%s:%s" % (args.base, args.subfolder)
    cfg["temperature"] = fitted
    cfg["finetune_history"] = history
    with open(os.path.join(args.out, "rl_agent_config.json"), "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("saved fine-tuned model ->", args.out)


if __name__ == "__main__":
    main()
