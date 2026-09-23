"""验证 Laya 模型卡中的关键声明。

覆盖：
  A. 模型卡三个示例（choice 路由 / score 紧急度 / noul 钓鱼检测）——英文检查点
  B. 单次前向、零输出 token 的声明（usage.output_tokens == 0）
  C. 延迟实测（MPS）
  D. 跨语言崩溃声明：英文检查点在中文上"自信地错误"，multilingual 正常
  E. Router 路由器：语种检测 + 自动选择检查点（不加载模型即可给出 reason）

输出 results/verify_results.json
"""
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "results", "verify_results.json")

results = {}


def section(name):
    print("\n" + "=" * 8, name, "=" * 8)


def main():
    import laya
    from laya import Router

    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

    section("A/B: 英文检查点——模型卡示例 + 单次前向零生成")
    agent_en = laya.load("convaiinnovations/laya")
    print("device:", agent_en.device, "| cfg:", {k: agent_en.cfg.get(k) for k in ("encoder", "max_len", "head_max_len")})
    results["cfg_english"] = {k: agent_en.cfg.get(k) for k in ("encoder", "max_len", "head_max_len", "temperature", "temperature_by_options")}

    state = """Subject: Re: Invoice #4471 — final notice
From: billing@vendor-portal.example

Hi, we've resent invoice #4471 three times. Payment of $12,400 is now 45 days
overdue. If we don't receive funds within 5 business days we must escalate to
collections and pause your account. Please treat this as urgent."""

    questions = {
        "route": {
            "type": "choice",
            "instructions": "Route this message to the correct team.",
            "criteria": {
                "billing": "invoices, payments, overdue accounts",
                "technical": "bugs, outages, integration problems",
                "sales": "pricing, upgrades, new purchases",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "Rate urgency from 0 (informational) to 3 (critical).",
            "criteria": ["informational", "mild", "urgent", "critical"],
        },
        "phishing": {
            "type": "noul",
            "instructions": "Is this message likely a phishing attempt?",
            "criteria": {"false": "legitimate business message", "true": "phishing or social engineering"},
        },
    }

    t0 = time.perf_counter()
    res = agent_en.predict(state, questions)
    dt_first = (time.perf_counter() - t0) * 1000
    print(json.dumps(res["answers"], indent=2, ensure_ascii=False))
    print("first call: %.1f ms | output_tokens = %d" % (dt_first, res["usage"]["output_tokens"]))

    # 热身后测延迟（20 次取均值）
    for _ in range(5):
        agent_en.predict(state, questions)
    t0 = time.perf_counter()
    n = 20
    for _ in range(n):
        res = agent_en.predict(state, questions)
    dt = (time.perf_counter() - t0) * 1000 / n
    results["latency_english_mps"] = {"ms_per_call_3questions": round(dt, 1), "first_call_ms": round(dt_first, 1)}
    results["english_demo"] = res["answers"]
    results["zero_output_tokens"] = res["usage"]["output_tokens"] == 0
    print("warm latency: %.1f ms/call (3 questions, one forward pass)" % dt)

    section("D: 跨语言崩溃——同一中文工单，英文检查点 vs multilingual")
    cn_state = ("主题：发票4471催款提醒\n\n您好，我们已三次重发发票4471，款项12400元已逾期45天。"
                "如5个工作日内未收到付款，我们将移交法务并暂停贵司账户，请务必重视。")
    agent_ml = laya.load("convaiinnovations/laya", subfolder="multilingual")
    results["cfg_multilingual"] = {k: agent_ml.cfg.get(k) for k in ("encoder", "max_len", "head_max_len", "temperature", "temperature_by_options")}

    for tag, ag in [("english_on_chinese", agent_en), ("multilingual_on_chinese", agent_ml)]:
        r = ag.predict(cn_state, questions)
        results[tag] = r["answers"]
        print("%s -> route=%s conf=%.3f | urgency=%.2f | phishing=%.3f" % (
            tag, r["answers"]["route"]["choice"], r["answers"]["route"]["confidence"],
            r["answers"]["urgency"]["score"], r["answers"]["phishing"]["noul"]))

    section("E: Router 路由器（亚毫秒语种检测）")
    router = Router()
    for s, label in [(state, "english"), (cn_state, "chinese")]:
        rd = router.route(s, questions)
        print("route(%s) -> model=%s | reason=%s" % (label, rd.model, rd.reason))
    results["router"] = {"english_state": str(router.route(state, questions)),
                         "chinese_state": str(router.route(cn_state, questions))}

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nsaved ->", OUT)


if __name__ == "__main__":
    main()
