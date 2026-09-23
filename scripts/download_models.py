"""Download the two laya checkpoints used by verification and fine-tuning."""
import laya

print("== downloading English checkpoint (repo root) ==", flush=True)
agent_en = laya.load("convaiinnovations/laya")
print("english cfg max_len=%s head_max_len=%s encoder=%s" % (
    agent_en.cfg.get("max_len"), agent_en.cfg.get("head_max_len"), agent_en.cfg.get("encoder")), flush=True)

print("== downloading multilingual checkpoint (subfolder) ==", flush=True)
agent_ml = laya.load("convaiinnovations/laya", subfolder="multilingual")
print("multilingual cfg max_len=%s head_max_len=%s encoder=%s" % (
    agent_ml.cfg.get("max_len"), agent_ml.cfg.get("head_max_len"), agent_ml.cfg.get("encoder")), flush=True)

print("DONE", flush=True)
