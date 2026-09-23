"""上传本地 laya 检查点到 ModelScope（python scripts/upload_ms_model.py <local_dir> <model_id> <token>）"""
import sys

import modelscope
from modelscope.hub.api import HubApi

local_dir, model_id, token = sys.argv[1], sys.argv[2], sys.argv[3]

api = HubApi()
api.login(token)
try:
    api.create_model(model_id=model_id, visibility=1, license="Apache-2.0",
                     chinese_name="问问Laya 中文微调",
                     description="Chinese fine-tuned laya (RLCD): feishu workplace triage + customer-service decisions")
    print("model created:", model_id)
except Exception as e:
    print("create skipped:", e)

print("uploading folder (this may take a few minutes for ~620MB) ...", flush=True)
api.upload_folder(repo_id=model_id, folder_path=local_dir,
                  commit_message="laya-zh-v2: Chinese RLCD fine-tuned decision checkpoint")
print("upload done:", f"https://modelscope.cn/models/{model_id}")
print("modelscope version:", modelscope.__version__)
