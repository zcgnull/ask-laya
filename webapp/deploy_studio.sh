#!/bin/bash
# Ask Laya → ModelScope 创空间一键部署（依据 modelscope-studio skill 的标准流程）
#
# 用法：MODELSCOPE_API_KEY=xxxx bash deploy_studio.sh [owner/repo]
#   owner/repo 缺省为 <你的用户名>/ask-laya
#
# 步骤：whoami → 查询 gradio 免费硬件 → 创建(或复用)创空间 → git push → deploy → 轮询日志
set -euo pipefail

: "${MODELSCOPE_API_KEY:?请先 export MODELSCOPE_API_KEY=<你的访问令牌>}"
ENDPOINT="${MODELSCOPE_ENDPOINT:-https://modelscope.cn}"
API="$ENDPOINT/openapi/v1"
AUTH="Authorization: Bearer $MODELSCOPE_API_KEY"
REPO_ARG="${1:-}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STUDIO_DIR="$SCRIPT_DIR/studio"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

api() { local method=$1 path=$2; shift 2; curl -sS -m 60 -X "$method" "$API$path" -H "$AUTH" -H "Content-Type: application/json" "$@"; }

echo "== 1/6 身份验证"
ME=$(api GET /users/me)
SUCCESS=$(echo "$ME" | python3 -c "import json,sys;print(json.load(sys.stdin).get('success'))" 2>/dev/null || echo False)
if [ "$SUCCESS" != "True" ]; then echo "认证失败: $ME"; exit 1; fi
OWNER=$(echo "$ME" | python3 -c "import json,sys;d=json.load(sys.stdin)['data'];print(d.get('username') or d.get('name'))")
REPO="${REPO_ARG#*/}"; [ -z "$REPO" ] && REPO="ask-laya"
echo "  owner=$OWNER repo=$REPO"

echo "== 2/6 查询 gradio 免费硬件"
HW=$(api GET "/studios/hardware?sdk_type=gradio")
HARDWARE=$(echo "$HW" | python3 -c "
import json,sys
d=json.load(sys.stdin)['data']
items=d if isinstance(d,list) else d.get('hardware') or d.get('items') or []
free=[i for i in items if i.get('resource_type')!='paid']
print((free or items)[0]['name'] if items else 'platform/2v-cpu-16g-mem')")
echo "  hardware=$HARDWARE"

echo "== 3/6 创建/检查创空间"
INFO=$(api GET "/studios/$OWNER/$REPO")
if echo "$INFO" | grep -q '"success":true'; then
  echo "  已存在，复用"
else
  CREATE=$(api POST /studios -d "{
    \"owner\": \"$OWNER\", \"repo_name\": \"$REPO\",
    \"sdk_type\": \"gradio\", \"visibility\": \"public\",
    \"hardware\": \"$HARDWARE\", \"display_name\": \"Ask Laya 问问Laya\"}")
  echo "$CREATE" | grep -q '"success":true' || { echo "创建失败: $CREATE"; exit 1; }
  echo "  已创建 $OWNER/$REPO (public, gradio, $HARDWARE)"
fi

echo "== 4/6 同步代码（git push）"
cd "$STUDIO_DIR"
# 与本地 FastAPI 版保持同一套页面与 API：同步 server.py 与静态前端
cp "$SCRIPT_DIR/server.py" server.py
mkdir -p static && cp "$SCRIPT_DIR/static/"*.html static/
git init -q 2>/dev/null || true
git remote remove modelscope 2>/dev/null || true
MS_HOST="${ENDPOINT#https://}"
git remote add modelscope "https://oauth2:${MODELSCOPE_API_KEY}@${MS_HOST}/studios/$OWNER/$REPO.git"
git add -A && git -c user.name="${GIT_NAME:-ask-laya}" -c user.email="${GIT_EMAIL:-ask-laya@local}" commit -qm "Deploy Ask Laya to ModelScope Studio" 2>/dev/null || echo "  (无新变更)"
# 平台初始化的远端提交需要先合并（冲突保留本地版本）
git fetch modelscope master 2>/dev/null || true
if git rev-parse --verify -q modelscope/master >/dev/null; then
  git merge modelscope/master --allow-unrelated-histories -m "Merge remote init" 2>/dev/null \
    || { git checkout --ours -- . 2>/dev/null; git add -A; git -c user.name="${GIT_NAME:-ask-laya}" -c user.email="${GIT_EMAIL:-ask-laya@local}" commit -qm "Resolve conflicts, keep local version"; }
fi
git push -u modelscope HEAD:master
echo "  代码已推送"

echo "== 5/6 触发部署"
api POST "/studios/$OWNER/$REPO/deploy" | grep -q '"success":true' && echo "  已触发" || { api POST "/studios/$OWNER/$REPO/deploy"; exit 1; }

echo "== 6/6 轮询状态（最长 10 分钟）"
for i in $(seq 1 40); do
  sleep 15
  ST=$(api GET "/studios/$OWNER/$REPO" | python3 -c "import json,sys;d=json.load(sys.stdin)['data'];print(d.get('status') or d.get('deploy_status') or 'Unknown')" 2>/dev/null || echo Unknown)
  echo "  [$((i*15))s] status=$ST"
  case "$ST" in
    Running|Running_|running)
      echo "✅ 部署完成: $ENDPOINT/studios/$OWNER/$REPO"; exit 0;;
    *Error*|*error*|*Fail*|*FailED*)
      echo "❌ 部署失败，最近日志："; api GET "/studios/$OWNER/$REPO/logs/run" | head -c 3000; exit 1;;
  esac
done
echo "⚠ 超时未确认 Running，请手动查看: $ENDPOINT/studios/$OWNER/$REPO"
