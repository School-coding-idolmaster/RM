#!/usr/bin/env bash
# WSL：把 Windows 的 D:\RM\robomorph\.env（API key）经 ssh 加密复制到 Mac 的 ~/RM/robomorph/.env。
# Mac 原来的 .env 备份为 .env.bak_<时间>；复制后权限设为 600，并检查 key 是否存在（只打印长度，不打印内容）。
# 由 D:\RM\copy_env_to_mac.bat 调用。.env 被 git 忽略，不会进仓库。
set -u
src=/mnt/d/RM/robomorph/.env
[ -f "$src" ] || { echo "找不到 $src"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' "$src" || { echo "Windows 的 .env 里 ANTHROPIC_API_KEY 是空的"; exit 1; }
ssh -o ConnectTimeout=15 macmini 'cd ~/RM/robomorph && [ -f .env ] && cp -p .env ".env.bak_$(date +%Y%m%d_%H%M%S)"; true' \
  || { echo "连不上 Mac"; exit 1; }
scp -q "$src" macmini:RM/robomorph/.env || { echo "复制失败"; exit 1; }
ssh macmini 'cd ~/RM/robomorph && chmod 600 .env && git check-ignore -q .env && echo ".env 被 git 忽略：是" ; \
  for k in LLM_PROVIDER LLM_EFFORT ANTHROPIC_API_KEY; do v=$(grep -E "^$k=" .env | tail -1 | cut -d= -f2-); \
  if [ "$k" = ANTHROPIC_API_KEY ]; then echo "$k：${#v} 个字符"; else echo "$k=$v"; fi; done'
echo "完成：Mac 的 .env 已与 Windows 相同"
