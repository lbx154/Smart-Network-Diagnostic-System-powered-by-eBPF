#!/bin/bash

# ========================================================
# SmartNetDiag 自动化实验脚本 (Wget 版)
# ========================================================

if [ "$EUID" -ne 0 ]; then
  echo "❌ 错误: 请使用 sudo 运行此脚本"
  exit 1
fi

cleanup() {
    echo ""
    echo "🛑 正在停止实验..."
    if [ -n "$AGENT_PID" ]; then kill $AGENT_PID 2>/dev/null; fi
    if [ -n "$TRAFFIC_PID" ]; then kill $TRAFFIC_PID 2>/dev/null; fi
    tc qdisc del dev eth0 root 2>/dev/null
    echo "✅ 实验结束！请查看 net_data.csv"
}

trap cleanup EXIT

echo "🧹 清理旧日志..."
rm -f net_data.csv agent_output.log

# 1. 启动采集器
echo "🚀 [1/3] 启动 eBPF 数据采集器..."
python3 smart_agent.py > agent_output.log 2>&1 &
AGENT_PID=$!
echo "    -> Agent PID: $AGENT_PID"

# 2. 启动长连接流量 (Wget)
echo "🌊 [2/3] 启动背景流量 (下载大文件以维持长连接)..."
# 使用中科大镜像源下载 Ubuntu ISO，速度快且文件大
TARGET_URL="https://mirrors.ustc.edu.cn/ubuntu-releases/22.04/ubuntu-22.04.5-desktop-amd64.iso"

(while true; do 
    # -q:静默, -O /dev/null:不存盘, --timeout:防止卡死
    wget -q --timeout=5 --tries=2 -O /dev/null "$TARGET_URL"
    sleep 1
done) &
TRAFFIC_PID=$!
echo "    -> Traffic PID: $TRAFFIC_PID"

# 3. 启动故障注入
echo "😈 [3/3] 启动混沌制造者..."
echo "========================================================"
echo "实验开始！数据正在疯狂写入..."
echo "按 Ctrl+C 停止"
echo "========================================================"

python3 chaos_maker.py