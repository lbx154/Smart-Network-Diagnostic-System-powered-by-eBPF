#!/usr/bin/python3
from bcc import BPF
import socket
import struct
import time
import csv
from collections import deque

# ==========================================
# 1. eBPF 内核代码 (二合一)
# ==========================================
bpf_text = """
#include <uapi/linux/ptrace.h>
#include <net/sock.h>
#include <bcc/proto.h>
#include <linux/tcp.h>

#undef __HAVE_BUILTIN_BSWAP32__
#undef __HAVE_BUILTIN_BSWAP64__
#undef __HAVE_BUILTIN_BSWAP16__

// 两个通道：一个传 RTT 事件，一个传重传事件
BPF_PERF_OUTPUT(rtt_events);
BPF_PERF_OUTPUT(retrans_events);

struct rtt_data_t {
    u32 rtt;
};

struct retrans_data_t {
    u32 dummy; // 不需要传具体数据，只要触发一次就算一次重传
};

// --- 功能 1: 监控 RTT ---
int trace_tcp_rcv(struct pt_regs *ctx, struct sock *sk)
{
    struct tcp_sock *ts = (struct tcp_sock *)sk;
    u32 srtt = ts->srtt_us >> 3;
    
    // 过滤掉没数据的
    if (srtt == 0) return 0;

    // 简单过滤：只看本机流量以外的，或者你可以不过滤，后期数据清洗
    // 这里为了演示简单，全都要
    
    struct rtt_data_t data = {};
    data.rtt = srtt;
    rtt_events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}

// --- 功能 2: 监控重传 ---
int trace_retransmit(struct pt_regs *ctx, struct sock *sk)
{
    // 过滤本地流量 (可选，这里简化逻辑先不写复杂过滤)
    struct retrans_data_t data = {};
    retrans_events.perf_submit(ctx, &data, sizeof(data));
    return 0;
}
"""

# ==========================================
# 2. Python 用户态聚合逻辑
# ==========================================

print("正在初始化 Smart Agent (eBPF + CSV Logger)...")
b = BPF(text=bpf_text)

# 挂载两个钩子
b.attach_kprobe(event="tcp_rcv_established", fn_name="trace_tcp_rcv")
b.attach_kprobe(event="tcp_retransmit_skb", fn_name="trace_retransmit")

# 数据缓存
rtt_list = []
retrans_count = 0

# 回调函数
def handle_rtt(cpu, data, size):
    event = b["rtt_events"].event(data)
    global rtt_list
    rtt_list.append(event.rtt)

def handle_retrans(cpu, data, size):
    global retrans_count
    retrans_count += 1

# 打开 perf buffer
b["rtt_events"].open_perf_buffer(handle_rtt)
b["retrans_events"].open_perf_buffer(handle_retrans)

# 准备 CSV 文件
csv_filename = "net_data.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "avg_rtt_us", "retrans_count", "label"]) # label 用于后续标记是否异常

print(f"开始采集！数据将写入 {csv_filename}")
print("按 Ctrl+C 停止采集...")
print("-" * 50)
print(f"{'TIMESTAMP':<15} | {'AVG RTT (us)':<15} | {'RETRANS/s':<10}")

try:
    while True:
        # 1. 收集 1 秒内的数据
        time.sleep(1)
        b.perf_buffer_poll(timeout=10) # 处理所有等待的事件
        
        # 2. 计算聚合指标
        current_time = int(time.time())
        
        # 计算平均 RTT
        avg_rtt = 0
        if len(rtt_list) > 0:
            avg_rtt = sum(rtt_list) // len(rtt_list)
        
        current_retrans = retrans_count
        
        # 3. 打印到屏幕
        print(f"{current_time:<15} | {avg_rtt:<15} | {current_retrans:<10}")
        
        # 4. 写入 CSV (Label 默认为 0，表示正常，之后我们可以手动标记)
        with open(csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([current_time, avg_rtt, current_retrans, 0])
        
        # 5. 重置计数器，准备下一秒
        rtt_list = []
        retrans_count = 0

except KeyboardInterrupt:
    print("\n采集结束。")