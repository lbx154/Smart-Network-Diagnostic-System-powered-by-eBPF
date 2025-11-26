# 🚀 SmartNetDiag: 基于 eBPF + AI 的智能网络诊断系统

> **Smart Network Diagnostic System powered by eBPF & Isolation Forest**

[![eBPF](https://img.shields.io/badge/Linux-eBPF-orange.svg)](https://ebpf.io/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![AI](https://img.shields.io/badge/Model-Isolation%20Forest-green.svg)](https://scikit-learn.org/)
[![Dashboard](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)

SmartNetDiag 是一个轻量级、低开销的实时网络诊断系统。它利用 **eBPF (Extended Berkeley Packet Filter)** 技术在 Linux 内核态零拷贝采集 TCP 关键指标（RTT、重传），并结合 **孤立森林 (Isolation Forest)** 无监督学习算法，实现对网络异常（如拥塞、丢包）的实时检测与根因分析。

---

## 📂 项目目录结构

```text
SmartNetDiag/
├── 📄 run_experiment.sh   # [一键启动] 自动化实验脚本 (采集+流量+故障注入)
├── 📄 smart_agent.py      # [数据平面] eBPF 探针，负责内核数据采集与清洗
├── 📄 chaos_maker.py      # [测试工具] 基于 tc 的网络故障注入器
├── 📄 train_model.py      # [智能平面] 读取 CSV 数据，训练模型并评估
├── 📄 dashboard.py        # [应用平面] Streamlit 实时监控仪表盘
├── 📄 visualize_data.py   # [分析工具] 简单的数据分布可视化脚本
├── 📄 requirements.txt    # Python 依赖库列表
└── 📄 README.md           # 项目说明文档
```

---

## 🛠️ 环境搭建 (Installation)

本项目推荐运行在 **Ubuntu 20.04/22.04 LTS** (物理机、虚拟机或 WSL2) 环境下。

### 1. 系统依赖安装 (eBPF 工具链)

eBPF 依赖较新的内核头文件，请确保系统内核版本 >= 5.8。

```bash
# 更新源
sudo apt update

# 安装 BCC 工具链及内核头文件
sudo apt install -y bison flex build-essential libssl-dev libelf-dev zlib1g-dev \
libfl-dev systemtap-sdt-dev clang llvm \
bpfcc-tools python3-bpfcc libbpfcc libbpfcc-dev linux-headers-$(uname -r)

# 仅限 WSL2 用户：
# 如果遇到头文件缺失问题，需手动下载微软 WSL2 内核源码并编译头文件。
# (此处省略详细 WSL2 内核编译步骤，具体参考项目文档)
```

### 2. Python 依赖安装

```bash
# 安装项目所需的 Python 库
pip3 install -r requirements.txt
```

*`requirements.txt` 内容参考：*
```text
numpy
pandas
scikit-learn
streamlit
matplotlib
joblib
altair
```

---

## 🚀 快速开始 (Workflow)

整个系统分为三个阶段：**数据采集 → 模型训练 → 实时监控**。

### 第一步：数据采集与故障模拟 (Data Collection)

我们提供了一个自动化脚本，它会同时启动：
1.  **Smart Agent**: eBPF 探针，采集 RTT 和重传数据。
2.  **Traffic Generator**: 后台运行 `wget` 维持长连接流量。
3.  **Chaos Maker**: 使用 `tc` (Traffic Control) 随机注入“高延迟”或“丢包”故障。

```bash
# ⚠️ 必须使用 sudo 运行，因为 eBPF 需要 root 权限
sudo bash run_experiment.sh
```

*   **输出**：数据将实时写入 `net_data.csv`。
*   **操作**：运行约 5-10 分钟后，按 `Ctrl+C` 停止实验。

### 第二步：模型训练 (Model Training)

利用采集到的 `net_data.csv`，训练 Isolation Forest 模型。

```bash
python3 train_model.py
```

*   **功能**：
    *   清洗数据。
    *   训练无监督异常检测模型。
    *   生成可视化散点图 `model_result.png` 以验证模型效果。
*   **输出**：生成模型文件 `isolation_forest.pkl`。

### 第三步：启动实时监控看板 (Dashboard)

启动 Streamlit 前端页面，加载训练好的模型，对实时网络状态进行推断和展示。

**注意**：为了展示实时效果，建议重新运行 `run_experiment.sh` (让它在后台产生数据)，然后新开一个终端启动 Dashboard。

```bash
# 启动 Dashboard
streamlit run dashboard.py
```

*   **访问地址**：打开浏览器访问 `http://localhost:8501`
*   **功能演示**：
    *   观察 RTT 实时折线图。
    *   当后台注入故障时，观察右上角 AI 状态是否变为 🔴 **异常**。

---

## 📊 实验结果展示

### 1. 数据特征分布 (Data Distribution)
通过 eBPF 采集的数据呈现清晰的 "L" 型分布：
*   **正常流量**：聚集在原点 (低延迟，无重传)。
*   **拥塞异常**：沿 X 轴延伸 (高延迟，无重传)。
*   **丢包异常**：沿 Y 轴延伸 (低延迟，高重传)。

### 2. 实时监控界面
Dashboard 能够毫秒级捕捉网络波动，并标记异常点。

> *(此处可插入你的 Dashboard 截图)*

---

## 🌟 项目亮点 (Highlights)

*   **零侵入性**：基于 eBPF 技术，无需修改内核源码，无需重启应用，性能开销极低。
*   **真实指标**：通过 Hook `tcp_rcv_established` 和 `tcp_retransmit_skb`，获取内核协议栈真实的 RTT 和重传事件，比 Ping 更准确。
*   **智能诊断**：摒弃传统的静态阈值报警，使用 **Isolation Forest** 自动学习网络基线，能够适应不同的网络环境。
*   **全栈闭环**：实现了从底层内核采集、故障模拟、模型训练到上层可视化展示的完整工程链路。

---

## 📝 License

此项目仅供计算机网络课程学习与研究使用。

---

### 👨‍💻 作者
*   **姓名**：[你的名字]
*   **学号**：[你的学号]
*   **专业**：计算机科学与技术