import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib # 用于保存模型
import matplotlib.pyplot as plt

# 1. 加载数据
print("Loading data...")
df = pd.read_csv("net_data.csv")

# 简单的数据清洗：去掉空值
df = df.dropna()

# 选取特征：RTT 和 Retrans
features = ['avg_rtt_us', 'retrans_count']
X = df[features]

# 2. 训练模型
print("Training Isolation Forest Model...")
# contamination=0.1 表示我们要告诉模型：“大概有 10% 的数据是异常的”
# 如果你的实验里大部分时间都在制造故障，这个值可以调大，比如 0.3
model = IsolationForest(n_estimators=100, contamination=0.2, random_state=42)
model.fit(X)

# 3. 预测（验证一下效果）
df['anomaly'] = model.predict(X)
# Isolation Forest 的输出：1 是正常，-1 是异常
df['label'] = df['anomaly'].apply(lambda x: "Normal" if x == 1 else "Anomaly")

print("-" * 30)
print("模型评估预览:")
print(df['label'].value_counts())
print("-" * 30)

# 4. 保存模型
joblib.dump(model, "isolation_forest.pkl")
print("✅ 模型已保存为 isolation_forest.pkl")

# 5. 可视化训练结果 (画出异常点)
plt.figure(figsize=(10, 6))

# 画正常点 (绿色)
normal = df[df['anomaly'] == 1]
plt.scatter(normal['avg_rtt_us'], normal['retrans_count'], c='green', alpha=0.5, label='Normal')

# 画异常点 (红色)
anomaly = df[df['anomaly'] == -1]
plt.scatter(anomaly['avg_rtt_us'], anomaly['retrans_count'], c='red', alpha=0.6, marker='x', label='Anomaly')

plt.title("AI Detection Result: Normal vs Anomaly")
plt.xlabel("RTT (us)")
plt.ylabel("Retransmission Count")
plt.legend()
plt.grid(True, alpha=0.3)

plt.savefig("model_result.png")
print("✅ 结果图已保存为 model_result.png")