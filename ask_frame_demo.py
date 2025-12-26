import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 参数配置
# ===============================
fs = 100_000          # 采样率
symbol_rate = 5_000   # 符号率
sps = fs // symbol_rate
snr_db = 5            # 信噪比
payload_bits = 100

# 前导（用于“类帧结构”）
preamble = np.array([1, 0] * 8)
data_bits = np.random.randint(0, 2, payload_bits)
bits = np.concatenate([preamble, data_bits])

# ===============================
# 发送端：ASK / OOK
# ===============================
symbols = np.repeat(bits, sps)
tx_signal = symbols.astype(float)

# 加 AWGN
signal_power = np.mean(tx_signal ** 2)
noise_power = signal_power / (10 ** (snr_db / 10))
noise = np.sqrt(noise_power) * np.random.randn(len(tx_signal))
rx_signal = tx_signal + noise

# 在前面加纯噪声（模拟未知起点）
noise_prefix = np.sqrt(noise_power) * np.random.randn(3 * sps)
rx_signal = np.concatenate([noise_prefix, rx_signal])

# ===============================
# 接收端：滑动能量帧检测
# ===============================
window = sps
energy = np.convolve(rx_signal ** 2,
                      np.ones(window) / window,
                      mode='same')

# 自适应门限（噪声区估计）
noise_est = np.mean(energy[:2 * sps])
threshold = 4 * noise_est

# 找第一个超过门限的位置
candidates = np.where(energy > threshold)[0]
frame_start = candidates[0] if len(candidates) > 0 else None

print(f"检测到的帧起始点: {frame_start}")

# ===============================
# 解调（非相干）
# ===============================
rx_bits = []
if frame_start is not None:
    start = frame_start + sps  # 略过不稳定区
    for i in range(len(bits)):
        seg = rx_signal[start + i * sps:
                        start + (i + 1) * sps]
        rx_bits.append(int(np.mean(seg) > 0.5))

rx_bits = np.array(rx_bits)

# 计算 BER（仅 payload）
rx_payload = rx_bits[len(preamble):]
ber = np.mean(rx_payload != data_bits)
print(f"BER = {ber:.4f}")

# ===============================
# 可视化
# ===============================
plt.figure(figsize=(10, 6))

plt.subplot(3, 1, 1)
plt.title("接收信号（时域）")
plt.plot(rx_signal)
if frame_start:
    plt.axvline(frame_start, color='r', linestyle='--')

plt.subplot(3, 1, 2)
plt.title("滑动能量")
plt.plot(energy)
plt.axhline(threshold, color='g', linestyle='--')

plt.subplot(3, 1, 3)
plt.title("发送 vs 接收比特（前 50 个）")
plt.stem(bits[:50], linefmt='b-', markerfmt='bo', basefmt=' ')
plt.stem(rx_bits[:50] + 0.1, linefmt='r--', markerfmt='rx', basefmt=' ')
plt.tight_layout()
plt.show()
