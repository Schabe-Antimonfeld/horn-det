# 09 FPGA 模块划分与定点实现

## 1. RTL 模块结构

以下按第 07 章的 Log-Mel + Tiny CNN 当前参考基线划分模块。第 07 章的替代检测方案均待确定；选型完成后再评估对应前端、模型和部署位置，不表示所有候选均需实现为 RTL。

```text
top
│
├── audio_frontend
│   ├── i2s/pdm/tdm_rx
│   ├── dc_remove
│   └── channel_sync
│
├── calibration
│   ├── gain_correct
│   └── fixed_delay_correct
│
├── stft_engine
│   ├── frame_buffer
│   ├── hann_rom
│   └── fft_scheduler
│
├── horn_detector
│   ├── power_fusion
│   ├── mel_filterbank
│   ├── log_lut
│   └── tiny_cnn   (可放 PS)
│
├── localization
│   ├── noise_psd
│   ├── cross_spectrum
│   ├── phat_normalize
│   ├── horn_weight
│   ├── ifft_gcc
│   ├── pair_selector
│   ├── delay_lut
│   ├── srp_accumulator
│   ├── peak_search
│   └── confidence
│
└── camera_interface
```

## 2. FFT/IFFT 复用预算

每 10.67 ms 产生一帧。

每帧需要：

- 8 个 forward FFT；
- 若使用 28 个 pair 全部生成 GCC，需要最多 28 个 inverse FFT。

总计：

$$
8+28=36
$$

次 1024 点变换。

若 FFT IP 吞吐约 1 sample/clock，100 MHz 下纯 1024 点数据吞吐时间约：

$$
36\times\frac{1024}{10^8}
\approx0.369\ \mathrm{ms}
$$

0.369 ms 仅是所述时钟与连续吞吐假设下的数据传输估算，未计入 IP 配置/切换、流水延迟、帧间空隙、互谱/PHAT、权重生成、BRAM 访问和背压，不能据此认定整条链路有实时余量。

需对实际 IP 配置和存储结构进行周期级调度验证，并在目标板上测量。持续实时处理要求最坏情况下可按 10.67 ms 的输入间隔接收新帧且队列不持续增长；允许流水跨帧时，也要报告单帧延迟、缓冲占用和丢帧情况。第 06 章的搜索回退及跨 PL/PS 传输必须纳入最坏预算。

如果在生成 GCC 前只保留 top-K pair，IFFT 数还能进一步减少；先生成全部 GCC 再剪枝只能减少后续搜索工作量。

## 3. 推荐定点格式

本文统一约定：signed Qm.n 的 m 包含符号位，总位宽为 m+n；unsigned Qm.n 的 m 只计整数位。实值为整数编码乘以 2^(-n)。概率和置信度采用 unsigned Q0.16，角度以度为单位采用 signed Q9.7。

| 数据 | 建议 |
|---|---|
| PCM | signed 16 bit |
| Hann | signed 16 bit Q1.15，范围 [-1, 1-2^(-15)]，系数使用非负部分 |
| FFT Re/Im | signed 20~24 bit，需指定 FFT 缩放和二进制小数点 |
| Power | unsigned 32~40 bit，仅在明确缩放后适用 |
| PHAT Re/Im | signed 16 bit Q1.15，+1 饱和为 1-2^(-15) |
| Horn Weight | unsigned 16 bit Q0.16，范围 [0, 1-2^(-16)] |
| Noise PSD | unsigned 32 bit 或 block floating，尺度与 Power 一致 |
| GCC | signed 20~24 bit，需指定 IFFT 缩放和二进制小数点 |
| SRP accumulator | signed 32~40 bit，需按 GCC 范围和 pair 数核算 |
| Angle | signed 16 bit Q9.7，范围 [-256, 256-2^(-7)] 度 |
| Probability / Confidence | unsigned 16 bit Q0.16，范围 [0, 1-2^(-16)] |

这些位宽是起始建议，不是已验证配置。两个 signed B-bit FFT 分量的平方和需要保留最多 2B 位的无符号中间结果；未经缩放的 24-bit 分量不能直接用 32~40 bit 接收功率。互谱乘加也需保留乘积和加法保护位，再按约定缩放。

累加 K 个有符号 GCC 值至少预留 ceil(log2(K)) 个增长位；28 对对应 5 位。乘法保留完整中间结果，降位采用最近值舍入、半整数远离零，然后对目标类型饱和，禁止溢出回绕；LUT 和浮点/定点参考使用相同规则。概率/置信度 v 的编码为 min(65535, max(0, round(v*65536)))，所以数学上的 1 对应饱和编码 65535，解码值略小于 1。

FFT/IFFT 的级间缩放、截断位置、倒数 LUT 精度和 epsilon 下限需形成配置表。通过定点仿真与模块对拍统计溢出/饱和次数、角度偏差及置信度变化后，才能确定最终位宽。

## 4. PHAT 倒数

模长近似：

$$
M
=
\max(|a|,|b|)+\frac38\min(|a|,|b|)
$$

归一化：

$$
\Phi
\approx
G\cdot LUT\left(\frac1{M+\epsilon}\right)
$$

为提高 LUT 利用率，可先做 leading-zero detection 将 M 归一化到固定区间，再查 reciprocal LUT。

必须恢复归一化指数。模长接近零时使用 epsilon 下限或将该频点置零；绝对值计算需扩位以处理最小负数。近似模长和倒数可能导致分量略超出 Q1.15 范围，输出需按第 3 节饱和并验证误差。

## 5. Delay LUT

离线 Python 计算：

```text
delay_lut[pair][angle]
```

内容包括：

```text
integer_delay
fraction(optional)
```

运行时不需要 CORDIC 计算 sin/cos。

integer_delay 是第 05 章的有符号延迟，不是未经处理的 BRAM 地址；负整数按模 N 映射。分数格式、半整数舍入、pair 顺序、角度零点/正方向和声速必须与算法参考一致。定位有效性和 PSR 到置信度的映射沿用第 05、10 章。

## 6. 软硬件分工

若采用带 ARM/PS 的 Zynq，可按当前检测基线评估以下参考分工：

```text
PL:
采集 + FFT + GCC/PHAT + SRP

PS:
Tiny CNN + 配置 + 相机控制 + UI
```

该分工便于将声学流处理与分类/控制分别验证，但尚未冻结，也未证明其风险最低。最终划分取决于目标板资源、分类模型选型、跨 PL/PS 传输及第 2 节的最坏帧预算；需经定点对拍与板上测量后确定。没有 PS 的 FPGA 平台需要另行评估分类器和控制模块的实现位置。

## 7. 来源

- [R2] Furletov, 2022：低计算量 sector SRP 的实时设计动机。
- [R3] Hu et al., 2026：pair pruning 与局部精搜索降低计算量。
