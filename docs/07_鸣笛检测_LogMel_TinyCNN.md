# 07 鸣笛检测：Log-Mel + Tiny CNN

## 1. 任务定义

定位算法只能回答“声音来自哪里”，不能可靠回答“是不是汽车鸣笛”。因此保留独立二分类器：

$$
P_{horn}=P(\text{horn}|X)
$$

本章 Log-Mel + Tiny CNN 作为当前算法参考基线。第 8 节列出的替代模型、前端和训练方式均为**待确定部分**，需通过同一数据划分上的实验选择，尚未替换项目主线。

## 2. 多通道特征融合

不要直接把 8 路时域信号相加，因为未知 DOA 会产生相位相消。

先计算各通道功率谱：

$$
P_m[k,l]=|X_m[k,l]|^2
$$

再平均：

$$
P[k,l]
=
\frac1M\sum_{m=1}^{M}P_m[k,l]
$$

这样分类器得到相对方向无关的频谱描述。

## 3. Mel 频率

Mel 映射常写为：

$$
m(f)=2595\log_{10}\left(1+\frac f{700}\right)
$$

对线性功率谱施加 B 个三角滤波器：

$$
E_b[l]
=
\sum_k H_b[k]P[k,l]
$$

取对数：

$$
L_b[l]
=
\log(E_b[l]+\epsilon)
$$

推荐：

```text
B = 40
context = 24~32 frames
```

采用第 03 章 N=1024、H=512、48 kHz 参数时，C 帧上下文覆盖的音频跨度为：

$$
T_{context}=\frac{N+(C-1)H}{f_s}
$$

24 帧约覆盖 266.67 ms，32 帧约覆盖 352 ms。冷启动且不补帧时，至少收集这么长的音频才能组成首个完整分类输入；补帧策略若启用，需在训练和推理中保持一致并单独验证。

稳态因果滑窗只使用当前及历史帧，可每 10.67 ms 更新一次。上下文跨度不等于每个鸣笛事件都需等待相同时长，也不意味着确认延迟只有 10.67 ms；实际延迟取决于事件在窗口中的位置、特征/CNN 处理、平滑和确认规则，应按第 12 章测量。

## 4. Tiny CNN

比赛版可用：

```text
Input: 40 × 24 × 1

Conv 3×3, 16
DWConv 3×3 + PWConv 1×1, 24
DWConv 3×3 + PWConv 1×1, 32
Global Average Pooling
FC 32→2
Softmax
```

### 4.1 完整参考结构与计算量

当前 [Python 实现](../prototype/src/horn/models.py)固定采用 stride=1、same padding、每次卷积后 ReLU、不使用 BatchNorm/Dropout、最后全局均值池化。PyTorch 张量顺序是 `N×1×40×24`，40 是频率轴、24 是时间轴；Softmax 的类别顺序是 `[non-horn, horn]`。该结构是项目工程基线，不声称复现 HornBase 或其他论文原网络。

| 层 | 输出，不含 batch | 参数，含 bias | MAC/窗口 |
|---|---|---:|---:|
| 普通 3×3，1→16 | 16×40×24 | 160 | 138,240 |
| 深度 3×3，16 groups | 16×40×24 | 160 | 138,240 |
| 逐点 1×1，16→24 | 24×40×24 | 408 | 368,640 |
| 深度 3×3，24 groups | 24×40×24 | 240 | 207,360 |
| 逐点 1×1，24→32 | 32×40×24 | 800 | 737,280 |
| 全局均值 | 32 | 0 | 表中不计加法与除法 |
| 全连接，32→2 | 2 | 66 | 64 |
| **合计** | | **1,834** | **1,589,824** |

每 10.67 ms 执行一次时，约为 149.046 M MAC/s，不包含 STFT、Mel、对数、激活、池化和传输。参数较少不等于 ARM/Python 或单 MAC FPGA 自动满足实时。浮点参数约 7.16 KiB，最大单份浮点特征图约 120 KiB；实现还需要缓存和中间值。

网络不以麦克风数为输入通道数。14 路先按第 2 节功率平均，再生成一份 40×24 特征，因此从 8 路改为 14 路不需要将 CNN 第一层改成 14 通道。真实麦克风、功率融合和抽取滤波造成的数据分布变化仍须重新验证。

### 4.2 前端的精确约定

输入 PCM16 先除以 32768，得到与 `read_audio` 相同的归一化浮点幅度；DC 去除和增益/延迟标定只执行一次。去 DC 使用 `y[n]=x[n]-x[n-1]+0.995*y[n-1]`，冷启动状态为零，连续块间保留状态。采用不居中、不补未来样本的 1024 点对称 Hann、512 hop、48 kHz STFT；浮点 FFT 不除以 N。各有效通道先平方取功率再平均，不做未知方向的波形相加。

Mel 使用 HTK 标度、0–24 kHz、40 个三角滤波器、不做面积归一化；取 `ln(E+1e-12)`，不使用 dB、log10、额外标准化或 PCEN。最近 24 个完整特征帧按频率优先排列，不能把 24×40 转置后直接发送。冷启动需要 12,800 个采样、约 266.67 ms，不能用未来补帧缩短。

当前 PDM 采集滤波器在 0–16 kHz 平坦、16–24 kHz 为过渡带，公开录音和板端数据的高频分布可能不同。第 09 章整数 FFT 又具有 1/N 缩放；若以后用 PL FFT 代替此浮点前端，必须补偿功率尺度或按新前端重训，不能直接将缩放后的 Log-Mel 送入旧模型。

## 5. HornBase 数据集

HornBase 2024 是公开汽车鸣笛专用数据集，共 1080 个 1 秒 WAV：

```text
540 horn
540 non-horn
44.1 kHz
16 bit
Stereo
```

数据采集考虑：

- 开窗/关窗；
- 发动机开/关；
- 音乐与谈话；
- 车辆前后/侧方相对位置；
- 短鸣、长鸣、连续短鸣。

论文的技术验证使用 spectrogram + CNN，100 次模型训练平均准确率约 89%。这一数字只说明数据集可用于分类验证，不应直接当作本项目预期精度。

原始 44.1 kHz 音频应离线重采样到项目配置的 48 kHz，再按同一套 STFT/Mel 参数提取训练特征。先划分原始录音或来源组，再生成滑窗与增强样本，避免相邻窗口泄漏到测试集；详见第 12 章。

HornBase 原文说明，数据来自接收车辆内的两部手机，同一次鸣笛被裁成多个 1 秒窗口。双声道特征可选择固定通道或先分别计算功率谱再平均，处理策略需在训练、验证和测试中固定；不能将这些录音视为本项目 8 通道阵列定位数据。

划分时需将同一原始鸣笛事件的不同裁剪窗口和不同设备录音归入同一集合，并利用文件名与来源信息核查。1 秒片段的 horn 标签不保证其所有短子窗口都含有鸣笛，需要核查鸣笛区间。车内手机录音与路侧阵列存在设备及声学环境差异，最终选型还需路侧数据验证。

### 5.1 完整训练与选型流程

1. 核查 HornBase、道路录音和困难负例的来源与许可；HornBase 的非商业许可限制不能忽略。HornBase/普通立体声用于分类，不当作已知几何的 14 路阵列定位录音。
2. 为每条录音建立原始来源、事件和分组，标注穷尽的鸣笛区间。同一事件、来源及其增强版本保持在同一划分；来源不清时保留证据限制。没有官方划分时，现有 `split` 按独立连通组约 60/20/20 划分，而非保证片段数量比例。
3. 划分后才重采样、切窗和增强。增强仅作用于训练来源，可在准备 manifest 前生成；当前训练命令不自动执行道路混音增强。至少覆盖短鸣、长鸣、警笛、刹车/机械音、发动机、人声/音乐、风及纯背景。
4. 用当前因果前端提取特征。现有短窗标签最小鸣笛重叠 20 ms 只是演示选择，边缘不确定窗口会跳过，需按短鸣目标验证。不要把 1 秒正例的全部短窗自动标正。
5. 当前训练器使用 Adam、交叉熵、默认 batch=32、lr=1e-3；按验证损失保存最佳权重。30 epochs 可作为起始实验，不是已调优配置。固定来源划分与随机种子，比较 MFCC/SVM 基线以判断 CNN 的额外收益。
6. 在验证集确定概率阈值、平滑与事件规则，冻结后报告测试集。分别评价浮点 CNN、仅权重量化和完整整数 CNN；后二者不能混称。硬件缩放由 train/validation 校准，禁止读取 test 来选尺度。

在 `prototype/` 执行。先将以下内容保存到被忽略的 `data/cnn-profile.json`；它明确目标前端和 14 路几何，训练录音仍可按 manifest 选择单通道或功率融合。

```json
{
  "config": {
    "microphones": 14,
    "diameter": 0.24,
    "sample_rate": 48000,
    "frame": 1024,
    "hop": 512,
    "mel_bands": 40,
    "context": 24,
    "epsilon": 1e-12
  }
}
```

```powershell
uv run python main.py split --manifest data/unassigned.jsonl --output data/split.jsonl --seed 0
uv run python main.py prepare --manifest data/split.jsonl --profile data/cnn-profile.json --frontend logmel --output data/cnn.npz
uv run --extra learning python main.py train --dataset data/cnn.npz --model tiny --epochs 30 --batch-size 32 --learning-rate 0.001 --seed 0 --output weights/tiny.pt
uv run --extra learning python main.py evaluate-model --dataset data/cnn.npz --model weights/tiny.pt --output outputs/cnn/float.json
uv run --extra learning python ../fpga/cnn.py export --checkpoint weights/tiny.pt --dataset data/cnn.npz --calibration-split validation --output outputs/cnn_model --vectors 4
uv run python ../fpga/cnn.py verify --model-dir outputs/cnn_model
uv run python ../fpga/cnn.py evaluate --model-dir outputs/cnn_model --dataset data/cnn.npz --split validation --threshold 0.5 --output outputs/cnn/integer-validation.json
uv run python ../fpga/cnn.py evaluate --model-dir outputs/cnn_model --dataset data/cnn.npz --split test --threshold 0.5 --output outputs/cnn/integer-test.json
```

已有官方分组时跳过 `split`。以上 0.5 是命令示例，正式阈值应在整数模型的验证集上选定并冻结，再使用同一个阈值执行最终 test；不要从测试结果反复修改网络。`evaluate-model` 输出包含各划分的窗口指标，不等于已经完成连续事件验收。

### 5.2 当前证据与必交报告

当前代码已实现训练、保存/恢复、完整整数参考、权重导出和 RTL 对拍；未完成真实 HornBase/路侧正式训练，没有可声称有效的生产权重或道路精度。`cnn.py demo` 的随机未训练权重只验证数据通路，不可作为鸣笛检测器部署。

正式交付需同时保存：来源及许可、分组/区间标注、前端配置、训练/校准划分、权重与 manifest 哈希、整数输入/激活饱和数量、浮点和整数概率差异、冻结阈值、测试集分类结果、连续录音事件 Precision/Recall/F1、误报/小时、短鸣召回及事件延迟。缺少连续负样本时不计算或外推误报/小时。

## 6. 状态平滑

单窗输出容易抖动。指数平滑：

$$
\bar p_l
=
\beta\bar p_{l-1}+(1-\beta)p_l
$$

再使用迟滞：

```text
ON  threshold = Ton
OFF threshold = Toff
Ton > Toff
```

固定并记录 beta、Ton、Toff、确认/释放所需帧数及最大候选持续时间。参数仅由训练集和验证集确定；平滑、连续帧确认和推理更新频率共同影响事件延迟，不能只报告 CNN 运行时间。

## 7. 状态机

```text
IDLE
  ↓ energy candidate
CANDIDATE
  ↓ CNN confirm
CONFIRMED
  ↓ release
END
```

未通过确认或超过最大候选持续时间的 CANDIDATE 进入 END，随后回到 IDLE；CONFIRMED 在释放条件满足时进入 END。每次进入 CANDIDATE 分配 event_id，同一事件内保持不变。冷启动上下文不足时不能宣称 CNN 已确认。

CANDIDATE 且 doa_valid=1 时可以让相机提前开始转向；CONFIRMED 且 doa_valid=1、置信度满足控制门限时才正式抓拍。分类确认与定位有效性分开输出，具体见第 10 章。

### 7.1 FPGA 定点推理完整版

本章基线的 [tiny_cnn.v](../fpga/rtl/tiny_cnn.v)实现五个卷积层、全局均值、全连接与二分类概率。首版固定 40×24，不支持直接将 context 改为 32；其他候选、前端或网络结构必须重新实现并验证。

采用 signed INT8 权重、signed INT16 特征、signed 48-bit bias/累加器。每个张量使用二次幂尺度：实值为整数编码乘以 `2^-F`。导出器按校准范围和权重幅度选 F，并给出每层非负右移量 `Fin+Fw-Fout`；降位半整数远离零舍入，ReLU 后饱和，禁止回绕。bias 与乘积具有相同小数位；导出时核算最坏点积不超 48 bit。GAP 对 960 个非负特征求和，最近值舍入后除以 960，保留原小数位。

二分类 Softmax 等价于 `sigmoid(logit_horn-logit_nonhorn)`。RTL 用区间 [-8,8]、间隔 1/64 的 1025 项 LUT，输出 unsigned Q0.16；超出区间使用端点值。其近似误差与整个网络的量化误差分开报告。阈值和平滑使用相同的 Q0.16 解码值，不能混用编码除以 65535。

实现为 8 条复用乘加通路，特征按 `channel mod 8` 分 bank；两份特征缓存各分配 8×4096×16 bit，总分配位容量 128 KiB。按常见 BRAM36 组织约占 32 块，仅指这两份数据 RAM，不含权重、池化、DMA、采集或工具映射差异。不能据此宣称完整系统已装入 XC7Z020。

功能仿真中每窗计算 445,512 周期；100 MHz 假设下约 4.45512 ms，不含输入准备、DMA、输出背压或系统调度。100 MHz 布线 Fmax、实际 DSP/BRAM 和 PS/PL 总吞吐尚未测得，不把仿真周期换算视为板上实时证据。

### 7.2 FPGA 输入输出与模型交付

输入每窗 960 个 uint32 DMA 槽位，低 16 bit 为量化 Log-Mel，高 16 bit 必须符号扩展；顺序为 `f0/t0..23, f1/t0..23, ...`。TKEEP=0xf，只在第 960 字 TLAST=1。它不是原始 PCM 包，也不是 14 通道输入。提前 TLAST、缺失 TLAST、错误 TKEEP 和符号扩展均产生显式错误结果；缺失 TLAST 时先排空到下一 TLAST。输入或输出停顿必须遵守 AXI 握手。

输出固定 8 个 uint32、小端字，仅最后一字 TLAST：

| 字 | 内容 |
|---|---|
| 0 | `0x314e4e43`，ASCII `CNN1` |
| 1 | `0x00010008`，格式 1、8 字 |
| 2 | 本次推理序号，不是声学帧号或时间戳 |
| 3、4 | non-horn/horn 两个 signed INT16 logit，符号扩展至 int32 |
| 5 | bits 15:0 为 Q0.16 鸣笛概率；bit 16 为本窗激活饱和；bit 17 为输入格式错误；其余保留 0 |
| 6 | 输入收齐后到输出准备好的计算周期数，不包含输出背压 |
| 7 | 32-bit 模型标识，必须与 manifest 一致 |

导出包含 8 份权重 bank、8 份 bias bank、缩放配置、概率 LUT、测试输入/期望输出与 manifest。模型标识用于版本核对，不代替 SHA-256 完整性检查；构建脚本先检查各内存文件哈希，再保存模型快照并综合到片上 ROM。首版更新权重需要重新生成 bitstream，不支持运行时任意写权重。数据、权重和生成产物保留在被忽略目录。

### 7.3 PC、PS 与 PL 分工及联调

PC 使用现有 Python/PyTorch 训练与量化校准。PS/PC 首先完成本章前端，PL 只运行已冻结的整数 CNN；现有 `build_cnn.tcl`在 14 路采集工程上添加独立 `cnn_dma`，与 `audio_dma` 共用 HP0。板端 [cnn_host.py](../fpga/cnn_host.py)先提交结果接收，再发送特征，检查格式、序号、模型标识和错误，并保留调用方声学帧号。其 `probability` 方法兼容参考 Detector 的 NCHW 接口。

板端独立 DMA 客户端仅依赖现有 PYNQ/NumPy，可用于回放导出的特征；直接复用整个 prototype 还需确认板端 Python 与 SciPy 等依赖满足原型环境要求。Vivado 2019.2 本身不能证明旧版 PYNQ 镜像可直接运行 Python 3.12 原型。现阶段没有测得 ARM 上连续前端、CNN DMA、采集和相机共存的性能。

持续维护历史特征，冷启动不足 24 帧不确认；不能等候选事件才临时清空并收集上下文。默认事件平滑 beta=0.8、Ton=0.7、Toff=0.3、确认 3 帧/释放 5 帧只是演示参数，EWMA 会额外影响延迟，不能只用“3×10.67 ms”估算。主机必须保持概率、原始 STFT 帧号和时间戳关联；丢包时重置前端上下文及事件状态，不能把不连续 PCM 当连续音频处理。

验收分为整数参考与 RTL 逐位一致、Vivado 综合/时序及资源、板端 DMA 和完整事件评测。CNN 为阳性不证明某个空间峰就是鸣笛；尚未完成声源归属验证时继续保持 `doa_valid=0`，不能仅因 CNN 已搬到 PL 就启用抓拍。

## 8. 备选检测方案与选型（待确定）

**状态：待确定。** 以下内容记录研究候选与实验计划，不代表已验证的精度、延迟或 FPGA 资源配置。不同论文的数据和任务不同，不能直接横向比较其准确率，也不能把通用声音识别结果当成本项目鸣笛性能。

| 候选 | 方法 | 可能用途 | 主要待验证项 |
|---|---|---|---|
| FFT 频带/谱峰规则 | 频带与谱峰检查，可增加谐波和时间一致性规则 | FPGA 快速候选触发 | 不同喇叭、多普勒、发动机/警笛干扰及弱鸣笛漏检 |
| MFCC + 线性 SVM / 小型 MLP | 紧凑频谱特征与轻量分类器 | 传统基线或资源受限实现 | 跨设备泛化、短鸣召回、定点误差 |
| 预训练音频 CNN 迁移学习 | 冻结编码器训练分类头，再选择性微调 | PC 比较基线或 ARM/PS 推理候选 | 模型尺寸、短窗适配与实际运行时间 |
| 蒸馏后的轻量 CNN | 强模型离线指导小模型训练 | 最终部署候选 | 压缩后的事件召回、误报和量化误差 |
| CNN + GRU（CRNN） | 频谱特征与时间状态联合建模 | 时序事件检测候选 | 因果性、事件标注、状态缓存和部署成本 |
| PCEN + 分类器 | 用动态能量归一化替代对数压缩 | 前端对照实验，非独立分类器 | 道路噪声适应性、长鸣保留和定点成本 |

### 8.1 FFT 规则候选（待确定）

[R6] 的公开摘要描述了实时 FFT 频率分析、汽车喇叭常见频带匹配及道路多普勒问题。本项目可共享已有 FFT，用频带能量和谱峰显著性触发 CANDIDATE；谐波与持续时间检查是可增加的工程规则，不声称是该论文完整算法的复现。

规则检测优先作为低成本候选触发，而非直接替代分类确认。若规则门控跳过后续分类器，应对完整链路测量召回，避免只报告通过门控样本的分类结果。频带、能量和持续时间门限均待验证集确定。

### 8.2 MFCC + 轻量分类器（待确定）

建议比较以下传统基线：

```text
Shared power spectrum
→ Mel filterbank → Log → DCT / MFCC
→ Short-window statistics
→ Linear SVM / Small MLP
```

线性 SVM 推理主要是特征向量乘加，便于定点实现；RBF 核 SVM 的开销随支持向量数量增长，不能按线性分类器估算资源。MFCC 维数、统计窗口及是否加入动态特征待实验确定。SVM 分数不是概率，若用于第 10 章的 horn_probability，应在验证集上校准后固定映射。

[R7] 提供包含 car_horn、siren、engine_idling 等类别的 UrbanSound8K，可补充鸣笛和困难负例，并遵守其官方来源分组/折划分及许可。该文献在此支持数据与评测来源，不作为本项目 MFCC + SVM 已有效的证据；MFCC 基线本身仍需实测。

### 8.3 预训练音频模型迁移学习（待确定）

[R8] 的 PANNs 在 AudioSet 上预训练并研究向其他音频任务迁移，可作为以下路线的方法参考：

```text
Pretrained audio encoder
→ Horn classification head
→ Optional partial fine-tuning
```

先冻结编码器训练分类头，再根据验证集选择是否微调。冻结减少训练成本，不减少编码器推理成本；模型输入采样率、频谱前端、类别定义和最小有效上下文必须按实际模型确认。长片段模型不能直接缩短输入后就宣称满足低延迟，需要按因果短窗重新训练和评价。

优先在 PC 上建立比较基线；是否在 ARM/PS 部署取决于目标板实测，尚未确定。不要求将大型预训练网络直接移植到 FPGA。

### 8.4 蒸馏后的轻量 CNN（待确定）

[R9] 研究音频 Transformer 到 MobileNetV3 CNN 的知识蒸馏。本项目可参考其训练思路：

```text
Offline: Strong teacher → Lightweight student training
Online: Lightweight student only
```

教师模型保留在开发电脑上，在线只运行学生模型。教师/学生选择、蒸馏目标、短窗上下文和量化策略均待确定；该路线是通用音频标注方法向鸣笛任务的迁移，不代表文献已验证本项目。只有在迁移模型有效但部署成本过高时，再比较蒸馏的收益。

### 8.5 CRNN 时序检测（待确定）

[R10] 将 CNN 的频谱表示与 RNN 的时间建模结合，用于多声源声音事件检测。本项目可研究 CNN + GRU 对短鸣、长鸣、连续短鸣及警笛干扰的区分能力，具体收益需实验验证。

实时版本需使用单向 GRU 和因果前端，明确状态初始化、跨帧缓存和重置规则；这是工程改造，不声称复现原文网络。若输出帧级起止信息，需要相应事件标注，不能直接用 1 秒片段标签监督所有帧。该方案暂作后续候选，不作为第一版默认实现。

### 8.6 PCEN 前端（待确定）

[R11] 提出 Per-Channel Energy Normalization（PCEN），通过平滑能量估计和动态压缩替代常用对数压缩，并在噪声、远场关键词识别中评价。PCEN 是前端选择，仍需配合 SVM、CNN 或时序分类器。

可在保持分类器结构、数据划分及其余训练配置一致的条件下，比较 Log-Mel 与 PCEN。两组需各自重新训练，不能只在推理时替换前端。平滑时间常数、压缩参数及定点近似待确定，重点检查弱鸣笛、持续长鸣和背景突变；不能直接把关键词识别结论当成鸣笛检测结论。

### 8.7 待确定事项与选型实验

优先比较三组，不同时启用所有候选：

```text
A: MFCC + Linear SVM
B: Log-Mel + Tiny CNN
C: Pretrained lightweight audio CNN + Horn head
```

C 的具体预训练模型尚未确定；PANNs 可作为较强的参考基线，不能默认视为轻量部署模型。若 C 明显改善检测但资源不满足要求，再增加蒸馏实验。PCEN 作为独立前端对照，FFT 规则作为独立门控对照；CRNN 按时序需求另行比较。

按照第 12 章统一来源分组、训练/验证/测试划分及事件匹配规则，比较事件级 Precision、Recall、F1、误报/小时、短鸣召回率和确认延迟，并记录模型尺寸、乘加量、目标设备运行时间和量化误差。误报/小时需要连续监控音频，不能从平衡片段集直接推算。

仍待确定：

- 最终分类器、前端、特征参数及推理更新频率。
- 短窗标注方式、路侧补充数据和困难负例组成。
- 候选门控、确认/释放门限及 SVM 概率校准。
- ARM/PS 或 PL 的部署位置、模型量化和资源预算。

算法原型与训练可先用 Python，完成实验后再冻结交付参数。所有检测方案只判断声音中是否存在鸣笛，不自动解决鸣笛与某个 DOA 峰的归属；多声源匹配仍需按第 08 章联合验证。

## 9. 来源

- [R4] Dim et al., 2024, HornBase。
- [R2] Furletov, 2022：交通声分类、CNN 与 car horn 类别实验。
- [R6] Dim et al., 2020：FFT 频带鸣笛检测应用；新增内容仅依据已核对题录与公开摘要。
- [R7] Salamon et al., 2014：UrbanSound8K 数据与城市声音评测来源。
- [R8] Kong et al., 2020：PANNs 预训练音频模型与迁移学习方法。
- [R9] Schmid et al., 2023：Transformer 到轻量 CNN 的音频知识蒸馏方法。
- [R10] Çakır et al., 2017：CRNN 多声源声音事件检测方法。
- [R11] Wang et al., 2017：PCEN 前端方法，原任务为关键词识别。

题录维护于 [references.bib](../references/references.bib)，来源核对程度和复现边界见 [第 13 章](13_文献来源映射与复现边界.md)。
