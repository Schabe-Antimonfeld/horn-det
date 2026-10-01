# 03 STFT 与频域前端

## 1. 为什么需要分帧

道路声音非平稳，但在几十毫秒内可近似局部平稳。因此使用短时傅里叶变换把长时间音频转成连续时频帧。

## 2. STFT 定义

第 m 个麦克风离散信号为：

$$
x_m[n]
$$

第 l 帧：

$$
x_{m,l}[n]=x_m[n+lH],\quad 0\le n<N
$$

加窗后：

$$
x'_{m,l}[n]=x_{m,l}[n]w[n]
$$

FFT：

$$
X_m[k,l]
=
\sum_{n=0}^{N-1}
x_m[n+lH]w[n]e^{-j2\pi kn/N}
$$

## 3. Hann 窗

$$
w[n]
=
\frac{1}{2}
\left(1-\cos\frac{2\pi n}{N-1}\right)
$$

窗函数用于降低有限帧截断造成的谱泄漏。

## 4. 参数选择

推荐：

$$
N=1024,\quad H=512,\quad f_s=48000
$$

帧长：

$$
T_f=\frac{1024}{48000}\approx21.33\ \mathrm{ms}
$$

更新周期：

$$
T_h=\frac{512}{48000}\approx10.67\ \mathrm{ms}
$$

更新周期表示稳态下新帧到达的间隔，不等于从声音发生到检测结果输出的延迟。因果实现中，第 l 帧必须收到采样点 lH+N-1 后才能完成组帧；启动时需先收集约 21.33 ms 音频，再进行 FFT 和后续处理。分类上下文、状态平滑和确认规则还会增加等待，具体见第 07、12 章。

频率分辨率：

$$
\Delta f=\frac{48000}{1024}=46.875\ \mathrm{Hz}
$$

## 5. 共享频谱前端

同一个 X_m[k,l] 同时供给：

```text
Horn classifier
Cross-spectrum / PHAT
Noise PSD estimator
Horn-band weighting
```

因此不应为识别和定位各做一套 FFT。

## 6. 多通道同步要求

Grinstein 等指出，TDOA 定位依赖相对传播时延，因此多通道 ADC 必须同步。固定数字滤波延迟也必须一致或标定。

## 7. FPGA 帧调度

Hop=512 时，每秒每路帧数：

$$
F_{frame}=\frac{48000}{512}=93.75
$$

8 通道前向 FFT 数：

$$
8\times93.75=750\ \mathrm{FFT/s}
$$

这给出了前向 FFT 的平均调用需求，可作为单个流水 FFT IP 时分复用的预算输入。是否可行还取决于 IP 配置、FFT/IFFT 切换、存储带宽、背压和其他模块的调度，需按第 09 章验证最坏情况下的每帧处理预算。

## 8. 来源

- [R1] Grinstein et al., 2024：频域帧、DFT/STFT、TDOA 定义。
- [R2] Furletov, 2022：自动驾驶声学系统中的频域处理与分类/定位链。
