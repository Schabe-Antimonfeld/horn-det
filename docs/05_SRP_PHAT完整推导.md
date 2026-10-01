# 05 SRP-PHAT 完整推导

## 1. 从“先估 TDOA”转向“直接扫描方向”

GCC 两步法为：

```text
每对麦克风独立找 GCC 峰
→ 得到多个 TDOA
→ 再解几何方程
```

如果某个麦克风对的峰选错，错误会直接进入后续定位。

SRP 的思想是：

> 不先决定每个麦克风对的唯一 TDOA，而是对每个候选方向，直接读取各麦克风对在该方向理论 TDOA 处的相关值，然后全局累积。

## 2. 候选方向理论延迟

二维方向：

$$
\mathbf u(\theta)=
\begin{bmatrix}
\cos\theta\\
\sin\theta
\end{bmatrix}
$$

第 p 个麦克风对对应 i,j：

$$
\tau_p(\theta)
=
\frac{(\mathbf r_j-\mathbf r_i)^T\mathbf u(\theta)}{c}
$$

离散延迟：

$$
q_p(\theta)
=
\tau_p(\theta)f_s
$$

## 3. SRP 空间响应

连续形式：

$$
P(\theta)
=
\sum_{i<j}
R_{ij}^{PHAT}
\left(
\tau_{ij}(\theta)
\right)
$$

离散实现：

$$
P[\theta]
=
\sum_{p=1}^{P}
R_p
\left[
\operatorname{round}(q_p(\theta))
\right]
$$

8 麦克风共有：

$$
P=\binom82=28
$$

个麦克风对。

这里 R_p 使用有符号延迟坐标，与第 04 章的 \(\tau_{ij}=\tau_i-\tau_j\) 一致。若原始 IFFT 输出数组为 r_p[0..N-1]，整数延迟 q 的读取地址为：

$$
\operatorname{index}(q)=((q\bmod N)+N)\bmod N,\qquad R_p[q]=r_p[\operatorname{index}(q)]
$$

例如 N=1024 时，q=-3 对应地址 1021，q=3 对应地址 3；不要直接将负延迟作为无符号地址。每对的有效延迟范围由其基线长度决定。LUT 生成与硬件需使用相同的半整数舍入规则。

## 4. 为什么真实方向会形成峰

若候选方向等于真实方向，则理论延迟：

$$
\tau_{ij}(\theta)
$$

与各对 GCC-PHAT 的真实峰位置一致，所以 28 个相关值会同时较大。

如果候选方向错误，只有少数麦克风对可能偶然取到高值，累加结果较低。

因此：

$$
\hat\theta
=
\arg\max_\theta P(\theta)
$$

## 5. 从 beamforming 角度理解

令相对于公共参考的传播延迟为 \(\tau_m(\theta)\)，对应采样延迟 \(d_m(\theta)=f_s\tau_m(\theta)\)。若将不同通道按候选方向的采样延迟补偿后相加：

$$
y_\theta[n]
=
\sum_m x_m[n+d_m(\theta)]
$$

真实方向时，各通道相干叠加，输出功率最大。非整数采样位置需要分数延迟插值，秒单位的 tau 不能直接加到采样索引 n。SRP 的“Steered Response Power”名称来源于此。

## 6. 分数延迟

理论 q 通常不是整数：

$$
q=n+\delta,\quad 0\le\delta<1
$$

取 \(n=\lfloor q\rfloor\)、\(\delta=q-n\)，可以使用线性插值：

$$
R(q)
\approx
(1-\delta)R[n]+\delta R[n+1]
$$

第一版 FPGA 可先使用整数延迟，第二版再增加分数插值。

负分数延迟也使用向下取整，例如 q=-2.3 时 n=-3、δ=0.7；两端点分别按第 3 节映射地址。截取 GCC 缓存时需保留所有插值端点，不能在缓存边界外读取。

1° 是候选角度的搜索步长，不代表可达到 1° 定位精度。整数延迟量化会使相邻角度共享延迟组合，实际误差还受频带、噪声、阵列标定和多径影响，必须通过实验报告。

## 7. 粗细两级搜索

全范围：

```text
-90° ~ +90°, 1° step → 181 points
```

可先粗搜：

```text
5° step → 37 points
```

得到粗峰 θc，再只搜索：

```text
θc ± 5° 或 ±8°, 1° step
```

Furletov 的博士论文明确采用 GCC-PHAT 预选约 10° 扇区，再执行 SRP-PHAT，从而减少全 360° 搜索开销。

上述 5° 粗网格 SRP 与第 06 章的少量麦克风对 GCC 预选扇区是两种粗定位方式。局部搜索区间需裁剪到配置范围；粗定位不可信或局部结果异常时按第 06 章回退。

## 8. 置信度

只输出 argmax 不够。定义主峰 P1，排除主峰邻域后的旁瓣均值和标准差为：

$$
\mu_s,\quad\sigma_s
$$

Peak-to-Sidelobe Ratio：

$$
PSR
=
\frac{P_1-\mu_s}{\sigma_s+\epsilon}
$$

若 PSR 低，说明多个方向竞争或声场不明确。

PSR 是无量纲评分，既不是概率，也不天然位于 0~1。对外置信度采用本项目的工程映射：

$$
C=\operatorname{clip}\left(\frac{PSR-a}{b-a},0,1\right),\qquad b>a
$$

a、b 和定位有效性门限仅在验证集上确定，测试时固定；不能将 C 当作经过概率校准的正确率。主峰排除邻域、旁瓣统计范围和 epsilon 也需固定并记录。局部窗口的 PSR 不能反映窗口外的竞争峰，因此粗定位可信度必须单独通过检查，且不同搜索模式需分别验证门限。

有效旁瓣点不足、无有效频点、有效麦克风对不足或响应平坦时，应输出 doa_valid=0、C=0，不能将任意 argmax 作为有效方向。第 10 章定义对外编码。

## 9. 来源

- [R1] Grinstein et al., 2024：SRP 的时域/频域统一定义与工程变体。
- [R2] Furletov, 2022，第 4.3.1 节：GCC 预选扇区 + SRP-PHAT 局部搜索。
