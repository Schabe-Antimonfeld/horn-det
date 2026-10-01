# 04 GCC-PHAT 完整推导

## 1. 从时间延迟到相关峰

假设理想情况下：

$$
x_j[n]=x_i[n-d]
$$

这里 d>0 表示通道 j 比通道 i 晚到 d 个采样点。全项目沿用第 02 章的定义：\(\tau_{ij}=\tau_i-\tau_j\)，因此本例真实 TDOA 为 \(-d/f_s\)。

互相关定义：

$$
R_{ij}[\ell]
=
\sum_n x_i[n]x_j[n-\ell]
$$

当：

$$
\ell=-d
$$

两信号对齐，相关达到最大，因此：

$$
\hat\ell_{ij}
=
\arg\max_\ell R_{ij}[\ell]
$$

对应 TDOA：

$$
\hat\tau_{ij}=\frac{\hat\ell_{ij}}{f_s}
$$

不要把“j 相对 i 的晚到量 d”与“i 减 j 的 TDOA”混用，否则会使后续方向查表的符号反转。

## 2. 相关的频域表达

根据相关定理：

$$
R_{ij}[\ell]
=
\mathrm{IFFT}
\left\{
X_i[k]X_j^*[k]
\right\}
$$

定义互功率谱：

$$
G_{ij}[k]=X_i[k]X_j^*[k]
$$

本章使用负指数正变换、正指数逆变换。有限长度 FFT 得到循环相关；实际帧经过截断和加窗，不能将理想移位关系视为每一帧都精确成立。实现时只读取物理延迟范围，并按第 05 章将负延迟映射到 IFFT 数组末尾。

## 3. 普通互相关为什么会变宽

若声源在某些频率能量特别强，则 G 的幅值分布会被源频谱形状主导。逆变换得到的相关函数峰可能较宽，并容易被混响干扰。

## 4. PHAT 权重

PHAT 把互功率谱幅度归一化：

$$
\Phi_{ij}[k]
=
\frac{G_{ij}[k]}
{|G_{ij}[k]|+\epsilon}
$$

因此：

$$
|\Phi_{ij}[k]|\approx1
$$

剩下的主要信息是相位差。

GCC-PHAT：

$$
R_{ij}^{PHAT}[\ell]
=
\mathrm{IFFT}
\left\{
\Phi_{ij}[k]
\right\}
$$

或者展开为：

$$
R_{ij}^{PHAT}[\ell]
=
\frac1N
\sum_{k=0}^{N-1}
\Phi_{ij}[k]
e^{j2\pi k\ell/N}
$$

## 5. 为什么相位包含延迟

若：

$$
x_j(t)=x_i(t-\tau)
$$

根据傅里叶时移性质：

$$
X_j(f)=X_i(f)e^{-j2\pi f\tau}
$$

于是：

$$
X_i(f)X_j^*(f)
=
|X_i(f)|^2e^{j2\pi f\tau}
$$

PHAT 去掉 |X_i|²：

$$
\Phi_{ij}(f)
\approx
e^{j2\pi f\tau}
$$

对它做逆傅里叶变换理论上得到：

$$
\delta(t+\tau)
$$

这里 \(\tau\) 是 j 相对 i 的晚到时间，\(\tau_{ij}=-\tau\)。因此相关峰位于 \(t=-\tau=\tau_{ij}\)，与阵列几何及 SRP 查表一致。冲激形式仅对应理想、完整有效频带；有限带宽下的峰会展宽。

## 6. 限制搜索范围

由阵列几何可知：

$$
|\tau_{ij}|
\le
\frac{d_{ij}}{c}
$$

因此无需在全部 IFFT 点找峰，只搜索：

$$
|\ell|
\le
\left\lceil\frac{d_{ij}f_s}{c}\right\rceil
$$

## 7. PHAT 的实际问题

PHAT 会把低能量频点也归一化到近似单位幅度，因此纯噪声频点可能被放大。后续 Horn-aware SRP 会加入频点可靠性权重 W[k]：

$$
\widetilde\Phi_{ij}[k]
=
W[k]\Phi_{ij}[k]
$$

这也是目标频带增强的数学入口。

## 8. FPGA 实现

互功率谱：

$$
(a+jb)(c-jd)
=
(ac+bd)+j(bc-ad)
$$

需要 4 个实乘法；通过三乘法复数乘也可进一步节省 DSP。

PHAT 模长：

$$
|G|=\sqrt{Re(G)^2+Im(G)^2}
$$

可近似为：

$$
|G|
\approx
\max(|a|,|b|)+0.375\min(|a|,|b|)
$$

再通过 reciprocal LUT 求近似倒数。

此处模长近似中的 a、b 指 Re(G)、Im(G)，与前面的复乘输入变量不同。近似倒数和定点截断可能使分量超出表示范围，需按第 09 章执行饱和并与浮点参考对拍。

## 9. 来源

- [R1] Grinstein et al., 2024，第 2.4 节：CC、GCC-PHAT、TDOA。
- [R3] Hu et al., 2026：户外车辆定位中 GCC-PHAT 作为低频粗定位模块。
