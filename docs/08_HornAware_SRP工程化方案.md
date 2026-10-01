# 08 Horn-aware SRP 工程化方案

## 1. 设计目标

标准 SRP-PHAT 关注“最强空间一致声源”，但项目需要“汽车鸣笛声源”。道路中发动机、轮胎、风噪、警笛等会竞争空间峰。

因此加入目标特定频率权重。

频率权重只能增强符合先验的成分，不能保证在鸣笛与其他声源同时出现时定位峰一定属于鸣笛。需要在多声源竞争样本上验证；证据不足时不输出有效定位。

## 2. 与 Wang Sub-SRP 的关系

Wang et al. 的公开摘要能够确认：其 Sub-SRP 是针对车辆鸣笛，利用鸣笛频率特征，通过 filtering 和 spectral subtraction 抑制非 horn 噪声，再进行 SRP 类定位。

但当前未取得该论文完整公开 PDF，因此本节公式是**本项目工程化设计**，不是对论文原公式的逐式复现。

## 3. 背景噪声 PSD

使用多通道平均功率：

$$
P[k,l]
=
\frac1M\sum_m|X_m[k,l]|^2
$$

初始化时使用已确认无鸣笛的背景帧估计 N；背景估计尚未就绪时不宣称 SNR 权重或定位有效。每帧先用上一帧的噪声估计计算权重，再判断是否更新。在 IDLE 且当前帧未触发能量候选、通道有效时更新噪声：

$$
N[k,l]
=
\alpha N[k,l-1]+(1-\alpha)P[k,l]
$$

取 \(0\le\alpha<1\)。触发候选的当前帧以及 CANDIDATE、CONFIRMED、END 状态均冻结更新，回到满足背景条件的 IDLE 后恢复，防止在等待 CNN 确认期间把 horn 学成噪声。N[k,l] 表示处理第 l 帧后的估计，本帧权重使用 N[k,l-1]。

## 4. 谱减

估计目标剩余能量：

$$
S[k,l]
=
\max\left(P[k,l]-\lambda N[k,l-1],0\right),\qquad\lambda\ge0
$$

将谱减剩余能量转换为保留比例：

$$
W_{sub}[k,l]=\frac{S[k,l]}{P[k,l]+\epsilon}
$$

该比例位于 0~1，并在第 6 节进入最终权重。本方案保留原始互谱相位，以实数权重实现谱减抑制，不直接修改时域音频。

频点总功率与噪声功率之比：

$$
\xi[k,l]
=
\frac{P[k,l]}{N[k,l-1]+\epsilon}
$$

xi 是线性功率比，不是 dB，也不是纯目标信号与噪声的功率比；T0、T1 必须使用相同定义。P、N、S 需使用相同功率尺度，epsilon 使用对应尺度的正数下限。

## 5. 鸣笛先验权重

从 HornBase 训练集统计 horn 与 non-horn 平均对数谱：

$$
\mu_H[k]
=
E[\log(P[k]+\epsilon)\mid horn]
$$

$$
\mu_N[k]
=
E[\log(P[k]+\epsilon)\mid nonhorn]
$$

定义区分度：

$$
D[k]=\mu_H[k]-\mu_N[k]
$$

将 D 映射到 0~1：

$$
W_H[k]
=
\mathrm{clip}
\left(
\frac{D[k]-D_{min}}{D_{max}-D_{min}},0,1
\right)
$$

Dmin、Dmax 仅从训练集确定并满足 Dmax>Dmin；若区分度退化，使用已在验证集评估的固定频带回退，不能除以零。权重能否比固定带通更稳健需要消融验证，不能由该映射形式直接推出。

## 6. SNR 权重

可采用分段线性权重，取 T1>T0，门限在验证集确定后固定：

$$
W_{SNR}[k,l]
=
\begin{cases}
0,&\xi<T_0\\
\frac{\xi-T_0}{T_1-T_0},&T_0\le\xi<T_1\\
1,&\xi\ge T_1
\end{cases}
$$

最终：

$$
W[k,l]
=
W_H[k]\cdot W_{sub}[k,l]\cdot W_{SNR}[k,l]
$$

谱减保留比例和 SNR 门控作用相关，联合使用可能过度抑制弱鸣笛，需分别消融。关闭某个模块时将其权重设为 1；不能计算 S 却不接入最终权重，同时声称启用了谱减。

## 7. 加权 PHAT

标准 PHAT：

$$
\Phi_{ij}[k,l]
=
\frac{X_i[k,l]X_j^*[k,l]}
{|X_i[k,l]X_j^*[k,l]|+\epsilon}
$$

Horn-aware PHAT：

$$
\widetilde\Phi_{ij}[k,l]
=
W[k,l]\Phi_{ij}[k,l]
$$

加权 GCC：

$$
R^H_{ij}[q,l]
=
\mathrm{IFFT}\{\widetilde\Phi_{ij}[k,l]\}
$$

最终：

$$
P_H(\theta,l)
=
\sum_{i<j}
R^H_{ij}
\left(q_{ij}(\theta),l\right)
$$

$$
\hat\theta
=
\arg\max_\theta P_H(\theta,l)
$$

qij 沿用第 05 章的有符号延迟、整数地址映射和可选插值。全频谱 IFFT 需保持权重实数且满足 W[N-k]=W[k]，以保持互谱的共轭对称；使用实数逆变换时按接口提供半谱。若所有有效频点被抑制或空间响应不满足第 05 章条件，输出 doa_valid=0，不能对全零响应取任意方向。

## 8. 为什么这个形式 FPGA 友好

最终增加的运算主要是：

```text
每频点噪声 PSD IIR、谱减和功率比/LUT
+ 每频点组合 WH、Wsub、WSNR
+ 每个有效 pair 的互谱实部/虚部分别乘共享权重
```

不需要矩阵特征分解。权重生成可以跨 pair 共享，但施加权重的乘法随 pair 数量增长；倒数 LUT、PSD 存储和访存也应纳入第 09 章预算。

## 9. 推荐消融实验

```text
A: SRP-PHAT
B: SRP-PHAT + fixed horn band
C: SRP-PHAT + HornBase learned weight
D: C + spectral-subtraction retention weight
E: C + noise PSD/SNR gate
F: C + both noise weights
```

这样可分别评价先验、谱减比例和 SNR 门控，再验证联合效果。各组使用相同样本、几何参数和搜索设置，频率权重只从训练集学习，门限只在验证集调整。

## 10. 来源

- [R1] Grinstein et al., 2024：频率加权、鲁棒 SRP 的理论框架。
- [R4] HornBase：用于估计 horn 频谱先验。
- [R5] Wang et al.：公开摘要确认 horn frequency characteristics + filtering + spectral subtraction + Sub-SRP 思想；完整论文未附。
