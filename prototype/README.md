# Python 算法原型

使用 Python 3.12 和 uv，在本目录运行 `uv sync`、`uv run pytest`。模型阶段使用 `uv sync --extra learning` 安装 CPU PyTorch 与 scikit-learn。依赖版本由 `uv.lock` 固定。

音频统一为 `(samples, channels)`，频谱使用负指数正变换，麦克风对按 `i < j` 排序，TDOA 为 `tau_i - tau_j`，0° 指向 +x，正角转向 +y。参数入口是 `src/horn/config.py`。

| 阶段 | 文档 | 状态 |
|---|---|---|
| 参数、几何、校准 | 01、02 | 已实现，13 项测试通过 |
| 音频和 STFT | 03 | 已实现，累计 16 项测试通过 |
| 阵列仿真 | 11 | 已实现，累计 20 项测试通过 |
| GCC、SRP、搜索 | 04、05、06 | 已实现，累计 39 项测试通过 |
| Horn-aware 权重 | 08 | 已实现，累计 42 项测试通过 |
| 检测、训练和候选 | 07 | 已实现，累计 54 项测试通过；候选选型待确定 |
| 事件与相机接口 | 10 | 已实现，累计 59 项测试通过 |
| 定点和对拍导出 | 09 | 待实现 |
| 流水线及评价 | 01、12 | 待实现 |

数值测试与合成数据只证明参考实现的行为，不代表道路精度。真实音频、预训练权重与输出分别放在被忽略的 `data/`、`weights/`、`outputs/` 中；不提交虚拟环境、数据或模型二进制。

`frontend.STFT.feed` 接收任意大小的同步 PCM 块，只输出完整帧，不居中、不补未来采样；使用文档的对称 Hann。时间戳是末采样索引换算后的整数微秒。`read_audio` 使用多相滤波重采样，固定选取通道或保留多通道，分类功率融合不做波形相加。

`simulation` 提供带保护边界的分数延迟远场、线性 RIR 卷积与可复现的域随机化。RIR 是一阶镜像声源加随机指数衰减尾部的工程近似，不是完整房间求解器；white/engine/traffic/siren/wind 是合成噪声代理。实际录音可由调用方替换，原录音已有噪声和混响时不得将再处理的样本标成干净受控声场。

`localization.Search` 支持整数/分数延迟、全网格、粗细 SRP 和 GCC 扇区预选。粗方向使用多方向大基线、峰显著性与延迟残差检查；搜索中 pair 集合固定。局部失败扩大扇区，再回退到配置范围的粗细搜索，必要时全细网格。无能量、无频点、pair 不足、几何退化或平坦响应均输出无效。当前 PSR/置信度及残差门限只是演示配置，必须在验证集冻结后才可用于实路。

`weighting` 按第 08 章实现训练集先验、退化回退、谱减比例和线性总功率/噪声比门控。`NoisePSD.initialize` 必须由调用方明确指定无鸣笛背景，未就绪时权重为零；当前帧先生成权重再更新，触发帧和活动事件冻结。三个因素可分别关闭为 1，支持文档六组消融。默认固定频带与门限仅用于流程演示，尚未选型。

检测实现包括 HTK 三角 Mel（无面积归一化）、自然对数、13 维正交 DCT MFCC 的均值/标准差、因果 PCEN、FFT 候选规则、线性 SVM 验证集概率校准、小型 MLP、Tiny CNN、单向 GRU CRNN 与温度 KL 蒸馏。Tiny CNN 使用 stride=1、same padding 和 ReLU，共 1834 个参数；这些是文档未冻结位置的工程选择。CRNN 卷积只跨频率，`sequence` 显式返回 GRU 缓存，可分块等价运行；每条新录音重置缓存。神经模型默认不引入额外归一化，SVM 标准化只拟合训练集。PCEN 模型需独立训练。

数据 manifest 使用 JSONL，路径相对 manifest：`{"path":"audio.wav","group":"event-001","event":"event-001","origin":"recording-001","split":"train","label":1,"intervals":[[0.2,0.5]],"channel":0}`。split 可为 train/validation/test/unassigned；先划分关联的 group/event/origin，再切窗。正例必须提供穷尽的鸣笛区间，重叠不足 20 ms 的边缘窗口跳过，具体阈值待验证；负例无区间。不同手机录音仍使用相同事件标识。元数据缺失时不得声称无泄漏，不能仅凭文件名推测事件关系。公开数据许可由使用方在导入前核查。

迁移候选采用独立实现的 PANNs Cnn6 兼容编码器（[R8](../docs/13_文献来源映射与复现边界.md)），支持冻结及末层微调。真实权重来自 [Zenodo 3987831](https://zenodo.org/records/3987831)，文件 `Cnn6_mAP=0.343.pth`，23,698,321 bytes，MD5 `e25e26b84585b14c7754c91e48efc9be`，记录许可 CC BY 4.0。已核对上游 `pytorch/models.py` 的 commit `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`；不执行上游代码。前端使用 32 kHz、periodic Hann、1024 FFT、320 hop、reflect center、权重内置的 64-bin Mel 矩阵和 dB 压缩，并与权重内置 STFT 卷积核对拍。短窗适配只使用已完成的历史窗口，仍需重新训练鸣笛头并测量冷启动；不等同于共享 STFT 的主线，也不默认满足部署资源。离线分类数据固定选单通道，权重放在忽略目录。

`events` 实现迟滞与帧计数状态机，`Probability` 和 `Direction` 必须关联同一帧，候选分配 ID，END 保留 ID，IDLE 为零。默认状态门限是待验证的演示设置。wire 是本原型的联调提案：小端 `<BHhHQI`，19 bytes；首字节 bits 0–1=state、bit 2=horn_valid、bit 3=doa_valid，其余保留为零。Q0.16 和 Q9.7 按文档舍入/饱和。`camera` 只产生转向、抓拍、保存或不可达动作，支持反向云台、偏角和机械范围，抓拍按事件去重；模拟器记录转向到位后的曝光时间。实际设备通信和时钟同步需硬件联调。
