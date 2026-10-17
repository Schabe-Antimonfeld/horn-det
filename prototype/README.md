# Python 算法原型

使用 Python 3.12 和 uv，在本目录运行 `uv sync`、`uv run pytest`。完整模型与测试使用 `uv run --extra learning pytest`，CPU PyTorch 与 scikit-learn 在 learning extra 中；依赖版本由 `uv.lock` 固定。以下含模型的命令均携带 `--extra learning`，防止 uv 自动同步时移除可选依赖。

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
| 定点和对拍导出 | 09 | 已实现，累计 65 项测试通过 |
| 流水线及评价 | 01、12 | 已实现，累计 77 项测试通过；225 样本矩阵与端到端演示已运行 |

数值测试与合成数据只证明参考实现的行为，不代表道路精度。真实音频、预训练权重与输出分别放在被忽略的 `data/`、`weights/`、`outputs/` 中；不提交虚拟环境、数据或模型二进制。

`frontend.STFT.feed` 接收任意大小的同步 PCM 块，只输出完整帧，不居中、不补未来采样；使用文档的对称 Hann。时间戳是末采样索引换算后的整数微秒。`read_audio` 使用多相滤波重采样，固定选取通道或保留多通道，分类功率融合不做波形相加。

`simulation` 提供带保护边界的分数延迟远场、线性 RIR 卷积与可复现的域随机化。RIR 是一阶镜像声源加随机指数衰减尾部的工程近似，不是完整房间求解器；white/engine/traffic/siren/wind 是合成噪声代理。实际录音可由调用方替换，原录音已有噪声和混响时不得将再处理的样本标成干净受控声场。

`localization.Search` 支持整数/分数延迟、全网格、粗细 SRP 和 GCC 扇区预选。粗方向使用多方向大基线、峰显著性与延迟残差检查；搜索中 pair 集合固定。局部失败扩大扇区，再回退到配置范围的粗细搜索，必要时全细网格。无能量、无频点、pair 不足、几何退化或平坦响应均输出无效。当前 PSR/置信度及残差门限只是演示配置，必须在验证集冻结后才可用于实路。

`weighting` 按第 08 章实现训练集先验、退化回退、谱减比例和线性总功率/噪声比门控。`NoisePSD.initialize` 必须由调用方明确指定无鸣笛背景，未就绪时权重为零；当前帧先生成权重再更新，触发帧和活动事件冻结。三个因素可分别关闭为 1，支持文档六组消融。默认固定频带与门限仅用于流程演示，尚未选型。

检测实现包括 HTK 三角 Mel（无面积归一化）、自然对数、13 维正交 DCT MFCC 的均值/标准差、因果 PCEN、FFT 候选规则、线性 SVM 验证集概率校准、小型 MLP、Tiny CNN、单向 GRU CRNN 与温度 KL 蒸馏。Tiny CNN 使用 stride=1、same padding 和 ReLU，共 1834 个参数；这些是文档未冻结位置的工程选择。CRNN 卷积只跨频率，`sequence` 显式返回 GRU 缓存，可分块等价运行；每条新录音重置缓存。神经模型默认不引入额外归一化，SVM 标准化只拟合训练集。PCEN 模型需独立训练。

数据 manifest 使用 JSONL，路径相对 manifest：`{"path":"audio.wav","group":"event-001","event":"event-001","origin":"recording-001","split":"train","label":1,"intervals":[[0.2,0.5]],"channel":0}`。split 可为 train/validation/test/unassigned；先划分关联的 group/event/origin，再切窗。正例必须提供穷尽的鸣笛区间，重叠不足 20 ms 的边缘窗口跳过，具体阈值待验证；负例无区间。不同手机录音仍使用相同事件标识。元数据缺失时不得声称无泄漏，不能仅凭文件名推测事件关系。公开数据许可由使用方在导入前核查。

迁移候选采用独立实现的 PANNs Cnn6 兼容编码器（[R8](../docs/13_文献来源映射与复现边界.md)），支持冻结及末层微调。真实权重来自 [Zenodo 3987831](https://zenodo.org/records/3987831)，文件 `Cnn6_mAP=0.343.pth`，23,698,321 bytes，MD5 `e25e26b84585b14c7754c91e48efc9be`，记录许可 CC BY 4.0。已核对上游 `pytorch/models.py` 的 commit `d2f4b8c18eab44737fcc0de1248ae21eb43f6aa4`；不执行上游代码。前端使用 32 kHz、periodic Hann、1024 FFT、320 hop、reflect center、权重内置的 64-bin Mel 矩阵和 dB 压缩，并与权重内置 STFT 卷积核对拍。短窗适配只使用已完成的历史窗口，仍需重新训练鸣笛头并测量冷启动；不等同于共享 STFT 的主线，也不默认满足部署资源。离线分类数据固定选单通道，权重放在忽略目录。

`events` 实现迟滞与帧计数状态机，`Probability` 和 `Direction` 必须关联同一帧，候选分配 ID，END 保留 ID，IDLE 为零。默认状态门限是待验证的演示设置。wire 是本原型的联调提案：小端 `<BHhHQI`，19 bytes；首字节 bits 0–1=state、bit 2=horn_valid、bit 3=doa_valid，其余保留为零。Q0.16 和 Q9.7 按文档舍入/饱和。`camera` 只产生转向、抓拍、保存或不可达动作，支持反向云台、偏角和机械范围，抓拍按事件去重；模拟器记录转向到位后的曝光时间。实际设备通信和时钟同步需硬件联调。

`fixed` 提供半整数远离零舍入、饱和、完整中间乘积、整数 radix-2 FFT/IFFT、10-bit-address 倒数 LUT、近似模长 PHAT、PSD 更新、三个频率因子、Mel/log LUT、整数/分数延迟及 SRP。参考配置 FFT 每级右移 1 bit，前向总缩放 1/N，IFFT 总缩放 1/N，FFT/GCC 为 signed Q9.15（24 bits），PHAT 为 Q1.15，交叉谱保留 signed 49-bit/Q19.30 中间量，功率保留 unsigned 48-bit 中间量，融合后 POWER 为 unsigned Q10.30（40 bits）；实际噪声同功率尺度。28 对累加保留至少 5 增长位。log 下限设为一个 POWER code（2^-30），这是需要量化验证的工程选择。导出 `.npz`、逐行十六进制 `.mem` 与配置/缩放/饱和 JSON；输入是已校准的一帧。FFT IP 的级间算法不同会产生不同编码，联调必须匹配本配置或据 IP 重建参考。CNN 的 `quantize_model` 仅提供对称权重量化误差对照，不是 PL 整数推理内核。

## 执行

在 `prototype/` 执行；`main.py` 是源码启动入口，不依赖安装包。

```powershell
uv run --extra learning pytest
uv run python main.py audit
uv run --extra learning python main.py demo --output outputs/demo
uv run python main.py benchmark --output outputs/benchmark
uv run python main.py export-golden --output outputs/golden
```

`demo` 生成分组独立的合成训练/验证/测试录音，训练 SVM 和 Tiny CNN，统计先验、比较 8-bit 权重量化，再跑已知单声源阵列事件。全部产物保留在 `outputs/demo/`；它只是可复现的工作流验证，分类容易、监控时间短，不能外推精度或误报率。`benchmark` 覆盖文档的 9 角度 × 5 SNR × 5 噪声代理，共 225 个样本，比较 GCC、SRP、六组频率消融及整数/扇区/剪枝对照；默认先验来自另一组固定种子的合成训练源。报告包含逐样本编号、无效数量、MAE/MedAE/P90/SR10、CPU 时间与回退比例，背景 PSD 是受控的纯背景统计，不代表在线估计已经通过道路验证。

导入实际音频需先准备 manifest，确认录音来源与许可：

```powershell
uv run python main.py split --manifest data/unassigned.jsonl --output data/split.jsonl --seed 0
uv run python main.py prepare --manifest data/split.jsonl --frontend logmel --output data/logmel.npz
uv run --extra learning python main.py train --dataset data/logmel.npz --model tiny --epochs 30 --output weights/tiny.pt
uv run --extra learning python main.py evaluate-model --dataset data/logmel.npz --model weights/tiny.pt --quantize 8 --output outputs/classifier.json
uv run python main.py learn-prior --dataset data/logmel.npz --output weights/prior.npz
```

已有官方分组/折划分时保留它，直接使用指定的 train/validation/test manifest，跳过 `split`。该命令只接受 unassigned，并按关联来源的连通组划分，不自动推测 HornBase 文件名或替代 UrbanSound8K 官方折。训练仅访问 train，模型挑选和 SVM 校准使用 validation；评价选择的概率阈值写入报告，测试时固定。新 road 数据需另列来源，不能把训练源的增强版本当成测试源。

候选仍为**待确定**，分别重训比较：

```powershell
uv run python main.py prepare --manifest data/split.jsonl --frontend mfcc --output data/mfcc.npz
uv run --extra learning python main.py train --dataset data/mfcc.npz --model svm --output weights/svm.npz
uv run --extra learning python main.py train --dataset data/mfcc.npz --model mlp --output weights/mlp.pt
uv run python main.py prepare --manifest data/split.jsonl --frontend pcen --output data/pcen.npz
uv run --extra learning python main.py train --dataset data/pcen.npz --model tiny --output weights/pcen.pt
uv run --extra learning python main.py train --dataset data/logmel.npz --model crnn --output weights/crnn.pt
uv run --extra learning python main.py fetch-pretrained --output weights/Cnn6.pth
uv run --extra learning python main.py prepare --manifest data/split.jsonl --frontend panns --pretrained weights/Cnn6.pth --output data/panns.npz
uv run --extra learning python main.py train --dataset data/panns.npz --model transfer --pretrained weights/Cnn6.pth --output weights/teacher.pt
uv run --extra learning python main.py train --dataset data/panns.npz --model transfer --pretrained weights/Cnn6.pth --fine-tune --output weights/teacher-finetuned.pt
uv run --extra learning python main.py train --dataset data/logmel.npz --model tiny --teacher weights/teacher.pt --teacher-dataset data/panns.npz --output weights/student.pt
```

教师和学生输入可有不同前端，但必须来自同样的文件与采样窗口，`identities/groups/splits/labels` 逐项对齐后才允许蒸馏。Cnn6 的前端需要显式单通道 manifest。CRNN CLI 默认使用每个上下文从零状态开始的因果窗口分类，保持训练/推理一致；`sequence` 提供连续状态缓存，若启用 `Detector(..., recurrent=True)`，需用连续序列重新训练与验证，不能假定两种状态策略等价。

阵列文件运行与相机策略模拟：

```powershell
uv run --extra learning python main.py run --input data/array.wav --model weights/tiny.pt --prior weights/prior.npz --background-prefix 0.3 --output outputs/run
uv run --extra learning python main.py run --input data/array.wav --model weights/tiny.pt --mode srp --realtime --labels data/events.json --continuous --output outputs/paced
uv run --extra learning python main.py run --input data/stereo.wav --model weights/tiny.pt --mode detect --output outputs/detection
```

`--background-prefix` 是调用方明确确认无鸣笛的起始时长；不知道时保持 0，此时 Horn-aware 无有效 PSD，定位无效。事件标签 JSON 格式为 `[[start_seconds,end_seconds],...]`，只有 `--continuous` 才计算 FP/hour。`--realtime` 只是按采样时钟节奏播放 WAV，记录候选/确认输出延迟，不是实际 ADC 或相机测量；离线输出时钟不可与虚拟输入时钟相减。相机曝光指标单独标为 simulated。

按第 08 章，检测概率与空间峰不能自动证明同属鸣笛。默认 `doa_valid=0`，数值定位留在逐帧 `numerical_direction` 中；已知单声源仿真或已完成外部归属验证的条件才使用 `--association-verified` 启用定位输出及模拟转向/抓拍。这个开关记录验证前提，不实现多声源识别。多声源自动归属仍待验证，不能仅以 PSR 高就认定声源是鸣笛。

`--profile` 接收 JSON，字段为 `config/search/search_mode/fusion/camera/gate/weighting/calibration/channels/coordinates`，`search_mode` 可为 full/coarse/sector。未给出时使用明确记录的演示默认值，例如：

```json
{
  "search_mode": "sector",
  "fusion": {"beta": 0.8, "on": 0.7, "off": 0.3, "confirm": 3, "release": 5, "timeout": 100},
  "search": {"interpolation": true, "selected_pairs": 8},
  "calibration": {"gains": [1,1,1,1,1,1,1,1], "delays": [0,0,0,0,0,0,0,0]},
  "camera": {"offset": 0, "sign": 1, "lower": -90, "upper": 90}
}
```

流式校准使用因果 65-tap sinc，多 32 sample 滤波延迟；有通道误差时再加公共补偿延迟，均记录在运行报告。训练提取与在线前端均使用同一因果去直流滤波，已验证带 DC 偏置时 MFCC/Log-Mel/PCEN 的特征一致。结果包括逐帧 JSONL、wire binary、事件、相机模拟曝光和统计 JSON。模拟抓拍锁定请求时的角度，等待曝光期间不被后续转向改写，多个事件按独立目标排队；这个调度策略尚需实际相机联调。无模型时只能产生候选，不能确认鸣笛；普通双声道用 detect 模式，不能作为已知几何的阵列定位数据。

## 可调环形多相机与下视相机原型

`src/horn/surround.py` 提供理想针孔投影、地面求交、覆盖采样、多相机路由、360°高位来声 SRP、候选波束、冻结 PANNs 接入，以及带曝光时间和视觉速度的车辆关联。原有 `camera.py` 单云台策略、Config 默认值及 19-byte wire 保持兼容。详细公式和参数见[第 10 章](../docs/10_相机控制接口.md)。

在仓库根目录使用现有环境运行：

```powershell
.\prototype\.venv\Scripts\python.exe -B prototype/simulate_surround.py --output prototype/outputs/surround_v2
.\prototype\.venv\Scripts\python.exe -B prototype/simulate_surround.py --geometry-only --cameras 8 --height 8 --pitch 45 --hfov 90 --vfov 70 --output prototype/outputs/surround_custom
.\prototype\.venv\Scripts\python.exe -B -m pytest prototype/tests/test_surround.py -q -p no:cacheprovider
```

或在本目录执行 `uv run --extra learning python simulate_surround.py ...`。`--help` 列出 n、高度、俯角、视场、朝向、分辨率、喇叭高度及监控距离参数。`--profile` 的 rig/camera_overrides/acoustics 格式见第 10 章；命令行优先于 rig，逐相机覆盖最后应用。输入和输出路径相对命令执行目录。

默认仅为演示：6＋1 相机、h=6 m、环形俯角40°、视场90°×70°、下视120°×100°、1280×720、监控1～40 m。完整声学使用14路/0.24 m/48 kHz，2°方位×3°俯角搜索，每次一秒完成窗。候选数默认2可配置；0.2分数阈值、0最小分差继承旧实验，在新场景仍待验证。没有动态候选扩展或跨块在线状态集成。

完整仿真要求本地 `outputs/horn_detector_v1/panns-horn.pt`、`clips.json` 和原始音频有效；不会下载或重训，缺失则失败。这些训练权重和录音不随Git分发；新环境可先用 `--geometry-only`，它不加载PyTorch或录音，使用解析注入方向/响应检查接口。当前默认输出 `outputs/surround_v2/`，含36组覆盖扫描、176个独立关联场景、58个声学场景及三种方法对照。旧v1产物保留，不与v2配置混用。

车辆图像输入是理想长方体投影得到的框和地面锚点，并可加入像素误差、框漏检、朝向标定误差。移动速度由两个模拟锚点反投影估计；未运行真实检测/跟踪模型。地面锚点不可用框中心替代。静止声学闭环与独立移动关联分开报告，不能外推实际鸣笛归属；同方向目标漏检可产生错误唯一匹配。

默认 `associate` 要求 `AcousticEvidence` 包含同一音频窗的鸣笛候选俯角与SRP网格。缺证据或响应不足返回unmatched，峰过宽返回ambiguous；unique只表示当前证据兼容的单个簇，不是已证实的鸣笛身份。水平角旧行为必须显式设置 `Association(method="azimuth")`，仅用于对照。`source_unexplained=true` 表示没有兼容车辆，不能直接解释为确定漏检。全部门限可在profile的association段调整，格式见第10章，尚未实路校准。

当前相机无畸变/滚转，覆盖采样不计遮挡和像素尺寸。真实检测器、完整标定、相机I/O、视频缓存、硬件同步和动态声场待完成，不能将合成SVG当作实拍。声学无法区分的同方向近邻车辆、PANNs选错波束或未建模误差仍可能导致错配，首轮小样本零错配不是一般性保证。

## 当前验证证据与交付边界

完整测试包含真实 Cnn6 权重前端对拍（需先 fetch，否则该可选集成测试明确 skip），模型训练/保存/恢复、分组与事件匹配、因果流式等价和整数饱和。已执行 Tiny CNN、SVM、其他候选训练路径、迁移冻结/微调、端到端合成事件、225 个匹配样本与 1024-point golden 导出。文档本地链接、引用映射、参数、代码注释与空白检查使用 `audit`。

默认种子 0 的定位矩阵结果：

| Route | Valid/Total | MAE（有效子集） | P90 | SR10（全部样本） |
|---|---|---|---|---|
| GCC-PHAT | 182/225 | 0.484° | 1° | 80.89% |
| SRP-PHAT | 225/225 | 1.627° | 5° | 97.33% |
| Horn-aware SRP | 225/225 | 0.444° | 1° | 99.56% |

GCC 的较小 MAE 伴随 43 个无效结果，不能只按有效子集误差选型。当前 NumPy 实现的全网格搜索比 GCC 扇区搜索更快；减少评分次数不保证 Python 耗时下降，也不能推断 FPGA 结果。测试矩阵并未涵盖真实多径、安装误差和道路竞争声源，所有候选、门限和位宽仍待实测冻结。

算法代码与运行路径已覆盖文档中的参考原型范围。后续已有本地真实公开录音训练的Tiny CNN/PANNs实验，权重保留在被忽略的outputs中；这不等于来源完整独立的道路验收或生产模型。早期训练计划及上表77项测试为历史基线记录。

另有[完整Tiny CNN整数参考、RTL及DMA接口](../fpga/README.md)，其Icarus逐位对拍与本目录的 `quantize_model` 权重量化误差对照是不同证据；不能继续将已有RTL描述为尚未实现。尚未完成的是全链路厂商IP/时序/板上对拍、资源/功耗/Fmax、真实道路连续误报与声源归属、实际相机通信/同步。`fpga_resources`仍为空；PC仿真和合成车辆框不替代这些验收。源码、锁文件和测试提交Git，真实数据、模型、环境及生成报告保留本地。
