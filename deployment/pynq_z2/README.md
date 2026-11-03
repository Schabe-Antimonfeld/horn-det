# 最新版鸣笛算法与可复现交付

main只维护这一版运行算法：14路/48 kHz/240 mm圆阵，PCM16 → buffered Conditioning → Hann/折叠FFT → 全网格SRP → 候选FFT波束 → Log-Mel → BandTemporalMLP-30866 → 事件及保守声画关联。模型标签405187002，方向门限0.95、抓拍门限0.90、分差0；窗口48128、更新24064采样，完整5580方向/91对、65抽头波束。Hann融合、重叠帧及逐方向输出复用已合并到各自模块，不再通过历史版本层层继承。

整理前源码保存在分支`codex/archive-2026-11-03`，提交`a8cda7b`；原多相机版本仍在`legacy`。训练、PANNs/Tiny候选、旧配置和实验入口只在归档分支维护。本地主动忽略的原始数据、旧outputs和虚拟环境没有删除，也不包含在该Git分支快照内。

## 快速复现

电脑复现使用Python3.12、uv和已有GCC。无需下载训练数据、安装PyTorch或连接FPGA。解压交付包或在本目录执行：

```text
uv sync --frozen
uv run --frozen python reproduce.py --output outputs/reproduction
uv run --frozen pytest
```

Windows若GCC不在PATH，可在reproduce命令追加`--compiler D:/MinGW/bin/gcc.exe`。uv首次同步需要网络或已有依赖缓存，包内没有Python、uv、编译器和依赖轮子。依赖锁定NumPy2.5.3、pytest9.1.1及其测试依赖，均为原项目已有依赖。源码包含四个C核，必须在当前机器编译；不把Windows DLL复制到ARM。

`reproduce.py`按顺序校验资产SHA-256、编译或验证四个库、运行两段PCM回放、比对整理前冻结的完整事件证据，再用固定权重重新推理672个场景的冻结特征并计算指标。成功报告为`outputs/reproduction/report.json`，含环境、库/数据哈希、模型标签、事件和评测结果；输出目录必须是新目录。

## 结果与证据范围

| 协议 | 场景 | 联合Precision | 联合Recall | 联合F1 |
|---|---:|---:|---:|---:|
| 主测试360° | 168 | 0.9167 | 0.5238 | 0.6667 |
| 主测试可见区域 | 168 | 0.8913 | 0.4881 | 0.6308 |
| 压力360° | 168 | 0.9333 | 0.5000 | 0.6512 |
| 压力可见区域 | 168 | 0.9048 | 0.4524 | 0.6032 |

联合指标要求识别鸣笛并选对方向；独立检测指标和抓拍计数也在报告中。672个场景通过已冻结的640维候选特征重新推理，不重新合成全部阵列声场；两段示例回放另行覆盖PCM至事件的完整链路。正例由已探索公开录音构造场景重复得到，静音为生成样例；它们不是新道路测试，也不能验证连续道路事件召回或误报/小时。来源、许可和缓存边界见[DATA.md](DATA.md)。

整理验收（2026-11-03）：主仓库147项测试通过；算法ZIP在独立解压目录建立全新uv环境、从源码编译四库，并以Python隔离模式运行复现、自检及130项算法测试，均通过。此结果只属于当前AMD64电脑，不证明ARM500 ms或实物视觉闭环。

## 日常运行

```text
uv run --frozen python scripts/build.py
uv run --frozen python run.py --check
uv run --frozen python run.py --replay examples/horn-array14.wav --output outputs/replay
uv run --frozen python scripts/evaluate.py --output outputs/evaluation.json
uv run --frozen python scripts/package.py --output outputs/horn_algorithm.zip
```

只保留`config/default.json`与`model/`中的一套固定权重。修改阵列校准、相机和硬件路径时保留模型前端及窗口协议；配置校验拒绝旧模型/旧搜索后端。编译器记录的源码或二进制哈希变化时，不覆盖旧库，使用`build.py --directory NEW_DIRECTORY`或`reproduce.py --libraries NEW_DIRECTORY`。同样源码的已有库会重做数值自检，重复复现无需重编译。参数变化不是新版本，实际门限变动须另行验证。

## 上板与维护

ARM服务入口仍为`run.py`，使用已有PYNQ镜像的Python/NumPy并在板上执行`python scripts/build.py`；电脑uv锁文件面向Python3.12，不在旧PYNQ镜像强装NumPy2.5.3。PS源码保留Python3.6语法，旧镜像实际兼容性和500 ms目标仍待实机验收。PL采集、DMA、SRP协议及Tiny CNN RTL在源码仓库中保留，当前固定时间MLP运行在PS，不能塞入Tiny CNN核。算法压缩包不含bit/hwh、RTL或视觉权重。

GC2093为单颗固定广角摄像头，不覆盖360°；视觉默认关闭，实测镜头标定、外参、时间同步和车辆模型通过验收后才能开启。保留画外、遮挡和同方向多车的歧义。上板步骤见[HANDOFF.md](HANDOFF.md)，性能口径见[BOARD_PERFORMANCE.md](BOARD_PERFORMANCE.md)，相机资料见[CAMERA.md](CAMERA.md)。
