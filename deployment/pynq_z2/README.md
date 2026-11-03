# 最新版鸣笛算法与可复现交付

main只维护这一版运行算法：14路/48 kHz/240 mm圆阵，PCM16 → buffered Conditioning → Hann/折叠FFT → 全网格SRP → 候选FFT波束 → Log-Mel → BandTemporalMLP-30866 → 事件及照片声源候选点投影。模型标签405187002，鸣笛及方向门限0.90、抓拍门限0.90、单窗确认、分差0；窗口48128、更新24064采样，完整5580方向/91对、65抽头波束。Hann融合、重叠帧及逐方向输出复用已合并到各自模块，不再通过历史版本层层继承。

整理前源码保存在分支`codex/archive-2026-11-03`，提交`a8cda7b`；原多相机版本仍在`legacy`。训练、PANNs/Tiny候选、旧配置和实验入口只在归档分支维护。本地主动忽略的原始数据、旧outputs和虚拟环境没有删除，也不包含在该Git分支快照内。

## 快速复现

电脑复现使用Python3.12、uv和已有GCC。无需下载训练数据、安装PyTorch或连接FPGA。解压交付包或在本目录执行：

```text
uv sync --frozen
uv run --frozen python reproduce.py --output outputs/reproduction
uv run --frozen pytest
```

Windows若GCC不在PATH，可在reproduce命令追加`--compiler D:/MinGW/bin/gcc.exe`。uv首次同步需要网络或已有依赖缓存，包内没有Python、uv、编译器和依赖轮子。依赖锁定NumPy2.5.3、pytest9.1.1及其测试依赖，均为原项目已有依赖。源码包含四个C核，必须在当前机器编译；不把Windows DLL复制到ARM。

`reproduce.py`按顺序校验资产SHA-256、编译或验证四个库、运行两段PCM回放、比对冻结的数值与方向证据，并单独验证当前单窗确认状态及0.90门限，再用固定权重重新推理672个场景的冻结特征并计算指标。成功报告为`outputs/reproduction/report.json`，含环境、库/数据哈希、模型标签、事件和评测结果；输出目录必须是新目录。

## 结果与证据范围

| 协议 | 场景 | 联合Precision | 联合Recall | 联合F1 |
|---|---:|---:|---:|---:|
| 主测试360° | 168 | 0.8983 | 0.6310 | 0.7413 |
| 主测试可见区域 | 168 | 0.8689 | 0.6310 | 0.7310 |
| 压力360° | 168 | 0.8793 | 0.6071 | 0.7183 |
| 压力可见区域 | 168 | 0.8500 | 0.6071 | 0.7083 |

表中均为当前0.90门限，当前结果见报告joint、历史0.95核对见historical_joint。冻结特征是独立场景，不能评价连续事件或两窗策略。联合指标要求识别鸣笛并选对方向；独立检测指标和抓拍计数也在报告中。672个场景通过已冻结的640维候选特征重新推理，不重新合成全部阵列声场；两段示例回放另行覆盖PCM至事件的完整链路。正例由已探索公开录音构造场景重复得到，静音为生成样例；它们不是新道路测试，也不能验证连续道路事件召回或误报/小时。来源、许可和缓存边界见[DATA.md](DATA.md)。

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

GC2093为单颗固定广角摄像头，不覆盖360°；照片输出默认关闭，实测镜头标定、外参和相机取流验证后才开启。仅画声源候选点，不加载YOLO或其他车辆模型；点不构成车辆归属证明。上板步骤见[HANDOFF.md](HANDOFF.md)，性能口径见[BOARD_PERFORMANCE.md](BOARD_PERFORMANCE.md)，相机资料见[CAMERA.md](CAMERA.md)。

## 照片只标声源候选点

`vision`配置现在只接受`enabled`和`output=acoustic_points`。OpenCV仅用于已有的USB/JPEG读写及画点，不做神经网络推理，不需要车辆权重。未取得实测标定时继续保持`enabled=false`；商品FOV或模拟内参不能替代标定。

相机与阵列中心重合时，声学方位/俯角直接通过镜头模型投到原图，无需车框、车高或地面落点。偏移非零且缺少独立距离时记录`depth_required`，不强行画点；近正下方的方位不确定记录`azimuth_ambiguous`。画外、畸变模型有效域之外不画点。多个候选分别保留，不强选一辆车。相机标定、阵列平面俯角误差和同步误差仍会影响位置。

当前可靠单窗分数≥0.90即设置`horn_confirmed=true`，首次正窗直接进入CONFIRMED；方向门限和抓拍门限均为0.90。照片保存为`event-ID-CAMERA-confirmed.jpg`和`-points.jpg`，每事件/相机/阶段去重。释放仍需两次未达标更新，避免持续事件重复拍照；释放滞回不代表低分窗仍有可靠鸣笛证据，`horn_confirmed`要求当前窗可靠且达标。输入、前端或激活溢出不留图。

画外、多候选、方位不确定和缺少距离导致的视差限制仍保留。声学确认图不能证明某辆车鸣笛，时间同步与标定仍需验证。通用代码仍支持测试双窗及分离抓拍阈值，当前默认仅使用单窗0.90。

## 相机工具

已有板端OpenCV时可用`python scripts/camera.py probe --output outputs/usb-probe`检查USB协商。棋盘采集、镜头标定与独立控制点检查见[CAMERA.md](CAMERA.md)。本轮未安装新依赖，电脑最小复现环境不要求OpenCV；实物JPEG与画点仍需板端已有兼容OpenCV。

## 朝下广角标点模拟

在本目录执行：

```text
uv run --frozen python scripts/simulate_downward.py --source-directory ESC50_AUDIO_DIRECTORY --output outputs/downward_points
```

来源目录需含4-175855-A-43.wav、5-179860-A-43.wav、4-111671-A-42.wav、5-117118-A-42.wav；脚本校验原始哈希，不下载数据。`--geometry-only`只做几何检查，无需原录音。`--height`、`--hfov`、`--vfov`可调，默认6 m、144°、77.6°、垂直向下；不改变生产安装俯角40°和空标定状态。

输出index.html、点标注SVG、coverage.svg、逐例report.json、推理前plan.json及源码快照。场景车体为几何生成，橙点来自真实录音合成14路PCM→当前SRP→波束→固定模型的候选方向，不用真值替代点，也不画车辆框。原始录音不再分发；固定包内PCM回放和672场景评测仍可直接复现。

切换为单窗0.90前的历史结果：2026-11-03相同48场景（24有鸣笛/24无鸣笛）中：12个正例有单窗候选，10个具备画面保存资格，9个有可投影点，1个方位不确定而不画点；另2个候选在画外。无鸣笛场景无触发。两窗严格鸣笛确认仍0/24；新策略增加的是未确认照片，不是鸣笛召回或车辆识别成功率。9个可投影正例最近候选点误差中位15.6 px、最大24.8 px，仅为有点且真值可见子集的条件误差。模拟假设缓存有合时图像，未证明实际USB取流、同步或ARM500 ms。

历史已知重复阳性回放另作接线控制，验证旧CANDIDATE及两窗确认；当前同一回放首窗即确认，不计48场景。历史框底/固定车高诊断保留在本地downward_v3_20261103输出，当前运行与新模拟统一使用声学点。限制及安装检查见[相机说明](CAMERA.md)。

切换前点图验收：174项仓库测试、144窗与旧模拟逐项声学证据对拍、672场景固定权重评测、26份文档74个本地链接及Python3.6语法检查通过。该轮算法包为horn_algorithm_points_20261103.zip；它保留旧两窗0.95设置，不作为当前部署包。实物OpenCV/JPEG链路与ARM速度尚未在本机验证。

当前单窗0.90验收：190项测试通过，两段PCM回放分别首窗CONFIRMED与全程IDLE，672场景当前门限评测及历史分数/指标校验通过。当前部署包为horn_algorithm_single090_20261103.zip；模型权重、窗口、完整网格和点图路径不变，释放仍两窗。


## 前后端监控台

在本目录运行`uv run --frozen python web.py`，打开`http://127.0.0.1:8765`即可通过页面启动鸣笛/静音回放、查看真实分析窗口、停止会话及查询历史照片。前端沿用仓库frontend布局，后端使用标准库HTTP/SQLite，不新增依赖。回放无同步照片时保持空白，不显示素材图或车辆框；实机配置通过验收后可启动PYNQ采集和低频JPEG缓存预览。算法ZIP内置前端，启动命令相同。

计数按事件去重，时间/回放模式和相机缺帧状态明确显示，数据持久保存且不自动清理。局域网实机启动、接口、数据位置和限制详见包内`frontend/README.md`或仓库根目录frontend/README.md。电脑回放与页面联调不能证明ARM500 ms、USB取流或声画标定已通过。

监控台交付包为horn_dashboard_20261104.zip，包含同一算法及frontend；旧single090包保留作为不含监控台的历史交付。213项仓库测试与Chrome桌面/手机联调通过；实机部分仍按HANDOFF验收。
