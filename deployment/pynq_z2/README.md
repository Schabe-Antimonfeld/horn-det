# PYNQ-Z2 独立部署实现

main当前适配单颗GC2093 USB广角摄像头；legacy分支冻结1f11b16。默认相机为wide-0、1920×1080、厂商标称HFOV144°/VFOV77.6°；布局single不附带下视相机，不提供360°视觉覆盖。未标定时仅允许音频运行/名义几何演示；启用视觉必须提供实测calibration，支持brown或fisheye模型，尺寸不符会拒绝运行。相机中心为(offset_x,offset_y,height+offset_z)，height仍是阵列离地高度，yaw/pitch/roll为待实测安装角。

目标：Vivado 2019.2、XC7Z020、14 路/48 kHz/直径0.24 m阵列，Python运行在ARM PS。该目录不依赖PC原型的Python3.12或PyTorch；板端源码保持Python3.6语法兼容，实际镜像版本待合作者确认。

部署路径：PL采集及Tiny CNN，可选批量SRP插值累加；PS连续接收、FFT/PHAT/GCC、方向延时查表、候选波束、事件、相机与空间关联。SRP默认仍由NumPy累加，板上测速后再决定是否开启PL后端。PANNs不属于本板端路径；不能把PC耗时当作ARM实时性能。

默认窗口94包=48128采样，更新47包=24064采样；沿用1024/512对称Hann，仅使用完整帧。训练CNN上下文40×24，与一秒声学定位窗口不同。所有分数门限和可见车辆归属仍需实际场景校准。

商品截图参数、USB/标定验收、固定仿真结果和Tiny CNN限制见[单颗广角相机说明](CAMERA.md)。

## 连续窗口快速版（2026-11-01）

新增`config/band_stream.json`，在原快速SRP与FFT波束上复用连续窗口的重叠数据，目标为板端每窗声学处理≤500 ms。仍为14路、48128采样、24064更新、原模型/门限、91对完整网格；没有新增PL核或减少候选。旧`band_fast.json`作为上一版对照保持不变。

SRP每窗93帧，稳态只算47个新帧，按原12帧分组累加顺序缓存下一窗口需要的互谱和；1024点波束采用整除更新间隔的752采样步长，首窗64块、稳态重算33块。输入频谱复用不要求候选方向稳定；方向相同时再复用有效输出及Mel帧，窗首补零和分数延迟边界重算。epoch、计数、窗口或全部重叠PCM不一致时全量重算，避免跨缺包/跳窗错误复用。C Conditioning保持状态和校准语义，采集线程时间单列。

在目标机新包目录依次执行`python scripts/build_search.py`、`python scripts/build_stream.py`、`python run.py --check`，再用`python scripts/benchmark_stream.py --input synchronized-array14.wav --output NEW_BENCHMARK.json`对比。新库`ps/libstream.so`的ABI=1，原SRP库ABI=2；包内没有开发机DLL/ARM预编译库，不新增依赖或自动安装工具。默认BLAS/OMP单线程，保留手动环境设置。

完整376项回归、1008个构造场景和84个连续混合窗口对拍通过，主/压力联合F1及抓拍计数不变。本机原回放稳态16.09→11.35 ms，阶跃方向压力流17.63→13.68 ms，后者没有输出波形缓存命中。重置路径可能更慢，不能用电脑比例推算上板736→500 ms；当前没有新ARM测速，500 ms仍待验收。配置、计时口径、方向变化结果及PS/PL待办见[500 ms验收说明](BOARD_PERFORMANCE.md)。

新包路径为`outputs/stream_fast_v2_20261101/pynq_z2_stream_fast.zip`；从源码仓库打包使用：

```powershell
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/scripts/package.py --model prototype/outputs/band_improvements_v1/model-combined --config deployment/pynq_z2/config/band_stream.json --output deployment/pynq_z2/outputs/stream_fast_v2_20261101/pynq_z2_stream_fast
```

迁移实际设备时保留已验证的标定、相机、视觉模型及overlay配置。回退可关闭`streaming.enabled`并改`streaming.conditioning=numpy`，或使用原`band_fast.json`；其余搜索/波束/门限不同时改动。原波束与SRP快速包保留，不能用新增复用掩盖采集gap、分析跳窗或模型召回不足。

## 候选波束快速版（2026-10-31）

`config/band_fast.json`在既有快速SRP后接入PS上的批量分块FFT波束形成，`beamforming.implementation=fft`、`fft_size=1024`、`cache_size=64`。14路输入的分块频谱只计算一次，供所有候选方向使用；每个波束在频域合并14路后只做一次逆FFT，方向滤波系数按精确角度缓存，最多64项。候选数量、65抽头Hann窗sinc、分数延迟符号、公共延迟、输出长度、窗尾截断、一次DC处理及模型前端保持不变，不缩短滤波器、不减少麦克风或候选。

这是原FIR的线性卷积计算方式替换，不是STFT相位近似。overlap-save丢弃每块受循环卷积污染的历史段，并单独恢复参考实现中每路在整数延迟前强制为零的窗首规则；遗漏该规则会改变原模型输入。浮点加法次序不同，所以数值在容差内一致，不承诺逐位相同。波束仍在ARM上运行，不新增PL核或DMA协议，也不能直接复用SRP的PHAT频谱代替原始音频频谱。

省略`beamforming`或设`implementation=reference`使用原实现；`cached`仅缓存时延和FIR系数，`native`为独立double C FIR实验后端。默认FFT路径不需要波束C库，也不增加Python依赖。C库ABI=1，Linux名为`ps/libbeam.so`、Windows为`ps/beamforming.dll`，按需执行`python scripts/build_beam.py`，并在配置中显式选择native。已有库可用`python scripts/build_beam.py --check PATH_TO_LIBRARY`复核；错误架构、缺库或ABI不符会失败，不能静默切回其他后端。构建器不覆盖现有文件、不安装工具、不修改系统安全设置。ARMv7实际速度仍需在目标镜像测量。

同一SRP/模型/音频的波束专用比较，在部署包目录执行：

```text
python scripts/build_search.py
python run.py --check
python scripts/benchmark_beam.py --input synchronized-array14.wav --output NEW_BEAM_BENCHMARK.json
python run.py --replay synchronized-array14.wav --output NEW_REPLAY_OUTPUT
```

`benchmark_beam.py`只替换波束后端，报告波束与声学总耗时的中位/P90/最大值、初始化耗时、输入/源码/权重哈希及分数误差，并核对事件判定。额外比较1/2/4个固定候选；`--sweep`比较系数缓存、512/1024点FFT，配合`--native-library`可加入本机C库。此项与`benchmark_search.py`不同：后者保留配置中的波束设置，仅切换SRP；不要把其声学总耗时解释为历史完整原始链路。两个工具都不计采集等待、视觉、日志及冷启动。

可复现实验入口为源码仓库的`prototype/optimize_beam.py`：依次`plan`、`phase --split validation`、`select`、`phase --split test`、`phase --split stress`、`verify`。独立产物位于`prototype/outputs/beam_optimization_v1_20261031/`，不覆盖此前实验；如要让C参与选型，必须在首次plan时提供`--native-library`，不能测试后修改冻结计划。两套协议共336个验证场景预选1024点FFT，波束时间中位约为参考的33.8%，仅缓存约95.9%。这是公开录音构造阵列、复用来源划分的对拍，不是新道路盲测或连续事件性能证明。

验证、主测试、压力测试各336个场景，总计1008个，FFT与已选快速SRP搭配原波束时的方向、检测及抓拍门限计数一致；最大波形误差`5.56e-16`、特征误差`9.60e-14`、分类分数差`2.45e-15`。联合F1如下，比较基线是本轮开始前的快速SRP，不是历史全频SRP：

| 协议 | 原波束 | FFT波束 |
|---|---:|---:|
| 主测试360° | 0.6667 | 0.6667 |
| 主测试可见区域 | 0.6308 | 0.6308 |
| 压力360° | 0.6512 | 0.6512 |
| 压力可见区域 | 0.6032 | 0.6032 |

本机AMD64、NumPy2.5.3、相同48128样本完成窗、相同快速SRP、两个窗口各重复15次，`acoustic-benchmark-final.json`测得波束中位19.63→7.08 ms、P90为22.18→7.97 ms，声学总中位31.44→18.08 ms、P90为33.51→21.13 ms。逐窗配对耗时比例中位分别为0.360和0.601；它们不等于两个中位数直接相除。单独1/2/4波束测得9.89/20.33/42.87→6.37/6.98/9.42 ms，候选越多，共享输入频谱的收益越明显。报告同时记录首窗耗时；不要把稳态数据当冷启动或ARM性能。

C库恢复可读后，52项波束检查全部通过，含C/FFT/缓存、冲激边界、非连续数组、静音、完整窗特征、旧整数模型及事件判定。另作同PCM的C对照：直接FIR的波束耗时约为原实现96%，FFT约43%（`native-sweep.json`），因此没有改动验证阶段已选定的FFT方案。C对照只属于本机数值/速度证据，不声称C参加了上述1008场景评测。ARM上的两种实现仍需实测，不能把“使用C”本身当作加速结论。

事件的`beamforming_diagnostics`记录实际实现、候选数、抽头数、FFT长度及本次缓存命中数；`phase_seconds.beamforming`包括批量频谱计算和窗首修正。新版完整包保存为`outputs/beam_fast_v1_20261031/pynq_z2_beam_fast.zip`，旧SRP包保留。包内仍须先构建SRP库，`native_build_components`明确列出需要构建的组件；默认波束FFT不要求构建`libbeam.so`。模型、声学/视觉证据边界及硬件验收要求见[交接文档](HANDOFF.md)。

## 可选SRP快速版（2026-10-31）

`config/band_fast.json`保留同一时间BandMLP、方向0.95/抓拍0.90、14路、48128样本及24064样本更新，SRP使用PS上的可选C内核，波束使用上节FFT实现。SRP验证集预选为91对、完整5580方向、4倍折叠FFT、single PHAT累加；浮点GCC、延迟查表与空间响应仍由PS产生，不启用PL SRP DMA。仓库原默认配置与 `band_temporal.json` 保留原全频NumPy/原波束基线。

C内核通过ctypes加载本机库，ABI=2，不依赖新的Python包。库缺失、架构不匹配或ABI不符会明确失败，不会悄悄切回慢速路径。运行包只携带 `ps/search_kernel.c` 和构建入口，不复制开发电脑DLL到ARM；须在目标机用已有GCC编译，构建工具不会自动安装编译器。Windows默认生成 `ps/srp_search.dll`，Linux默认生成 `ps/libsrp.so`，并保存源文件/库哈希与数值自检JSON。已有库不覆盖，应在新包目录构建，或用 `--output` 新路径及配置内绝对 `native_library` 调试。

快速包另含 `config/native_exact.json`：全频、double、91对完整网格，同一C库即可运行，可作SRP精度优先对照，波束仍沿用包的配置。此前仅SRP快速版的主/压力时间约为基线5.9%–7.1%，原主协议和可见压力组各少1个正确接受方向；其完整声学本机诊断约为基线24%，含视觉的ARM全流程目标未验收。这些历史数据不包含本轮波束加速。详细数值与证据边界见第12章。

迁移到已有实机配置时，只合并经过验证的search选项与对应模型/门限，保留真实麦克风校准、相机内外参、视觉模型、USB及overlay路径。默认包视觉关闭是因为不含已验证视觉权重；不得通过关闭原有视觉来声称含视觉的全流程提速达标。

在新解压的快速包根目录执行：

```text
python scripts/build_search.py
python run.py --check
python scripts/benchmark_search.py --input synchronized-array14.wav --output NEW_BENCHMARK.json
python run.py --output NEW_LIVE_OUTPUT
```

最后一条仍需要匹配的采集bit/hwh及PYNQ；编译C库不能代替FPGA固件。benchmark默认对最多6个完成窗口各重复3次，先预热，交替执行原NumPy与快速链路；分别报告SRP和完整声学中位/P90/最大值、初始化耗时、比例及0.65秒目标。它排除采集等待、视觉、日志与冷启动，不提供鸣笛事件质量结论；ARM实测输出才是板端速度证据。

源码根目录打包使用：

```powershell
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/scripts/package.py --model prototype/outputs/band_improvements_v1/model-combined --config deployment/pynq_z2/config/band_fast.json --output deployment/pynq_z2/outputs/beam_fast_v1_20261031/pynq_z2_beam_fast
```

包标记 `native_build_required=true`、`native_library_included=false`。不能打包含开发机绝对库路径的调试配置。`search.algorithm`另支持 `cached` / `hierarchical`，`pair_mode`支持all/mixed35，`fft_fold`支持1/2/4/8，`native_precision`支持double/single。区域上界剪枝和粗峰细搜已经实现，但本机验证时C完整网格稍快且证据更完整，故未默认启用层级搜索。固定35对和更激进频率梳因召回损失未通过约束，不以更少打分点冒充更好的方案。

折叠FFT保留全部1024点Hann窗口及全部时间帧，计算频率梳后使用较短GCC周期；它会混叠其他延迟，不是全频PHAT的等价替换。源码验证先于选型，比较内容及复用来源限制见[第12章](../../docs/12_实验设计与评价指标.md)。选型依据是验证集，测试/压力结果不回调参数，不能据PC速度推算含视觉的ARM全流程已经达到6.5秒的5%–10%。

事件新增 `phase_seconds`（srp/beamforming/features/detector）和 `srp_diagnostics`，后者记录实际FFT长度、麦克风对数、完整/粗网格证据、打分方向数、区域剪枝、补搜和全搜回退。`processing_seconds`仍只表示声学处理；抓拍结果的 `capture.processing_seconds` 及 `capture.phase_seconds` 分别记录相机/路由、视觉、关联、标注存图，便于板端分解总延迟。未计算的细网格不填零，也不允许粗证据获得确定车辆归属。

## 最新时间组合模型部署版（2026-10-28）

使用 `config/band_temporal.json` 加载已训练的 `BandTemporalMLP-30866`（640→48→2、模型标签405187002）。ARM PS用NumPy完成时间特征及推理，PL只需14路PDM→PCM采集。方向门限0.95、抓拍门限0.90、分差0；低分抓拍标为 `capture_only`，不确认鸣笛或具体车辆。窗口固定48128采样、更新24064采样，每个候选波束产生七个40×24原始Log-Mel上下文。模型身份、前端、维度和窗口不符会报错；不要将NPZ交给Tiny CNN Verilog。

新版配置明确选择 `spectral` / NumPy SRP，视觉默认关闭。仓库的 `config/default.json` 继续保留旧Tiny默认；打包时显式传入新配置，包内 `config/default.json` 才会采用时间模型。训练权重与生成的zip被Git忽略，仓库源码不包含训练二进制。使用根目录的既有环境：

```powershell
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/run.py --config deployment/pynq_z2/config/band_temporal.json --model prototype/outputs/band_improvements_v1/model-combined --check
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/scripts/package.py --model prototype/outputs/band_improvements_v1/model-combined --config deployment/pynq_z2/config/band_temporal.json --output deployment/pynq_z2/outputs/temporal_deploy_v1_20261028/pynq_z2_temporal_release
```

生成的独立包在自己的根目录运行，不依赖PC源码或PyTorch/SciPy：

```text
python run.py --check
python run.py --replay synchronized-array14.wav --output NEW_REPLAY_OUTPUT
vivado -mode batch -source scripts/build_ps.tcl -tclargs NEW_BUILD BOARD_REPOSITORY synth
```

`build_ps.tcl`复用包内14路采集Verilog及PYNQ-Z2约束，不接收模型目录、不实例化CNN或SRP加速核；输出 `horn_capture.bit/hwh`。bit模式仍须设置实测 `PDM_INPUT_MIN_NS/MAX_NS` 并通过时序、CDC及DRC验收。补齐后将同次bit/hwh放入包内 `firmware/`，再执行 `python run.py --output NEW_LIVE_OUTPUT`；如需可选PL SRP，另按下方旧构建入口生成含 `srp_dma` 的匹配overlay，不可对纯采集固件直接开启FPGA SRP。

本机2026-10-28重测：Vivado2019.2缺少XC7Z020器件支持，纯采集入口又在缺少PYNQ-Z2板卡定义处停止。当前包的 `firmware_included=false`、`board_validated=false`，不含可烧录固件。PS推理与回放接入完成不代表ARM实时或道路效果已验收，后续工作见[交接文档](HANDOFF.md)。

## 可选频带MLP与PCM恢复实验

`detector.backend=spectral` 使用7826参数BandMLP，输入同样是完成的40×24 Log-Mel上下文，PS内部计算保留频带位置的160项统计，再做两层NumPy矩阵乘法；不使用PyTorch、PANNs或CNN DMA。实时采集仍要求与14路接口一致的overlay和PYNQ。默认配置未提升到该研究后端，冻结评测结果与限制见[第12章](../../docs/12_实验设计与评价指标.md)。

在仓库根目录使用本地训练产物（模型/门限必须成套）：

```powershell
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/run.py --config prototype/outputs/horn_recovery_v1/spectral_numpy-config.json --check
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/run.py --config prototype/outputs/horn_recovery_v1/spectral_numpy-config.json --replay PATH_TO_ARRAY14.wav --output NEW_REPLAY_OUTPUT
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/scripts/package.py --model prototype/outputs/horn_recovery_v1/spectral_model --config prototype/outputs/horn_recovery_v1/spectral_numpy-config.json --output prototype/outputs/horn_recovery_v1/standalone-spectral
```

打包器将配置中的模型目录改为包内`model`，检查后端与架构匹配。新实验的Tiny模型同样要求显式提供其`recovered_integer-config.json`，以免沿用旧门限；其配置为慢速整数回放，上板需要改成fpga并使用同模型标签的bit/hwh。Tiny的manifest指定原始或去整体均值Log-Mel前端，旧manifest默认原始前端。

旧 `scripts/build.tcl` 构建采集/CNN/可选SRP，仍需要独立的TinyCNN导出目录；BandMLP的NPZ不能输入TinyCNN Verilog。纯采集可使用新增 `scripts/build_ps.tcl`，不需要CNN权重；如果复用采集/CNN/SRP overlay，BandMLP路径只使用采集及可选SRP，不要求闲置CNN核的模型标签与MLP相同。当前尚无可用目标bit/hwh，运行包明确不含firmware，不称为可直接上板验收的固件。

PC上12个热身后测量窗中，完整NumPy声学＋事件（不含采集、相机、视觉和I/O）BandMLP中位约131 ms、整数软件Tiny约179 ms；这是本机48000样本窗口，不能换算成ARM实时性能或PL CNN时间。新增证据无效/方向歧义语义见[相机说明](CAMERA.md)。

## 分阶段记录

1. 配置及部署合同：已实现；Vivado2019.2启动成功，但缺少XC7Z020器件支持和PYNQ-Z2板卡定义，尚不能创建目标工程。
2. 连续采集、断流处理与DMA生命周期：已实现，148项完整测试通过；真实DMA吞吐待板上测量。
3. 无SciPy/PyTorch的板端声学与几何基线：已实现，152项完整测试通过，DC/标定、SRP、波束、特征与PC参考对拍一致；ARM性能未测量。
4. Vivado构建验证：目标器件支持缺失，不能综合。scripts/build.tcl在既有采集/CNN工程加入srp_batch和独立srp_dma；Python可选后端已接入。单方向核和批量包装通过Icarus及Python报文对拍，尚无目标overlay构建/板上证据。
5. 训练模型整数推理与特征对拍：已实现，155项完整测试及本地真实训练模型RTL对拍通过。整数软件后端供回放验证；板端默认FPGA后端，不加载PyTorch。
6. 相机/视觉适配、事件与独立运行服务：已实现离线回放、PYNQ启动入口、HTTP快照与USB V4L2进程适配、可选OpenCV推理；真实视觉模型/曝光/标定未验证，服务不输出确定车辆归属。无相机帧可重试，成功抓拍按事件去重。
7. 完整打包、回归与合作者交接：脚本支持独立源码/运行包及逐文件SHA256校验；最终验证记录见本地outputs/validation.json，待验收项见[HANDOFF.md](HANDOFF.md)。

各阶段通过针对测试、原测试回归和一致性检查后提交；未通过的硬件验收不标成完成。输出、权重及构建产物在本目录被忽略。

## 构建

`scripts/build.tcl <model_directory> <new_build_directory> <board_repository_or_dash> create|synth|bit`复用既有采集/CNN工程并加入SRP DMA，输出horn_cnn.bit/hwh。原fpga/scripts/build_cnn.tcl直接调用仍只构建采集/CNN。三个DMA共享PS HP0，control_bus为4路输出、memory_bus为5路输入。bit模式仍要求经验证的PDM_INPUT_MIN_NS/MAX_NS，不能编造数值绕过接口检查。构建主机需NumPy及完整模型目录；板端不执行Vivado。

`rtl/srp_accumulator.v`保留原单方向协议：每对依次接收signed24-bit GCC下界、上界的32-bit符号扩展值和低16-bit Q0.16插值系数。默认91对，末系数TLAST；输出4字为标识0x31505253、signed48结果低/高字、错误位，结果保留额外16位小数。

部署工程使用`rtl/srp_batch.v`包装：一个DMA输入包首字为方向数1..128，随后按方向依次传入91对的三字数据，仅整个批次最后一个字置TLAST，TKEEP全部为0xf。输出每方向4字，仅批次末字置TLAST。短包、错误系数/TKEEP/符号扩展或TLAST会报错；Python检查错误位、输出长度及signed48扩展。发生DMA/协议错误后必须重载overlay，不自动切换后端掩盖失败。

`search.backend`可选`numpy`（默认）或`fpga`；后者要求使用本目录构建入口产生并经验证的含srp_dma的bit/hwh。每批取min(block_directions,128)，默认5580方向共44批。GCC采用signed24、22位小数（Q2.22，含符号的整数部分2位），半整数远离零舍入且拒绝溢出；插值系数向下量化Q0.16，浮点端点1直接取上界。每方向误差界为`91/(2*2^22) + sum(abs(upper-lower))/65536`。量化可能改变近似平峰的候选顺序，需道路重评。

默认网格每窗输入6,093,536 bytes、输出89,280 bytes；每次更新约0.50133 s，额外DDR/DMA流量约12.33 MB/s，尚不含CPU查表和编码开销。减少DMA启动次数不保证比NumPy更快。该核只做插值累加，不是完整FFT/GCC/SRP流水线；`scripts/synthesize_srp.tcl`用于器件支持补齐后的资源评估，不声称已得到100MHz时序结果。WAV回放只用NumPy；事件中记录实际srp_backend。

上述Q2.22仅用于PS浮点GCC到此核的边界，与PC全定点参考的Q9.15不同；本后端不接收该参考的原始编码。2026-10-17接入后完整回归185项通过，覆盖Python编码到Icarus批量RTL输出的逐字对拍、误差界和模拟DMA运行入口。实际Vivado入口重测仍停在缺少板卡定义，未执行新增接线的目标IP验证。

## 本机验证与打包

仓库根目录使用现有环境：

```powershell
.\prototype\.venv\Scripts\python.exe -B -m pytest -c prototype/pyproject.toml prototype/tests deployment/pynq_z2/tests -q -p no:cacheprovider
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/run.py --model prototype/outputs/horn_detector_v1/fpga_model --check
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/run.py --model prototype/outputs/horn_detector_v1/fpga_model --replay PATH_TO_ARRAY14.wav --output NEW_REPLAY_OUTPUT
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/scripts/package.py --model prototype/outputs/horn_detector_v1/fpga_model --output deployment/pynq_z2/outputs/standalone
.\prototype\.venv\Scripts\python.exe -B deployment/pynq_z2/scripts/package.py --verify deployment/pynq_z2/outputs/standalone
```

模型参数路径在命令行按当前工作目录解释；配置内相对路径按部署根目录解释。模型来自本地旧实验，缺失时明确失败，不下载或使用未训练权重替代。源码包不含训练二进制、图像模型或构建产物；打包器将指定模型加入被忽略的本地目录/zip，并收集vendor/fpga所需源码，不需要原仓库运行。

可通过 `--bitstream` 和 `--handoff` 同时加入同次构建的bit/hwh；不提供则bundle.json明确标注firmware_included=false。打包不等于实际板上验收，始终记录board_validated=false。视觉/USB采集需要另行批准/提供OpenCV，视觉还需要指定SHA256的兼容模型，本轮未安装依赖或下载模型。输入格式仅支持明确的decoded YOLO，不支持任意检测模型输出。

时间模型的固件文件在包内统一命名为 `firmware/horn_capture.bit/hwh`，其他既有模型包沿用 `horn_cnn.bit/hwh`；打包器同步修正配置中的overlay路径。可选systemd模板接受两种bit文件名，实际入口仍检查配置指定的同名bit/hwh。模板未在本机安装或启动。

`run.py --check`只检查配置及模型；实时入口还要求匹配的bit/hwh、板端PYNQ和NumPy。程序仅在PS运行Python，不要求MicroBlaze/RISC-V。默认vision.enabled=false：可记录音频事件，不会伪造照片或车辆检测。main默认camera.sources为wide-0的/dev/video0、1920×1080 MJPG30；建议按实际USB设备改为稳定的/dev/v4l/by-id路径，确认不是板载麦克风接口。HTTP仍可通过`{"id":"wide-0","backend":"http","uri":"http://CAMERA/snapshot.jpg"}`配置；历史WAV回放不能混用当前实况图像。

时间戳和标定未验证时，照片可标出候选框，归属保持不确定。默认camera_cache为8帧、间隔0.25 s、超时2 s；USB按间隔保留历史，HTTP按间隔请求，约保留1.75 s历史，需覆盖窗口中点及处理延迟。配置省略该节时沿用历史4帧/0.5 s，缓存不足时不重复标记为已抓拍而允许后续重试。CPU负载大时可能跳分析窗；程序不会把跳窗当成连续确认。健康记录包含窗口数、丢包、跳窗及清理异常，需用于真实负载验收。

USB以spawn独立进程读帧、有界队列和配置长度的JPEG缓存传回PS；停止超时会终止该采集子进程，无法清理则报告失败。检查协商尺寸/FOURCC/报告帧率，回退会报错；报告30 FPS不等于持续30 FPS。USB JPEG经OpenCV重新编码，主机read中点并非曝光时间；照片、车辆框和反投影统一使用该原始分辨率JPEG，未做先去畸变再混用原图坐标。可兼容的视觉模型在强畸变图像上的效果仍待评估。橙框仅为声学兼容的可能声源，其他视觉框为灰色。camera_health记录协商模式与采集异常。

## 相机工具

以下命令在部署目录执行；除check-ground外需要经批准安装的OpenCV和实际USB摄像头，不需要车辆模型或FPGA即可独立验收采集。工具不会安装依赖。

```text
python scripts/camera.py probe --output outputs/usb-probe
python scripts/camera.py collect --count 30 --period 1 --output outputs/checkerboards
python scripts/camera.py calibrate --images outputs/checkerboards --model fisheye --columns 9 --rows 6 --square 0.025 --output config/calibration.json
python scripts/camera.py check-ground --points ground-controls.json
```

棋盘规格为内部角点数和以米为单位的实际格边长，需按实物修改；模型需根据真实拟合结果选择，不由商品畸变百分数推断。工具要求至少15张非重复可识别图像，以每第五张作为留出图，训练视图至少覆盖4×3网格的10格，并记录训练/留出RMS。默认1.5 pixels只是可改的工具误差限，不是道路验收标准。maximum_angle限制相对光轴的可用锥角，默认80°为操作起点，必须通过边缘控制点验证后设置；低RMS不证明整个视野/外参可靠。

输出标定后，在camera.calibration填`"config/calibration.json"`（相对部署根目录解析）或内嵌完整JSON。参数包含format=1、model、matrix(3×3零斜切内参)、coefficients(brown为k1,k2,p1,p2,k3；fisheye为k1..k4)、width、height、maximum_angle；尺寸改变、裁切或重对焦必须重新确认标定。地面检查文件为`{"world":[[x,y,0],...],"pixels":[[u,v],...]}`，至少4组独立实测对应点；先设置安装height/yaw/pitch/roll/offset，再检查像素和米误差。不得把从同一模型反算的点当真实控制点。

## 同步参考实现

几何与整数核心分别由 `scripts/export_geometry.py`、`scripts/export_integer.py` 在源码仓库生成，文件内记录原始SHA256；测试要求生成结果完全相同。几何参考更新后需重生成并重新对拍。板端不执行这些导出脚本，也不导入PC原型。
