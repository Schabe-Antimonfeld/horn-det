# 11/3—11/4板端交接对应的流程优化

本轮以main的1a82e3e为代码基线，针对`ALGORITHM_HANDOFF(2).md`第19节开展优化。没有把本仓库同期前端/阈值修改当作板端测速。保留14路、48 kHz、48128/24064采样、完整5580方向/91对、65抽头、固定权重和当前单窗0.90；交接板端原记录仍是0.95/0.90、两窗，不能混报版本。

## 依据与基线

配套包`pynq_z2_handoff_20261104_2bf7452.zip`，来源提交`2bf7452b7795c7ccd15896a9bdabaf29ecf9ed77`，ZIP SHA256为`714313c9e0250f8b625b0c064e8b115ba7d8c8fd12b7ad53ecad49b52a249a6e`。316份清单文件逐项校验通过，原包仅解压到本地忽略目录，不将板端旧代码整体覆盖main，也不自动迁入固件/针序。

复核第19节原始events得到：

| 原始条件 | 连续可复用窗 | 连续窗中位/ms | 首窗/重建窗中位/ms |
|---|---:|---:|---:|
| 无相机、实际双候选，60窗 | 16 | 643.785 | 925.87，44窗 |
| 旧相机初始化联合长测，299窗 | 1 | 867.321 | 1197.63，298窗 |
| 最终相机入口短测，12窗 | 0 | 无样本 | 1007.347 |

各测试的环境音和候选数量不同，不是配对性能实验。数据支持超预算与跳窗、缓存重建共同出现；不证明某个单一函数造成全部变慢。相机原始JPEG路径已在交接包修复，不能将其当作本轮相对该包的新加速结果。

## 接入main的改动

1. USB MJPG协商支持原始返回时保留相机JPEG，检查8 MiB边界、SOF尺寸和8位精度；不用每帧解码再编码。驱动不支持时保留原像素路径，模式不符仍报错，不静默降分辨率。工作进程持续读帧，只按缓存period发布，减少进程间大图传输。保留main的ImageIO画点，不恢复YOLO。
2. 相机第一帧就绪后才启动DMA采集；模型和Pipeline也在采集之前初始化。启动等待单列于startup.json，初始化失败不算就绪。退出时继续排空子进程队列，避免等待feeder冲刷形成死锁。nearest支持decode=False，网页预览仍直接复用编码缓存。
3. DMA使用cacheable缓冲，但保持recvchannel.wait完成缓存失效后再解码；不启用交接中未通过的prefetch。完整PCM范围检查改用等价min/max归约。CapturePump使用逐包构建的独立重叠窗口，去掉发窗瞬间的全窗拼接，队列容量仍为1。
4. PS饱和标志仅聚合当前窗口覆盖的包，饱和包离开滑窗后恢复；不忽略真实饱和。输入断续仍重置状态。此项不能修复旧PL全会话锁存标志，必须沿用硬件队员已经修复、验收的PMODA固件。
5. 保留采集线程可选nice设置，新增`--capture-nice -10`由操作者在有权限的Linux镜像上显式使用，不自动提权。已初始化对象的GC freeze只在本实例拥有时恢复；不解除其他调用者的freeze。
6. 新增double精度DSP库，默认只启用有稳定收益的时间统计。波束合成、稀疏Mel和另一种Hann遍历仅作可选板端对照，默认关闭；不改精度、不加fast-math。波束C核可读真实数组步长，避免强制拷贝FFT结果。接近判定门限时，时间统计回到NumPy参考评分，避免微小累加舍入影响门限。

其他四个C库源码、PHAT分组累加、FFT、完整方向搜索、连续性全量校验及65抽头语义保持。没有用增大队列、少算方向、降低候选上限或缩短窗口来换成绩。

## 同一输入的电脑配对结果

输入为交接中的`microphones-record-60.wav`，SHA256为`0c024a33d28bd853f6d6c0c99a53764d00c3ba171b682fe2139522e61c6048f8`，14路/48 kHz/PCM16，2863616帧，59.658667秒。它是环境录音，没有鸣笛/方向真值；只能核对数值与性能，不能算道路召回、方向误差或SNR。源录音不随本算法包再次分发。

AMD64电脑、Python3.12、NumPy2.5.3、现有GCC。每轮顺序交替，5轮118窗，每种实现590个窗口；按实际候选数和连续性分组。比较选定实现与本轮集成后的NumPy DSP对照，不将不同日期的非配对计时计算为收益：

| 分组 | 每实现样本 | NumPy中位/ms | 选定实现中位/ms | NumPy P90/ms | 选定P90/ms |
|---|---:|---:|---:|---:|---:|
| 双候选连续窗 | 530 | 10.128 | 9.877 | 11.655 | 11.401 |
| 单候选连续窗 | 55 | 8.680 | 8.434 | 9.776 | 9.447 |
| 首窗，实际双候选 | 5 | 17.529 | 16.552 | 18.144 | 18.128 |

双候选连续窗分类阶段中位0.420→0.167 ms；整窗中位仅改善约2.5%。观察到的双候选连续窗最大值16.498→20.437 ms，尾延迟没有保证改善。不能将这些比例乘到ARM的644/1007 ms上来宣称达标；相机原始路径收益来自交接中的板端测量，不在本电脑声学配对时间内。

试验中，C波束核整理数组的复制曾造成回退，已改成直接读步长；C波束/Mel在整窗中未取得稳定净收益，新的Hann遍历在电脑上也未胜出，因此未默认启用。C时间统计微基准与整窗均有收益，才进入默认路径。选型不是“C一定比NumPy快”。

本地证据位于`outputs/fullflow_v1_20261104`：source-verification.json、main-baseline.zip、paired-selected-real.json、paired-skips.json及原交接解压快照。脚本支持按输入序号刻意跳窗，核对skipped_window重建；这不是对真实DMA压力的替代。

## 当前软件验证

227项仓库测试通过，涵盖PCM/窗口所有权、跨包饱和恢复、间断重置、C数值/数组步长、近门限回退、原始JPEG及相机发布。固定权重的672场景复现通过，鸣笛样例4窗均CONFIRMED，静音2窗均IDLE；Chrome桌面/手机14项联调通过。真实环境录音的590对连续/首窗结果与受控跳窗54对结果均保持决策和方向一致，选定路径最大分数差7.50e-16以内。这些是软件回归证据，不构成新道路准确率或板端性能验收。

## 自检、构建与回退

```text
python scripts/build.py
python run.py --check
python reproduce.py --output outputs/NEW_REPRODUCTION
python scripts/benchmark_fullflow.py --input /path/microphones-record-60.wav --output outputs/NEW_PAIRED.json --paired --limit 12 --repeats 3
```

五库为libsrp.so（ABI2）、libstream.so/libconditioning.so/librefinement.so（ABI1）、libdsp_v3.so（ABI3）；Windows对应DLL只用于电脑验证，不复制到ARM。源码或二进制哈希不匹配时用`build.py --directory NEW_LIBRARY_DIRECTORY`构建；自检必须加载同一份源码的库。配置中的绝对native_library路径须按目标平台设置。没有新增依赖；不使用fast-math。

`acceleration.enabled=false`关闭新增DSP，恢复NumPy时间统计；默认beam/mel/prepare均为false。板端如需逐项测量，可在上述基准命令分别加`--native-beam`、`--native-mel`或`--native-prepare`，按1/2候选、连续/重建分别判断。一次只加一项，不把未测的组合直接上线。基准逐轮流式读WAV，不把整段float64多路快照保留在内存中；双Pipeline配对仍增加内存，板上先限12窗并检查可用内存，再延长。

`--skip-every 4 --limit 24`模拟输入窗口缺失，`--reset-every N`模拟epoch变化。它们只影响实验调度，不是生产剪枝。输入/模型哈希、实际配置、库哈希、候选数、重用标记、每窗CPU/墙钟时间和原始分数均保留。此脚本的CPU时间只在支持thread_time的解释器上记录，否则为null，不冒用进程CPU时间替代。

## 板端最后验收

本包不包含或覆盖PMODA bit/hwh、接线或开机固件。以main默认配置为模板，将硬件队员当前已验收的成对overlay路径、实测校准及相机设置填入新的board.json；不能直接拿旧band_refined/YOLO配置覆盖。沿用交接包固件，旧main RTL构建结果不代表包含第19节修复。本轮没有重新综合PL或证明物理针序/几何标定。

先相同WAV配对，再真实采集、再叠加相机/网页预览和事件存图。每种条件先12窗，然后300秒：

```text
python run.py --config /absolute/path/board.json --output outputs/NEW_LIVE --limit 12 --capture-nice -10
python web.py --config /absolute/path/board.json --host 0.0.0.0
```

nice=-10需Linux已有权限，不可用时先不传该参数并在报告中说明。硬件实际操作由持板队员执行。相机开启仍要求实测标定，原始JPEG及诊断图不构成车辆归属证据。

逐窗分析首窗/连续/重建、1/2候选、中位/P90/最大、超过500ms及501.333ms的数量，连同gap、dropped、skipped_windows一起报告。仅status=completed不算验收通过，较低中位也不代替零跳窗要求。

- processing_seconds：Pipeline.analyze墙钟时间。
- analysis_cpu_seconds：同一次analyze的线程CPU时间，不支持时null。
- queue_wait_seconds：窗口发出到开始分析；availability_to_complete_seconds：窗口发出到分析结束。回放没有真实采集就绪时刻，记录null。
- post_analysis_seconds：当窗相机保存、预览等后处理，另列，不包含JSON写入。
- health.json.last_io：最近一窗JSON编码及events写入时间；startup.json单列相机就绪与启动时间。这些不是端到端延迟。
- Conditioning位于采集工作线程，可与分析重叠，不能直接加进processing_seconds。

500ms仍未取得本轮ARM验收。双候选即使保留连续缓存，原板端644ms也明显超预算；本轮电脑约2.5%的声学收益不能说明已经解决该缺口。若板端新分项仍不足，下一步需评估逐包分摊或PL FFT/波束等结构调整，单独验证数值与并发资源预算。
