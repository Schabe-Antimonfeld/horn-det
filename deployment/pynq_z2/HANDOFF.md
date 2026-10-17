# 合作者上板交接与剩余工作

## 本机已知阻塞

2026-10-17使用D:/Xilinx/Vivado/2019.2/bin/vivado.bat实际启动2019.2。创建xc7z020clg400-1项目失败：Device 21-436 No parts matched，Coretcl 2-106 Specified part could not be found。安装中没有可用PYNQ-Z2板卡定义。日志在本地outputs/probe-project.log。没有目标综合/布局布线/时序/资源/bitstream结果；不能用Icarus通过替代。

合作者需通过已有Vivado安装器补齐Zynq-7000器件支持，并配置适配2019.2的官方PYNQ-Z2板卡文件。不使用其他板卡DDR预设代替。板卡资料见已有[FPGA说明](https://github.com/Schabe-Antimonfeld/horn-det/blob/main/fpga/README.md)；本交接包内则查vendor/fpga/scripts/build.tcl的板卡检查。

另已实际修复Vivado注入PYTHONHOME/PYTHONPATH指向Python2、导致外部Python3校验器启动失败的问题：build_cnn.tcl调用校验器时使用-E。修正后构建入口通过模型检查，停在缺少PYNQ-Z2板卡定义；日志为outputs/build-attempt-fixed.log，不是成功综合日志。

批量SRP接入后再次执行部署构建入口，模型检查通过，仍在板卡定义检查处停止；日志outputs/srp-build-attempt.log。因此connect_srp.tcl接线尚未由目标Vivado工程验证，必须在补齐器件及板卡后完成validate_bd_design、综合和布局布线。

## 构建前必须提供

1. PYNQ镜像、Python、NumPy和PYNQ版本。源码仅做Python3.6语法及当前Python3.12数值测试；建议NumPy>=1.17，尚未在ARM镜像实际执行。
2. 阵列电源、PDM电平转换、时钟缓冲、线缆和引脚映射。实际为14颗IM72D128V01、直径240 mm；通道0朝+x，其余沿+y递增，不符则调整通道/坐标。
3. PDM输入min/max外部延迟的实测或完整接口预算，设置PDM_INPUT_MIN_NS/PDM_INPUT_MAX_NS。未确认时只运行create/synth，不能假造时序值生成可用bitstream。
4. 相机数量、协议、真实内外参、畸变和曝光时间来源。当前仅有HTTP JPEG快照缓存适配；USB/RTSP驱动尚未实现。

## 部署包与命令

包中的run.py、ps/、config/、model/和vendor/fpga/不依赖PC prototype。model中是本地真实录音训练的TinyCNN整数文件，科研/非商业使用；并未达到道路验收，PANNs不在板端运行。

```text
python run.py --check
python run.py --replay synchronized-array14.wav --output replay-results
vivado -mode batch -source scripts/build.tcl -tclargs model NEW_BUILD BOARD_REPOSITORY synth
```

补齐器件、板卡和接口时序后使用新的构建目录运行bit模式，将同次构建的horn_cnn.bit和horn_cnn.hwh放入firmware/，同时确保构建模型标签与model/manifest.json一致，再运行 `python run.py --output live-results`。run.py默认使用FPGA CNN，缺文件、模型不匹配或DMA出错会失败，不以软件随机模型替代。

附带scripts/horn-det.service作为可选systemd模板，默认路径/home/xilinx/horn-det，使用--output-root为每次启动创建独立UTC时间/进程编号目录。先验证实际Python路径、板端权限、退出与DMA清理，再由合作者安装启用。本机未安装或启动该服务；模板用SIGINT请求清理，不强制SIGKILL。

## 必须依次完成的实物验收

| 步骤 | 验证与留存证据 | 不通过时处理 |
|---|---|---|
| 电气与时钟 | 阵列供电/逻辑电平、PDM CLK占空比及边沿、DATA交接波形 | 停止连接或采集，修正接口 |
| 采集 | 每路单独声源确认通道序，PCM符号、噪声、频响、计数连续性 | 保留原始包，不拼接缺失样本 |
| 标定 | 增益及分数延时；在audio.gains/delays中设置14个值 | 未标定结果只作研究候选 |
| CNN | 同一特征比较PS整数参考与FPGA logits/概率/标签 | 查模型快照、尺度、DMA与饱和 |
| 可选SRP | 使用部署构建入口生成含srp_dma的overlay；相同PCM分别测试search.backend=numpy/fpga，保存完整响应、峰值、窗口耗时及DMA流量 | 超过量化误差界则查协议/尺度；无加速收益则保留NumPy |
| 连续负载 | 长时间录音，同时运行FFT/SRP/CNN与相机，统计gap及skipped_windows | 增大更新间隔或优化，不能隐瞒丢包 |
| 时间 | PCM计数到单调时钟、前端/标定/波束群延迟、相机曝光时间 | 维持不确定标注，不按处理结束时间归属 |
| 几何与道路 | 真实方位/俯角、相机覆盖、遮挡与坡度、车辆接触点 | 校正模型与误差界 |
| 连续效果 | 独立车辆/地点数据，事件召回、FP/hour、错配/歧义及抓拍延迟 | 在验证集调整后重新冻结并盲测 |

PCM原始1.344 MB/s，32-bit DMA加包头约2.691 MB/s、93.75包/s。窗口48128采样约1.00267 s，更新24064采样约0.50133 s。Python采集线程与分析分离，队列有界，只保留最新待分析窗；跳过分析窗会重置连续确认。simple DMA重装仍可能丢音频，必须实测；必要时迁移SG DMA/原生驱动，当前没有无损保证。

## 视觉与归属尚未完成的部分

当前视觉默认关闭，未安装新依赖、未下载车辆检测权重。已有OpenCV DNN适配要求明确SHA256、输入归一化和decoded YOLO输出(N,5+C)，仅支持COCO索引2/3/5/7车辆类别。需先批准并确定适配ARMv7镜像的OpenCV版本和车辆模型，再进行模型算子、内存、单帧耗时和真实图像验证；不能默认任何YOLO/ONNX模型都兼容。当前模块不是已验证的车辆检测器。

HTTP图像缓存记录主机请求中点，并非硬件曝光时间；车框底部中心只是地面锚点启发式。当前服务即使几何只剩一辆，也将车辆归属保持ambiguous，直至实现并验证真正的曝光同步、完整相机标定、接触点与跟踪。照片可标注候选，但不得视作确定鸣笛归属。WAV回放禁止与当前实时相机图像混用。

还需要实际相机适配、跨帧跟踪、缓存时长/内存预算与有效地面锚点；默认配置中的n/h/俯角/FOV可以修改，真实可支持的相机数及频率必须测试，不能认为参数可改即任意规模实时。

## 加速与模型限制

独立基线默认将FFT/GCC/SRP、波束和Log-Mel放在PS NumPy，采集和Tiny CNN在PL。可选SRP插值累加已接入部署构建脚本和Python实时入口：srp_batch包装原核，独立srp_dma与采集/CNN共享HP0。Python配置search.backend=fpga可开启，numpy为默认；旧的仅采集/CNN overlay不支持该后端。完整FFT/GCC/波束PL加速仍未实现。

默认5580方向以最多128方向分44批，PS执行延时查表及signed24 Q2.22编码；协议与误差界见[部署说明](README.md)。每窗额外输入约6.09 MB、输出约0.089 MB，板上收益可能受查表、编码、DMA和DDR竞争限制。先在实物以相同音频/网格对比响应、峰值、处理耗时、CPU占用及采集gap/skipped_windows，再决定是否启用。已有Icarus随机报文/极值/反压/错误与复位对拍，以及模拟DMA运行入口测试；这些不证明AXI IP参数、目标资源、100MHz时序或真实DMA吞吐。出错后停止并重新加载overlay，不能继续沿用故障会话。

Tiny CNN跨来源效果弱于此前PC PANNs研究结果；按验证集重评波束域、短鸣与量化，必要时重新训练/蒸馏或修改网络及对应RTL。当前0.65阈值来自旧片段级整数实验、分差0.05及事件计数为工程起点，未在本板端流程校准；一个声学窗对多个CNN上下文取最大值不能直接换算连续误报率。
