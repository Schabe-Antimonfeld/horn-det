# PYNQ-Z2 独立部署实现

main当前适配单颗GC2093 USB广角摄像头；legacy分支冻结1f11b16。默认相机为wide-0、1920×1080、厂商标称HFOV144°/VFOV77.6°；布局single不附带下视相机，不提供360°视觉覆盖。未标定时仅允许音频运行/名义几何演示；启用视觉必须提供实测calibration，支持brown或fisheye模型，尺寸不符会拒绝运行。相机中心为(offset_x,offset_y,height+offset_z)，height仍是阵列离地高度，yaw/pitch/roll为待实测安装角。

目标：Vivado 2019.2、XC7Z020、14 路/48 kHz/直径0.24 m阵列，Python运行在ARM PS。该目录不依赖PC原型的Python3.12或PyTorch；板端源码保持Python3.6语法兼容，实际镜像版本待合作者确认。

部署路径：PL采集及Tiny CNN，可选批量SRP插值累加；PS连续接收、FFT/PHAT/GCC、方向延时查表、候选波束、事件、相机与空间关联。SRP默认仍由NumPy累加，板上测速后再决定是否开启PL后端。PANNs不属于本板端路径；不能把PC耗时当作ARM实时性能。

默认窗口94包=48128采样，更新47包=24064采样；沿用1024/512对称Hann，仅使用完整帧。训练CNN上下文40×24，与一秒声学定位窗口不同。所有分数门限和可见车辆归属仍需实际场景校准。

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

`run.py --check`只检查配置及模型；实时入口还要求匹配的bit/hwh、板端PYNQ和NumPy。程序仅在PS运行Python，不要求MicroBlaze/RISC-V。默认vision.enabled=false：可记录音频事件，不会伪造照片或车辆检测。main默认camera.sources为wide-0的/dev/video0、1920×1080 MJPG30；建议按实际USB设备改为稳定的/dev/v4l/by-id路径，确认不是板载麦克风接口。HTTP仍可通过`{"id":"wide-0","backend":"http","uri":"http://CAMERA/snapshot.jpg"}`配置；历史WAV回放不能混用当前实况图像。

时间戳和标定未验证时，照片可标出候选框，归属保持不确定。HTTP缓存目前只保留最近4帧/相机，请求周期0.5 s，缓存不足时不重复标记为已抓拍而允许后续重试。CPU负载大时可能跳分析窗；程序不会把跳窗当成连续确认。健康记录包含窗口数、丢包、跳窗及清理异常，需用于真实负载验收。

USB以spawn独立进程读帧、有界队列和最近4帧JPEG缓存传回PS；停止超时会终止该采集子进程，无法清理则报告失败。检查协商尺寸/FOURCC/报告帧率，回退会报错；报告30 FPS不等于持续30 FPS。USB JPEG经OpenCV重新编码，主机read中点并非曝光时间；照片、车辆框和反投影统一使用该原始分辨率JPEG，未做先去畸变再混用原图坐标。可兼容的视觉模型在强畸变图像上的效果仍待评估。橙框仅为声学兼容的可能声源，其他视觉框为灰色。camera_health记录协商模式与采集异常。

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
