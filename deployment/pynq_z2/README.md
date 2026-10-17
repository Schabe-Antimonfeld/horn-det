# PYNQ-Z2 独立部署实现

目标：Vivado 2019.2、XC7Z020、14 路/48 kHz/直径0.24 m阵列，Python运行在ARM PS。该目录不依赖PC原型的Python3.12或PyTorch；板端源码保持Python3.6语法兼容，实际镜像版本待合作者确认。

部署路径：PL采集及Tiny CNN；PS连续接收、NumPy声学计算、候选波束、事件、相机与空间关联。PANNs不属于本板端路径。先交付完整可回放基线，再以实测预算决定额外PL加速；不能把PC耗时当作ARM实时性能。

默认窗口94包=48128采样，更新47包=24064采样；沿用1024/512对称Hann，仅使用完整帧。训练CNN上下文40×24，与一秒声学定位窗口不同。所有分数门限和可见车辆归属仍需实际场景校准。

## 分阶段记录

1. 配置及部署合同：已实现；Vivado2019.2启动成功，但缺少XC7Z020器件支持和PYNQ-Z2板卡定义，尚不能创建目标工程。
2. 连续采集、断流处理与DMA生命周期：已实现，148项完整测试通过；真实DMA吞吐待板上测量。
3. 无SciPy/PyTorch的板端声学与几何基线：已实现，152项完整测试通过，DC/标定、SRP、波束、特征与PC参考对拍一致；ARM性能未测量。
4. Vivado构建验证：目标器件支持缺失，不能综合。已有采集/CNN源码通过scripts/build.tcl引用并由打包器完整收集；新增可选SRP插值累加核通过Icarus功能测试，尚未集成至overlay，不在默认运行路径。
5. 训练模型整数推理与特征对拍：已实现，155项完整测试及本地真实训练模型RTL对拍通过。整数软件后端供回放验证；板端默认FPGA后端，不加载PyTorch。
6. 相机/视觉适配、事件与独立运行服务：已实现离线回放、PYNQ启动入口、HTTP快照缓存和可选OpenCV适配；真实视觉模型/曝光/标定未验证，服务不输出确定车辆归属。无相机帧可重试，成功抓拍按事件去重。
7. 完整打包、回归与合作者交接：脚本支持独立源码/运行包及逐文件SHA256校验；最终验证记录见本地outputs/validation.json，待验收项见[HANDOFF.md](HANDOFF.md)。

各阶段通过针对测试、原测试回归和一致性检查后提交；未通过的硬件验收不标成完成。输出、权重及构建产物在本目录被忽略。

## 构建

`scripts/build.tcl <model_directory> <new_build_directory> <board_repository_or_dash> create|synth|bit`复用既有采集/CNN工程，输出horn_cnn.bit/hwh。bit模式仍要求经验证的PDM_INPUT_MIN_NS/MAX_NS，不能编造数值绕过接口检查。构建主机需NumPy及完整模型目录；板端不执行Vivado。

可选 `rtl/srp_accumulator.v` 每对依次接收signed24-bit GCC下界、上界的32-bit符号扩展值和低16-bit Q0.16插值系数。默认91对，末系数TLAST；输出4字为标识0x31505253、signed48结果低/高字、错误位。结果保留额外16位小数，包错返回零并置错。该核只做插值累加，不是完整FFT/GCC/SRP流水线；`scripts/synthesize_srp.tcl`用于器件支持补齐后的资源评估，不声称已得到100MHz时序结果。

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

可通过 `--bitstream` 和 `--handoff` 同时加入同次构建的bit/hwh；不提供则bundle.json明确标注firmware_included=false。打包不等于实际板上验收，始终记录board_validated=false。HTTP视觉需要另行批准/提供OpenCV及指定SHA256的兼容模型，本轮未安装依赖或下载模型。输入格式仅支持明确的decoded YOLO，不支持任意检测模型输出。

`run.py --check`只检查配置及模型；实时入口还要求匹配的bit/hwh、板端PYNQ和NumPy。程序仅在PS运行Python，不要求MicroBlaze/RISC-V。默认vision.enabled=false：可记录音频事件，不会伪造照片或车辆检测。开启视觉前配置camera.sources，格式为 `{"id":"ring-0","uri":"http://CAMERA/snapshot.jpg"}`；历史WAV回放不能混用当前实况图像。

时间戳和标定未验证时，照片可标出候选框，归属保持不确定。HTTP缓存目前只保留最近4帧/相机，请求周期0.5 s，缓存不足时不重复标记为已抓拍而允许后续重试。CPU负载大时可能跳分析窗；程序不会把跳窗当成连续确认。健康记录包含窗口数、丢包、跳窗及清理异常，需用于真实负载验收。

## 同步参考实现

几何与整数核心分别由 `scripts/export_geometry.py`、`scripts/export_integer.py` 在源码仓库生成，文件内记录原始SHA256；测试要求生成结果完全相同。几何参考更新后需重生成并重新对拍。板端不执行这些导出脚本，也不导入PC原型。
