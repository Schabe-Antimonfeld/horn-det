# PYNQ-Z2 独立部署实现

目标：Vivado 2019.2、XC7Z020、14 路/48 kHz/直径0.24 m阵列，Python运行在ARM PS。该目录不依赖PC原型的Python3.12或PyTorch；板端源码保持Python3.6语法兼容，实际镜像版本待合作者确认。

部署路径：PL采集及Tiny CNN；PS连续接收、NumPy声学计算、候选波束、事件、相机与空间关联。PANNs不属于本板端路径。先交付完整可回放基线，再以实测预算决定额外PL加速；不能把PC耗时当作ARM实时性能。

默认窗口94包=48128采样，更新47包=24064采样；沿用1024/512对称Hann，仅使用完整帧。训练CNN上下文40×24，与一秒声学定位窗口不同。所有分数门限和可见车辆归属仍需实际场景校准。

## 分阶段记录

1. 配置及部署合同：已实现；Vivado2019.2启动成功，但缺少XC7Z020器件支持和PYNQ-Z2板卡定义，尚不能创建目标工程。
2. 连续采集、断流处理与DMA生命周期：已实现，148项完整测试通过；真实DMA吞吐待板上测量。
3. 无SciPy/PyTorch的板端声学与几何基线：已实现，152项完整测试通过，DC/标定、SRP、波束、特征与PC参考对拍一致；ARM性能未测量。
4. Vivado构建验证：目标器件支持缺失，不能综合。已有采集/CNN源码通过scripts/build.tcl引用并由打包器完整收集；新增可选SRP插值累加核通过Icarus功能测试，尚未集成至overlay，不在默认运行路径。
5. 训练模型整数推理与特征对拍。
6. 相机/视觉适配、事件与独立运行服务。
7. 完整打包、回归与合作者交接。

各阶段通过针对测试、原测试回归和一致性检查后提交；未通过的硬件验收不标成完成。输出、权重及构建产物在本目录被忽略。

## 构建

`scripts/build.tcl <model_directory> <new_build_directory> <board_repository_or_dash> create|synth|bit`复用既有采集/CNN工程，输出horn_cnn.bit/hwh。bit模式仍要求经验证的PDM_INPUT_MIN_NS/MAX_NS，不能编造数值绕过接口检查。构建主机需NumPy及完整模型目录；板端不执行Vivado。

可选 `rtl/srp_accumulator.v` 每对依次接收signed24-bit GCC下界、上界的32-bit符号扩展值和低16-bit Q0.16插值系数。默认91对，末系数TLAST；输出4字为标识0x31505253、signed48结果低/高字、错误位。结果保留额外16位小数，包错返回零并置错。该核只做插值累加，不是完整FFT/GCC/SRP流水线；`scripts/synthesize_srp.tcl`用于器件支持补齐后的资源评估，不声称已得到100MHz时序结果。
