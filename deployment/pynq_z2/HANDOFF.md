# 最新主线的合作者交接

当前唯一配置为`config/default.json`，固定模型在`model/`；从旧版本迁移时保留实测麦克风gains/delays、相机标定、视觉权重、USB及overlay路径。算法顺序、门限及复现命令见[README.md](README.md)。历史源码归档在`codex/archive-2026-11-03`的a8cda7b，不把本地旧outputs当运行依赖。

## 电脑复现

Python3.12＋uv＋已有GCC，执行`uv sync --frozen`和`uv run --frozen python reproduce.py --output outputs/reproduction`。四库从C源码构建，校验资产，重放鸣笛和静音，并用固定权重重算四组共672个冻结特征场景的指标。`uv run --frozen pytest`验证当前数值、连续性、事件、相机及打包逻辑。算法ZIP包含源码、锁文件、模型、样例、许可、缓存及测试；不含虚拟环境、DLL/SO、PL或视觉权重。初次同步仍需依赖下载或缓存。

独立交付检查已通过：全新uv环境、四库默认目录源码编译、隔离模式PCM回放及672场景评测、包内130项算法测试。主仓库147项测试含保留的硬件协议检查；只验证电脑软件/仿真行为。

## PYNQ-Z2

1. 当前固定模型运行在Zynq ARM PS。保留同接口、已经合作者验收的14路采集bit/hwh；若需重新构建采集，在源码仓库运行`vivado -mode batch -source scripts/build_ps.tcl -tclargs NEW_BUILD BOARD_REPOSITORY bit`，仅含算法的压缩包没有RTL工程。
2. 在实际PYNQ镜像安装目录使用已有Python/NumPy及GCC运行`python scripts/build.py`，生成libsrp.so（ABI2）、libstream.so、libconditioning.so、librefinement.so（后三者ABI1）。不复制Windows DLL，不直接把电脑Python3.12锁定环境同步到旧镜像。
3. 使用校准后的配置执行`python run.py --check`，再`python run.py --replay synchronized-array14.wav --output NEW_REPLAY`，最后`python run.py --output NEW_LIVE`。默认采集overlay路径为firmware/horn_capture.bit；不要用Tiny CNN核运行时间MLP。
4. 检查每窗processing_seconds和phase_seconds、首窗/稳态/重置、500 ms超时数量/比例、gap、skipped_windows与采集线程Conditioning耗时。长录音至少60秒，包含干扰、方向变化、静音、丢包和跳窗；时间报告必须注明机器及采集/视觉是否开启。
5. 默认视觉关闭，示例镜头参数未标定；GC2093 USB、镜头内外参、曝光时间与车辆模型需按[相机说明](CAMERA.md)验收。没有这些证据时，保持歧义和画外来源，不把唯一几何匹配称为车辆鸣笛证明。

## 仍待实机完成

用户此前截图稳态中位751→529 ms属于整理前版本的记录；本轮没有板卡，不宣称达到500 ms。电脑源码测试、C自检或Icarus不能代替ARM速度和PL时序验收。当前Vivado2019.2在本机缺XC7Z020器件/板卡支持的历史阻塞仍未解除；没有重新生成bit/hwh。SRP索引压缩尚未接入，等待真实分项再决定。道路连续事件召回、误报/小时、校准、时钟、视觉负载仍需合作者采集证据。

旧代码及详细实验过程可从归档分支取得；本地原始数据、训练权重和旧环境没有删除，也不会因创建Git分支自动上传。当前固定模型、复现样例和评测缓存则作为最新版受哈希保护的交付资产保存。
