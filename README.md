# 鸣笛检测与定位

main保存项目各部分的最新有效内容：14路阵列采集、最新时间频带MLP算法、单颗GC2093广角相机的声源候选点投影。算法只保留一个运行入口、一份配置和一套固定权重。

- [最新版算法与复现](deployment/pynq_z2/README.md)：源码、C核、固定模型、示例回放、672场景冻结特征评测、测试和uv锁文件。
- [FPGA采集与RTL](fpga/README.md)：保留最新采集、DMA、Tiny CNN和可选SRP协议源码；当前时间MLP在ARM上运行。
- [技术文档](docs/README.md)：原理、方案记录、评价边界和参考文献。
- [交接与未完成项](deployment/pynq_z2/HANDOFF.md)、[Changelog](CHANGELOG.md)。

整理前完整源码已归档到分支`codex/archive-2026-11-03`（a8cda7b），旧配置、模型方案、训练及实验入口由该分支保留；`legacy`继续保存更早的多相机版本。本地被忽略的数据、输出和环境仍保留，Git分支不包含这些未跟踪资产。主线固定模型、样例与评测缓存现在随源码交付。

在`deployment/pynq_z2/`执行：

```text
uv sync --frozen
uv run --frozen python reproduce.py --output outputs/reproduction
uv run --frozen pytest
uv run --frozen python scripts/package.py --output outputs/horn_algorithm.zip
```

需要Python3.12、uv及已有GCC；首次依赖同步需网络或缓存。Windows可给复现命令追加`--compiler D:/MinGW/bin/gcc.exe`。包内不含编译器、虚拟环境、开发机DLL、bit/hwh或视觉权重。

固定权重复现覆盖PCM样例至事件及冻结特征至评测指标，不包含重新训练或完整原数据重建。公开来源仍按科研/非商业范围使用，来源和许可见[数据说明](deployment/pynq_z2/DATA.md)。仿真与电脑测试不等于道路或ARM验收，500 ms目标仍待实机验证。照片只标声源候选点，不运行YOLO；可靠单窗≥0.90即确认鸣笛证据并留图，方向与抓拍门限同为0.90，释放仍需两次未达标更新。画外、遮挡及同方向多车不确定性仍保留，点不等于车辆身份确认。

网页监控台见[前端与后端说明](frontend/README.md)。在deployment/pynq_z2中运行`uv run --frozen python web.py`，访问127.0.0.1:8765；支持真实固定权重音频回放与PYNQ实机接入口，不使用素材图冒充设备画面。
