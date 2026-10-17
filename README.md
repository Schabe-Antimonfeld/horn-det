# FPGA Horn Project

- [CHANGELOG.md](CHANGELOG.md)：功能、修复、验证进展与待确定事项。
- `docs/`：分模块技术文档。
- `references/`：参考文献说明与原 PDF 公开下载快捷方式。
- `prototype/`：使用 uv 的 Python 算法原型、测试和执行说明，覆盖进度见 [原型说明](prototype/README.md)。

当前项目主线：

```text
Tiny CNN Horn Detection
+
GCC-PHAT / SRP-PHAT
+
Horn-aware frequency weighting
+
coarse-to-fine DOA
+
FPGA streaming implementation
+
Camera pan/capture
```

新增可调多相机研究原型：360°高位声学候选与 PANNs、环形相机＋下视相机、地面投影及车辆方向关联。系统参数包括相机数量、安装高度、向下俯角和视场角，见[第 10 章](docs/10_相机控制接口.md)及[运行说明](prototype/README.md)。该扩展不改变上述参考基线的默认参数和旧接口。

车辆框由仿真生成，尚未接入真实视觉检测、相机通信或实路验证；唯一几何匹配不能单独证明车辆鸣笛。

当前关联默认验证声源俯角及鸣笛候选的 SRP 支持，缺证据拒绝、宽峰保留歧义；只看水平角保留为对照。默认仿真结果在本地 `prototype/outputs/surround_v2/`，固定小样本对照摘要见[第 12 章](docs/12_实验设计与评价指标.md)。真实音频、已训练权重和生成报告被忽略，不随 Git 仓库发布。

已有[14路PDM采集及Tiny CNN整数推理RTL](fpga/README.md)，通过本地功能仿真；PANNs和多相机关联仍运行于PC原型，不能使用Tiny CNN内核直接部署。Vivado综合、时序和实际板上闭环尚待验收。
