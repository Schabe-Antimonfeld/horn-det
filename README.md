# FPGA Horn Project

`legacy`分支冻结广角适配前的`1f11b16`版本；`main`推进单颗GC2093 USB广角摄像头方案。主线默认单摄像头配置，144°水平/77.6°垂直为厂商标称，不能覆盖360°。真实视觉需加载实测畸变标定；14路声学和可选SRP加速保持原实现。

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

已有[14路PDM采集及Tiny CNN整数推理RTL](fpga/README.md)，通过本地功能仿真；PANNs研究模型在PC运行，Tiny CNN内核不支持直接部署PANNs。PS多相机源码见下方独立部署目录；Vivado综合、时序和实际板上闭环尚待验收。

新增[PYNQ-Z2独立部署源码](deployment/pynq_z2/README.md)：PL采集/Tiny CNN与PS NumPy声学、事件、可选相机适配，含独立打包和校验；可选批量SRP插值累加已接入部署构建和Python DMA后端，默认保持NumPy，收益待板上测试。当前Vivado2019.2缺XC7Z020器件支持，未生成可用bitstream；真实相机/视觉模型、时钟和标定等剩余事项列在[合作者交接](deployment/pynq_z2/HANDOFF.md)。该路径不在板端运行PANNs，不将PC结果等同于板端效果。
