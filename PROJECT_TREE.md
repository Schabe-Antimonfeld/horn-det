# 项目目录

```text
horn-det/
├── AGENTS.md
├── README.md
├── CHANGELOG.md
├── PROJECT_TREE.md
├── docs/
│   ├── README.md
│   ├── 01_项目总体方案.md
│   ├── 02_阵列几何与远场声学模型.md
│   ├── 03_STFT与频域前端.md
│   ├── 04_GCC_PHAT完整推导.md
│   ├── 05_SRP_PHAT完整推导.md
│   ├── 06_粗到细搜索与动态剪枝.md
│   ├── 07_鸣笛检测_LogMel_TinyCNN.md
│   ├── 08_HornAware_SRP工程化方案.md
│   ├── 09_FPGA模块划分与定点实现.md
│   ├── 10_相机控制接口.md
│   ├── 11_无法自采数据时的数据方案.md
│   ├── 12_实验设计与评价指标.md
│   └── 13_文献来源映射与复现边界.md
├── prototype/
│   ├── pyproject.toml
│   ├── uv.lock
│   ├── README.md
│   ├── main.py
│   ├── simulate_surround.py
│   ├── src/horn/
│   └── tests/
├── fpga/
│   ├── README.md
│   ├── cnn.py
│   ├── cnn_host.py
│   ├── rtl/
│   ├── sim/
│   ├── scripts/
│   └── constraints/
├── deployment/pynq_z2/
│   ├── README.md
│   ├── HANDOFF.md
│   ├── run.py
│   ├── config/
│   ├── ps/
│   ├── rtl/
│   ├── scripts/
│   └── tests/
└── references/
    ├── README.md
    ├── references.bib
    └── *.url
```

`prototype/src/horn/surround.py` 为多相机几何及声画关联原型，`prototype/tests/test_surround.py` 包含解析测试。`prototype/outputs/surround_v2/` 为当前被忽略的本地仿真产物目录，保存配置、覆盖扫描、逐场景结果及合成框标注SVG；v1保留旧结果。产物不包含真实照片或重新训练的模型。
