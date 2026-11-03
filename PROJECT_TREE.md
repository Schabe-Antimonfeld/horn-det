# 最新主线目录

```text
horn-det/
├── README.md / CHANGELOG.md / AGENTS.md
├── docs/                      原理与技术记录
├── references/                文献与来源
├── fpga/                      最新采集、Tiny CNN及构建源码
├── deployment/pynq_z2/         唯一算法实现与板端接口
│   ├── README.md / DATA.md / HANDOFF.md / BOARD_PERFORMANCE.md
│   ├── run.py                 PCM回放或板端服务
│   ├── reproduce.py           编译、自检、回放、评测
│   ├── pyproject.toml / uv.lock
│   ├── config/default.json    唯一默认配置
│   ├── model/                 唯一固定权重与manifest
│   ├── examples/              PCM样例、预期事件与来源
│   ├── evaluation/            四套冻结特征、真值与许可
│   ├── assets.json            固定资产SHA-256
│   ├── ps/                    最新Python算法和四个C核
│   ├── scripts/               编译、评测、打包及硬件联调工具
│   ├── rtl/                   可选SRP协议硬件
│   └── tests/                 当前实现及协议测试
└── prototype/README.md        归档指引
```

历史研究代码和旧配置在codex/archive-2026-11-03分支，不在main形成多个可选版本。本地旧实验产物、原始数据和环境被忽略且保留。算法ZIP只包含当前算法、模型、配置、复现资产、测试和环境说明，不含RTL或依赖旧outputs；硬件源码仍在主仓库。
