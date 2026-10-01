# FPGA 禁鸣区定向鸣笛检测项目文档

本目录用于比赛项目后续开发参考。当前版本系统闭环限定为：

```text
车辆鸣笛
   ↓
8 路麦克风同步采集
   ↓
STFT / FFT
   ├──────────────┐
   │              │
   ▼              ▼
鸣笛识别       Horn-aware SRP
Tiny CNN       声源定向
   │              │
   └──────┬───────┘
          ▼
 Horn + DOA + Confidence
          ↓
      相机转向
          ↓
       定向抓拍
```

## 推荐主算法

1. **鸣笛检测参考基线**：Log-Mel + Tiny CNN；第 07 章的 FFT 规则、MFCC 分类器、迁移学习、蒸馏、CRNN 和 PCEN 均为待确定候选，尚未替换主线。
2. **基础定位**：GCC-PHAT + SRP-PHAT。
3. **比赛改进定位**：目标频带增强、谱减与粗到细搜索组成的 Horn-aware SRP。
4. **实时优化**：根据阵列几何选择可靠麦克风对，减少无效 GCC / SRP 运算。
5. **相机接口**：输出 event_state、horn_valid、doa_valid、horn_probability、azimuth、doa_confidence、timestamp、event_id；编码与有效性规则以第 10 章为准。

流程图展示当前参考基线，不表示替代检测模型或软硬件部署位置已经冻结。相机提前转向和正式抓拍均需检查定位有效性及对应控制门限。

## 文档索引

- [01_项目总体方案.md](01_项目总体方案.md)
- [02_阵列几何与远场声学模型.md](02_阵列几何与远场声学模型.md)
- [03_STFT与频域前端.md](03_STFT与频域前端.md)
- [04_GCC_PHAT完整推导.md](04_GCC_PHAT完整推导.md)
- [05_SRP_PHAT完整推导.md](05_SRP_PHAT完整推导.md)
- [06_粗到细搜索与动态剪枝.md](06_粗到细搜索与动态剪枝.md)
- [07_鸣笛检测_LogMel_TinyCNN.md](07_鸣笛检测_LogMel_TinyCNN.md)
- [08_HornAware_SRP工程化方案.md](08_HornAware_SRP工程化方案.md)
- [09_FPGA模块划分与定点实现.md](09_FPGA模块划分与定点实现.md)
- [10_相机控制接口.md](10_相机控制接口.md)
- [11_无法自采数据时的数据方案.md](11_无法自采数据时的数据方案.md)
- [12_实验设计与评价指标.md](12_实验设计与评价指标.md)
- [13_文献来源映射与复现边界.md](13_文献来源映射与复现边界.md)

## 主要参考来源

主要参考来源包括：

- Grinstein et al., *Steered Response Power for Sound Source Localization: a tutorial review*, 2024。
- Furletov, *Sound Processing for Autonomous Driving*, TU Darmstadt 博士论文, 2022。
- Hu et al., *Robust Real-Time DOA Estimation for Outdoor Vehicle Acoustic Sources Using Dynamic-Pruning GCC-PHAT and Adaptive Forgetting Factor OPAST-MUSIC*, Sensors, 2026。
- Dim et al., *HornBase: An audio dataset of car horns in different scenarios and positions*, Data in Brief, 2024。

Wang et al., *Tailored for vehicle horn: A novel sound source capture method* 与本项目高度相关，但当前未获得可公开下载的完整 PDF，因此只将公开摘要能够确认的思想作为补充参考，不将本文档中的 Horn-aware SRP 公式声称为其原式复现。

鸣笛检测候选方案的新增来源 R6~R11、各来源的核对程度及复现边界见 [第 13 章](13_文献来源映射与复现边界.md)，完整题录维护于 [references.bib](../references/references.bib)。
