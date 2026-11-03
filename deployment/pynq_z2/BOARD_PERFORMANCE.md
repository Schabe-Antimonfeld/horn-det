# 当前版本的性能与500 ms验收

目标为48128采样完成窗的声学处理≤500 ms，更新间隔24064/48000≈501.3 ms。声学时间包含SRP、候选波束、特征、模型及事件判断；约1.003秒采集窗口、相机取图、车辆检测、日志和冷启动另记，不能把它说成鸣笛到照片的总延迟。

main现只保留最后选定实现：14路、完整5580方向/91对、fold4/single PHAT、65抽头波束、buffered Conditioning、Hann融合、逐方向输出与重叠帧复用、BandTemporalMLP-30866和0.95/0.90双门限。没有借整理改门限、网格、对数、精度或模型。历史方案消融入口和原始记录在归档分支及开发机旧outputs中；最新运行入口与代码结构见[README.md](README.md)。

## 已有证据

整理前446项回归、1008构造场景和84连续混合窗口对拍通过；同机正例/方向阶跃流整窗中位17.56→14.47 ms、36.32→32.02 ms，重置及最大值未保证改善。用户板端截图中位751→529 ms未提供逐窗/分项JSON，不能把它和电脑比例相乘来宣称达标。

本轮整理先冻结a8cda7b的模型、PCM回放预期和四套评测特征，再通过当前统一模块验证固定权重推理。672个场景的主360°/可见联合F1保持0.6667/0.6308，压力组0.6512/0.6032；仅缓存上游特征不构成全部原始PCM流水线重测。正例四窗仍为CANDIDATE、CONFIRMED、CONFIRMED、CONFIRMED，静音两窗为IDLE。源码数值与回退测试另外检查边界；精度与速度必须区分。

## 如何测当前版本

`reproduce.py --output NEW_DIRECTORY`输出当前机器六窗的时间及超时数，主要用于安装和正确性核对，样本太短且含重复录音，不作吞吐/最坏时延验收。对实际至少60秒同步14路PCM，运行`python run.py --replay LONG_ARRAY.wav --output NEW_REPLAY`，从events.jsonl分别统计首窗、稳态、重置的中位/P90/最大和逐窗超时比例。实况再检查gap、skipped_windows、相机负载、CPU频率/温度与RSS。

- `srp_diagnostics`含preparation_seconds、fft_seconds、phat_seconds、ifft_seconds、search_seconds及新/复用帧数；preparation_seconds包含在fft_seconds中，不重复相加。
- `beamforming_diagnostics`记录共享FFT和合成耗时、逐候选output_reuse_flags及output_reused_beams。一个候选变化不影响另一个精确方向的缓存资格；两个都变化则没有输出复用收益。
- `stream_diagnostics`校验epoch、计数、窗口大小及全部重叠PCM；缺包、跳窗、倒序或内容变化时重算。窗首门控和分数延迟边界重算，不能略过这些校验提速。
- Conditioning在采集线程并行处理，不能直接把逐包收益从processing_seconds中相减。保留原PCM归一化、DC、增益、校准FIR与整数延迟语义。

当前无新的ARM500 ms、视觉全流程或PL时序证明。已知旧WindowBuffer削顶标记可能保守地持续到历史重置，未在本轮放宽；发生时保存包与epoch另行排查。性能报告中的board_validated=false必须保留。
