# 固定资产、来源与复现边界

## 资产合同

`assets.json`记录整理前提交a8cda7b、模型标签及每个固定文件的SHA-256。复现开始前检查全部资产；权重、示例、缓存或标签不符即失败，不静默下载替代文件。模型为BandTemporalMLP-30866，权重SHA-256为`1826a9ba4109de8dcdb6ad7d78507f1ab0de9b9861ae0d6b32b9fcab563159b7`，沿用已选combined、seed17、epoch14权重，不重新训练或选择门限。

- `model/`：一套manifest及NumPy权重，manifest中历史checkpoint仅作来源说明，不需要对应训练目录。
- `examples/horn-array14.wav`：14路PCM16/48 kHz，由验证协议original第98场景的未Conditioning PCM重复三次构造；保留原来源和构造元数据于`provenance.json`。目标ESC-50文件4-175855-A-43.wav，干扰4-111671-A-42.wav，经过阵列传播、混合、PCM量化和重复，不是原始道路阵列录音。
- `examples/silence-array14.wav`：生成的72192采样全零同步PCM；只验证无证据时保持IDLE。
- `examples/expected.json`：整理前算法的六个窗口事件。计时和实现诊断不冻结，方向、分数、候选、有效性与事件字段必须一致。
- `evaluation/`：四套协议各168个场景，84正例和84负例；NPZ含每场景两个候选的640维double时间频带特征和仅供核对的旧分数。JSON含真值角度、标签、候选、来源文件及混合PCM哈希。评测实际执行当前模型矩阵运算并重算混淆计数；不会把旧分数直接作为新结果。

评测缓存的上游是整理前最新版fold4/single、全网格/91对SRP及65抽头波束，先用PCM哈希核对构造输入，再比对每个模型分数。该缓存不证明未来修改的STFT、SRP或波束仍保持全部672个场景的行为；本轮整理保留的数值测试和示例PCM回放负责这些模块的实现检查。要重新生成全部声场或训练，使用归档分支及依法取得的原数据。

## 来源与许可

交付包限科研/非商业用途，保留来源署名和改编说明，不视为数据或模型作者背书。

- HornBase：Cleyton Dim等，HornBase - A Car Horns Dataset，Mendeley Data v2，DOI [10.17632/y5stjsnp8s.2](https://doi.org/10.17632/y5stjsnp8s.2)。当地保存的原始元数据及许可字段见`evaluation/HornBase-source.json`，数据集v2标注CC BY 4.0；论文许可与数据集许可分别记录，不混用。缓存由单通道录音经裁剪、阵列传播、混合及特征变换得到，不附带完整原始数据集。
- ESC-50：Karol J. Piczak，ESC: Dataset for Environmental Sound Classification，2015。源仓库固定commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`，整体许可CC BY-NC 3.0；全文和各原录音作者/链接保留在`evaluation/ESC50-LICENSE.txt`。样例及特征为改编产物，使用该非商业范围。
- 全部涉及的源文件名、数据集、原始哈希、分组和划分在`evaluation/sources.json`；不保留开发者机器的原始绝对文件路径。
- 模型训练曾使用PANNs教师；Kong等，PANNs，IEEE/ACM TASLP 2020，DOI [10.1109/TASLP.2020.3030497](https://doi.org/10.1109/TASLP.2020.3030497)。教师原预训练记录Zenodo3987831为CC BY 4.0；本包不包含教师权重或PANNs推理代码。

这些协议复用已探索的来源划分，HornBase场景之间还共享父录音。没有新独立道路盲测，不把重复样例窗口当独立准确率样本，也不输出道路FP/hour结论。
