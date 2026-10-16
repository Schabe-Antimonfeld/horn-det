# PYNQ-Z2 14 路 PDM 采集首版

目标为 Vivado 2019.2、`xc7z020clg400-1`、14 颗 IM72D128V01 和直径 240 mm 的圆阵。实现范围是 PDM 接收、抽取成 PCM16、分包、跨时钟 FIFO 和 AXI DMA 写入 PS DDR。FFT、SRP、鸣笛分类与相机控制继续使用现有参考流程。

RTL 已通过 Icarus 功能仿真；本机没有 Vivado，尚未验证 IP 参数接受、综合资源、布线、厂商原语时序或板上波形。既有 DATA 逻辑电平兼容、时钟负载和双麦交接风险仍需硬件方关闭。代码与引脚数量相符不等于阵列板已可直接连接。

## 1. 文件与处理结构

| 文件 | 用途 |
|---|---|
| `rtl/pdm_capture.v` | 采集顶层、50 ms 启动屏蔽、控制同步及调试状态 |
| `rtl/pdm_rx.v` | /4 时钟、BUFG、7 个 IDDR、双麦通道映射 |
| `rtl/pdm_decimator.v` | 每通道 5 阶 CIC /32、63 tap 补偿 FIR /2、舍入与饱和 |
| `rtl/axis_pcm.v` | 双包缓冲、包头、PCM 符号扩展、TLAST 与整包丢弃 |
| `sim/tb_capture.sv` | 总线、独立滤波参考、分包、背压、溢出与复位测试 |
| `constraints/pynq_z2.xdc` | Pmod B 映射、PDM 派生时钟、可输入的外部延迟预算 |
| `scripts/build.tcl` | 从板卡 preset 创建 PS、时钟、FIFO、DMA 与 GPIO 工程 |

```text
PS FCLK 100 MHz -> MMCM 12.288 MHz -> /4 BUFG -> PDM_CLK 3.072 MHz
                                                 |
7 DATA -> IDDR -> 14 PDM -> CIC /32 -> FIR /2 -> 48 kHz PCM16
          双边沿                     |
                           2 x 512 帧包缓冲
                                    |
                           AXI-Stream 32 bit
                                    |
                       异步 AXIS FIFO -> AXI DMA -> PS DDR
```

抽取及包缓冲工作在 12.288 MHz；DMA、控制总线和 HP0 工作在 100 MHz。跨域流使用 AMD AXIS Data FIFO，不能替换为无握手的多位寄存器采样。控制位使用两级同步；GPIO 状态中的丢包计数是逐位同步的调试快照，精确计数以包头为准。

首版 FIR 每通道复用一个 24x18 乘法通路，不是预算文档中假设的单一全局 MAC。14 通道并行、每个输出执行 63 个 tap，满足 RTL 的处理排程；实际 DSP/BRAM/LUT 映射仍以 Vivado 报告为准。

## 2. 麦克风连接和采样语义

| Pmod B 针 | FPGA 管脚 | 信号 |
|---|---|---|
| 1 | W14 | PDM_CLK 输出，送入已验证的阵列时钟分配电路 |
| 2 | Y14 | DATA0：mic 0/1 |
| 3 | T11 | DATA1：mic 2/3 |
| 4 | T10 | DATA2：mic 4/5 |
| 7 | V16 | DATA3：mic 6/7 |
| 8 | W16 | DATA4：mic 8/9 |
| 9 | V12 | DATA5：mic 10/11 |
| 10 | W13 | DATA6：mic 12/13 |
| 5、11 | GND | 公共地 |
| 6、12 | 3.3 V | 是否供阵列须由最终电源设计确定 |

每对的偶数编号麦 SELECT 接 GND，在下降沿后驱动、下一上升沿由 IDDR Q1 接收；奇数编号 SELECT 接 VDD，在上升沿后驱动、下一下降沿由 Q2 接收。12.288 MHz 域在后一安全时刻取出这两个已完成的位，作为同一个抽取序号。该固定半个 PDM 周期的采样相位关系须保留在标定记录中。

综合分支显式实例化 7 系列 `BUFG` 和 `IDDR/OPPOSITE_EDGE`；普通仿真分支是它们的理想功能替身。Icarus 不证明原语延迟、IDDR 放置、时钟路由或 setup/hold。

不能让 PYNQ base overlay 同时占用 Pmod B。板载 200 欧姆、ESD、线缆、时钟缓冲和 DATA 接收器均须纳入接口预算。XDC 的 DRIVE=8、SLEW=FAST 是初始实现设置，需要根据最终负载调整。未经电平验证，不要以本代码代替接口接收电路。

mic 0 的坐标默认对应 +x，后续编号沿 +y 方向递增；真实 PCB 如非此顺序，需在主机通道映射或算法 `coordinates` 中明确修正。直径表示声学中心圆，不是 PCB 外径。

## 3. 抽取、数值格式和前端边界

- PDM 位 1 映射为 +1，0 映射为 -1；CIC 五级积分与梳状运算采用 32 bit 模运算，不对积分器饱和。
- CIC /32 的直流增益为 2^25；结果右移 2 位，半整数远离零舍入并饱和为 signed Q1.23。
- 96 kHz 下执行 63 tap 对称 FIR，然后 /2。系数是 signed 18 bit、17 个小数位，和严格为 131072。
- FIR 保留 48 bit 累加器，再右移 25 位，半整数远离零舍入并饱和至 PCM16/Q1.15。正满量程对应 32767，负满量程对应 -32768。
- 冷启动历史由有效长度屏蔽，不要求清零 RAM；复位后不会读入上次采集的历史。

量化系数的解析频响复算：CIC+FIR 的 0–16 kHz 相对直流增益约 -0.00890 至 +0.00644 dB；FIR 的 24–48 kHz 最大响应约 -86.79 dB。它们是离散滤波器频响，不是麦克风实测 SNR、完整 PDM 量化噪声或道路识别结果。16–24 kHz 属于过渡带，不能把此版本当成到 20 kHz 都平坦的前端。

系数可在现有 prototype Python 环境用下列方法复算，不增加依赖：

```python
import numpy as np
from scipy.signal import firls, freqz

points = np.linspace(0, 16000, 17)
cic = np.ones_like(points)
cic[1:] = (np.sin(np.pi * points[1:] / 96000) /
           (32 * np.sin(np.pi * points[1:] / 3072000))) ** 5
bands, desired = [], []
for index in range(16):
    bands.extend(points[index:index + 2])
    desired.extend(1 / cic[index:index + 2])
bands.extend([24000, 48000])
desired.extend([0, 0])
values = firls(63, bands, desired, weight=[1] * 16 + [1000], fs=96000)
values /= values.sum()
codes = np.sign(values) * np.floor(np.abs(values) * 131072 + 0.5)
codes = codes.astype(np.int64)
codes[31] += 131072 - codes.sum()
print(codes.tolist())
frequency, response = freqz(codes / 131072, worN=65536, fs=96000)
```

本版 PL 不做 DC 去除和增益/延迟标定，保留给 [Python 流水线](../prototype/src/horn/pipeline.py)，避免重复执行。硬件前端频响与公开数据的录音前端不同，分类模型和频率先验仍需真实 PCM 验证。名义线性相位滤波群延迟约 348.145 微秒，另有接收和实现流水；完整音频时间轴不能用 Linux 收包时刻代替。

## 4. DMA 包格式与丢包策略

每包为 8 个 uint32 包头字，再跟 512x14 个 int32 音频字。默认共 7176 字、28704 byte，小端。每个音频字是 PCM16 的符号扩展，不是左移到高 16 位。先时间、后通道：`sample0/ch0..13, sample1/ch0..13, ...`。

| uint32 字序号 | 内容 |
|---|---|
| 0 | `0x314d4450`，内存字节为 ASCII `PDM1` |
| 1 | 高 16 bit 格式/滤波版本=1，bits 15:8 包头字数=8，bits 7:0 通道数=14 |
| 2、3 | 首样本计数 low/high，合成 uint64 |
| 4 | 累计整包丢弃计数，uint32，计入正在启用采集但无空闲包缓冲的情况 |
| 5 | 每通道样本数=512 |
| 6 | 名义 PCM 采样率=48000 |
| 7 | bit 0：FIR 输出曾发生饱和；其余保留为 0。该位是复位后累计状态，不是本包峰值标志 |

包头与所有 payload 字的 TKEEP 均为 0xf；仅最后一个音频字 TLAST=1。背压时 TVALID、TDATA 和 TLAST 保持，不能暂停或拉伸麦克风时钟。无缓冲时丢弃完整 512 样本组，采样计数继续前进，不输出拼接或半包数据。

样本计数从启动屏蔽完成后的第一份 PCM 起计，关闭采集时也继续前进。启用/停用只在组边界决定是否保存该组；停用会完成并排空已经开始的包，不是 AXIS 复位。异步 FIFO 或 DMA 中可能仍有旧数据；重新开始独立录音应重新下载 overlay，使整条链路复位。

两份包缓冲仅提供约 21.33 ms 的原始容量，后面另有 FIFO；不是无限 DDR 环形采集器。首版使用 simple DMA，Python/Linux 重装 DMA 过慢时会丢包，包头必须检查。长时间无损采集需要后续独立评估 SG DMA/驱动与 CPU 负载。

## 5. 本地仿真

在仓库根目录，使用已安装的 Icarus：

```powershell
iverilog -g2012 -Wall -s tb_capture -o "$env:TEMP\horn_capture_tests.vvp" fpga/rtl/pdm_rx.v fpga/rtl/pdm_decimator.v fpga/rtl/axis_pcm.v fpga/rtl/pdm_capture.v fpga/sim/tb_capture.sv
vvp "$env:TEMP\horn_capture_tests.vvp"
```

可增加 `'-Ptb_capture.PAIRS=6'` 检查 12 通道参数化；PowerShell 中需保留引号。板级 Tcl/XDC 固定为本次确认的 14 通道，不会因此自动改成另一块板。

测试覆盖：带传播/高阻交接的双麦总线和通道顺序；用五个 boxcar 卷积得到的独立 CIC FIR 参考；宽整数 FIR、半整数舍入、饱和、正负满量程；启动屏蔽、复位后的历史清理；512 帧包长度、符号扩展、TLAST/TKEEP、随机及持续背压、整包溢出、停用及中包复位。原语替身和零延迟 RTL 不替代 Vivado 厂商库及板级验证。

本次使用 Icarus 14.0 完成 14/12 通道仿真；另使用 [Xilinx 公开 Unisim 库](https://github.com/Xilinx/XilinxUnisimLibrary/tree/1c8e05fd1e9a79ceb8b996a0996674122eed086f)的 `BUFG.v`、`IDDR.v` 和 `glbl.v` 编译 `SYNTHESIS` 分支并运行同一 14 通道测试，通过功能检查。模型保留在本机临时目录，未分发或修改；该检查未开启器件时序验证，不等于 Vivado 2019.2 的厂商库/SDF 验收。Vivado Tcl 另做语法与模拟命令流程检查，板端接收示例通过模拟 DMA 的正常、数据间断、格式错误和溢出检查，均不代表实际运行过 Vivado 或 PS DMA。

## 6. 创建 Vivado 2019.2 工程

需先在 Vivado 安装适配 2019.2 的官方 PYNQ-Z2 board files，确认板卡 preset 的 DDR、PS 时钟和 MIO 与实物一致。板卡资料入口：[TUL PYNQ-Z2](https://www.tulembedded.com/FPGA/ProductsPYNQ-Z2.html)。脚本不下载或执行网络脚本，也不会套用其他板卡的 DDR 参数。

在 Vivado 命令行运行，输出目录应放在仓库外，且不能已有同名工程：

```powershell
vivado -mode batch -source D:/SpecialEffects/components/horn-det/fpga/scripts/build.tcl -tclargs D:/fpga_build/horn_create D:/board_files create
vivado -mode batch -source D:/SpecialEffects/components/horn-det/fpga/scripts/build.tcl -tclargs D:/fpga_build/horn_synth D:/board_files synth
```

如果 board files 已在 Vivado 默认库中，把第二个路径替换为 `-`。默认创建工程，不自动生成 bitstream。脚本配置 FCLK100、MMCM M48/D5/O78.125、12.288 MHz，之后 RTL /4；会检查 Clocking Wizard 是否保留所需频率参数。FCLK 与音频钟复位由 PS reset 和 LOCKED 管理，PDM 时钟生成未通过前不要接阵列负载。

若要执行 `bit`，先根据最终 CLK 分配、DATA 接收器、线缆、FPGA 阈值与负载，设置环境变量 `PDM_INPUT_MIN_NS`、`PDM_INPUT_MAX_NS`。它们表示相对内部 PDM 时钟边沿，到 FPGA DATA 输入变化/有效的完整外部 min/max 延迟。min 必须覆盖旧数据最早释放，不能直接使用新数据 tDD_min；max 必须保证到 FPGA 实际逻辑阈值有效。不能把文献里的占位 10 ns 或单独的 tDV 当成整个接口预算。

设置经确认的两项值后，用另一个新输出目录把模式改为 `bit`。XDC 对上升和下降沿分别施加输入延迟，不对 DATA false-path。工程会生成时序、CDC、DRC 和资源报告；负 slack 时拒绝写 bitstream。仍须人工确认无未约束关键端点、检查 IDDR/BUFG 放置及复位路径，且在 mic CLK pad 测量 48–52% 占空比和不超过 13 ns 的边沿。

正常产物为输出目录下 `horn_capture.bit` 和 `horn_capture.hwh`，两者需同名并一起拷到板端。Vivado/HWH 与 PYNQ 镜像版本兼容性仍需在目标板确认。工程不生成 USB 相机、云台或完整 PS 软件镜像。

## 7. PYNQ 板端接收示例

使用适配 PYNQ-Z2 的现有 PYNQ 镜像。示例只使用 PYNQ 自带的 numpy 和 Python 标准库；每次重新下载 overlay，先接收 DMA，再启用采集。不要同时运行 base overlay。

```python
from pathlib import Path
import json
import time
import wave
import numpy as np
from pynq import Overlay, allocate

overlay = Overlay('/home/xilinx/horn_capture.bit')
control = overlay.capture_control
dma = overlay.audio_dma
control.write(0x00, 0)
deadline = time.monotonic() + 2
while not (control.read(0x08) & 2):
    if time.monotonic() > deadline:
        raise TimeoutError('Audio clock/reset/warmup did not complete')
    time.sleep(0.001)

channels, hop = 14, 512
buffer = allocate(shape=(8 + channels * hop,), dtype=np.uint32)
blocks, headers = [], []
expected = None
try:
    for index in range(100):
        dma.recvchannel.transfer(buffer)
        if index == 0:
            control.write(0x00, 1)
        deadline = time.monotonic() + 2
        while not (dma.mmio.read(0x34) & 2):
            if dma.mmio.read(0x34) & 0x70:
                raise RuntimeError('AXI DMA S2MM error')
            if time.monotonic() > deadline:
                raise TimeoutError('No complete DMA packet')
            time.sleep(0.0001)
        dma.recvchannel.wait()
        header = [int(value) for value in buffer[:8]]
        if header[0] != 0x314d4450 or header[1] != 0x0001080e:
            raise RuntimeError('Unexpected packet format')
        if header[5:7] != [hop, 48000] or header[7] & ~1:
            raise RuntimeError('Unexpected capture configuration')
        first = header[2] | (header[3] << 32)
        if expected is not None and first != expected:
            raise RuntimeError('Audio gap detected; do not concatenate as continuous WAV')
        if header[4] != 0:
            raise RuntimeError('Capture overflow detected')
        if header[7] & 1:
            print('FIR saturation has occurred since reset; check levels')
        expected = first + hop
        blocks.append(buffer[8:].view(np.int32).astype('<i2').reshape(hop, channels).copy())
        headers.append(header)
finally:
    control.write(0x00, 0)
    dma.mmio.write(0x30, 4)
    deadline = time.monotonic() + 2
    while dma.mmio.read(0x30) & 4 and time.monotonic() < deadline:
        time.sleep(0.001)
    if dma.mmio.read(0x30) & 4:
        raise RuntimeError('DMA reset timed out; buffer remains allocated')
    buffer.freebuffer()

audio = np.concatenate(blocks)
with wave.open('array14.wav', 'wb') as output:
    output.setnchannels(channels)
    output.setsampwidth(2)
    output.setframerate(48000)
    output.writeframes(audio.tobytes())
Path('array14.json').write_text(json.dumps({
    'headers': headers, 'sample_rate': 48000, 'channels': channels,
    'diameter_m': 0.24, 'firmware_format': 1,
    'clock_origin': 'PCM counter after hardware warmup; not Linux wall time',
    'evidence': 'captured audio; channel calibration and road performance not established'
}, indent=2), encoding='utf-8')
```

GPIO 状态：bit 0 为音频域同步后的 enable；bit 1 为启动屏蔽结束；bit 2 为 FIR 饱和累计标志；bits 31:8 为丢包计数低 24 位的非原子调试快照。字节偏移 0x00 控制 enable，0x08 读取状态。bit 1 仅表示计时结束，不检测麦克风在线、DATA 卡死或输入音频真实性；这些仍需用原始 PCM 和接口波形检查。

将 WAV 和 JSON 保留在现有被忽略的数据目录，再通过 `--profile` 指定 `microphones=14`、`diameter=0.24`、`sample_rate=48000`。先检查静音、已知声源、逐通道映射和同步；没有可靠背景时不要启用背景初始化，没有声源归属证据时不要启用 `--association-verified`。生产参数、分类阈值和实际精度仍按 [原型边界](../prototype/README.md)验收。

## 8. 还需要的板上验收

先验证电平、时钟、全通道 PDM 与 48 kHz PCM，再运行 DMA 长录音及 USB 并发压力测试。保存无丢包证据、计数连续性、频响/噪声、通道增益/相位/延迟标定。时间戳需建立 PCM 计数与 PS/相机时钟的关系，不能直接使用收包时刻。以上全部通过前，不将功能仿真结果描述为已验证的板上采集性能。

器件与接口依据：[Infineon IM72D128V01](https://www.infineon.com/assets/row/public/documents/24/49/infineon-im72d128-datasheet-en.pdf)、[AMD 7 系列 IDDR](https://docs.amd.com/r/2019.2-English/ug953-vivado-7series-libraries/IDDR)、[AMD AXI DMA PG021](https://docs.amd.com/r/en-US/pg021_axi_dma)、[项目 FPGA 章节](../docs/09_FPGA模块划分与定点实现.md)。
