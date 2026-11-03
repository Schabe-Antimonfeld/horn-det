# 鸣笛监控台：前端与后端

现有单页布局已接入真实接口。前端是原生HTML/CSS/JavaScript，无需Node、npm或构建；后端是Python标准库HTTP服务、SQLite索引和一个受管理的算法子进程，不新增第三方依赖。当前算法为单窗0.90，照片只显示声源点，不运行车辆检测。

## 电脑启动

在仓库的`deployment/pynq_z2`目录执行，使用已有uv环境和GCC：

```text
uv run --frozen python scripts/build.py
uv run --frozen python web.py
```

浏览器打开`http://127.0.0.1:8765`，选择“鸣笛音频回放”，点击启动。默认样例按采样时钟播放约2.5秒，产生4个分析窗口，首窗CONFIRMED；静音样例产生2个IDLE窗口。声音样例不会在浏览器扬声器播放。暂停画面只冻结本地预览，不停止算法；停止按钮请求算法完成当前窗口并释放资源。

默认服务数据在`deployment/pynq_z2/outputs/dashboard`。重启后保留事件、照片索引和会话状态；不自动删除历史。使用`--data NEW_DIRECTORY`可建立独立实例。只用`python -m http.server`或直接打开HTML无法提供算法接口。

## PYNQ-Z2启动

在板上用已有PYNQ/Python/NumPy/OpenCV和GCC编译C库，保留验收过的采集bit/hwh：

```text
python scripts/build.py
python web.py --host 0.0.0.0 --port 8765 --config /absolute/path/board.json
```

同一局域网电脑访问`http://板卡IP:8765`。选择“PYNQ-Z2实机采集”。配置需与现有`config/default.json`兼容，设置实测阵列/镜头标定、相机位姿和overlay路径。相机默认关闭；只有实际标定通过后才启用`vision.enabled`。商品FOV或simulation_only标定不能开启实机相机。前端不远程改阈值、硬件路径或镜头参数。

预览复用算法相机缓存的JPEG，每分析窗最多发布一次，页面约每秒请求一次；不保证视频帧率，不开启第二个USB设备，不重复做模型推理。预览仅为未标点的相机缓存；事件照片的点来自该事件声学方向投影，可在详情中切换原图/点图。回放从不把当前相机图像配给历史音频；没有同步照片时图片区为空。

相机预览3秒未更新即视为无新鲜画面。曝光时间、声画同步、真实镜头标定、ARM耗时及USB负载仍待实机验收。点只能表示估计声源方向，不能确认车辆身份。状态中的“声学处理耗时”不含采集等待和相机保存；没有CPU温度、车辆数或存储保留天数的虚构指标。

## 接口

所有接口同源；不需要配置跨域地址。

| 方法与路径 | 内容 |
|---|---|
| GET `/api/status` | 当前/最近会话、健康信息、最新窗口、UTC今日计数、门限、控制令牌 |
| POST `/api/runs` | `{"mode":"replay","source":"horn"}`、source=silence，或`{"mode":"live"}` |
| POST `/api/stop` | JSON空对象，异步请求当前进程优雅停止 |
| GET `/api/events?limit=20` | 最新分析窗口，最多100；mode明确标识回放/实机 |
| GET `/api/photos?kind=all&range=all&q=&limit=9` | 游标分页；kind=horn/candidate，range=today/week |
| GET `/api/photos?...&cursor=ID` | 查询该ID之前的照片，避免新记录插入导致分页重复 |
| GET `/media/ID/original`、`/media/ID/points` | 仅能读取已索引照片；缺失文件返回404 |
| GET `/api/preview/RUN_ID` | 当前活动会话的新鲜JPEG缓存，无帧返回404 |

POST需在`X-Control-Token`发送status返回的control_token；浏览器Origin须同源。该令牌防止普通跨站操作，不是用户登录系统。默认仅监听本机，绑定0.0.0.0用于可信局域网；本版不提供公网认证或TLS。API不接受任意命令、上传或任意文件路径。

SQLite保存每窗事件，鸣笛计数按“会话ID＋事件ID”去重；抓拍按“会话＋事件＋相机＋阶段”去重。今日统计使用UTC，页面日期时间用浏览器本地时区显示；回放记录时间是本次执行时间，另列音频采样区间，不冒充原录音日期。多个原始来源或像素候选不推断成多个车辆。

## 故障与验收

- 相机关闭、缺帧、画外或投影不确定均保留事件/说明；不拿素材图填补。素材文件仅保留在原目录，本版不加载。
- 同一数据目录有进程锁，活动会话不允许重复启动。进程异常写入失败状态和日志；服务重启将未结束会话标记interrupted，并等待旧进程心跳过期。
- 正常停止依靠stop文件，后台心跳过期也会在窗口边界停止。若硬件读挂住，服务退出等待12秒后可能强制终止自己的进程，日志提示核验FPGA/DMA状态；此后必须先完成硬件检查，不能把强制终止当作资源已干净释放。
- 本地数据包含SQLite、control配置/日志及sessions事件/JPEG。空间由操作者管理，服务不会自动清理或上传。
- 自动化验证包括真实固定权重回放、启停、持久化、去重、过滤分页、半行写入、路径边界和同源控制。硬件采集和真实USB/JPEG解码不由电脑测试替代。

可用`python web.py --frontend ABSOLUTE_FRONTEND_DIRECTORY`指定前端位置。算法ZIP已内置本文件、index.html与app.js，解压后仍用同一个web.py入口，无需复制素材或安装Node。

本轮电脑验证：213项仓库测试通过；本机Chrome验证真实鸣笛/静音回放、停止、分页、点图/原图、空筛选、手机布局与断线，共14项检查且无JavaScript异常。照片交互使用隔离的测试图片，未作为真实摄像头数据；实际公开录音回放用于验证模型与事件。
