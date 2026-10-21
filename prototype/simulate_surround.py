"""Run reproducible fixed-camera geometry and real-audio array simulations."""

import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
from dataclasses import asdict, replace
import hashlib
from html import escape
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as element_tree

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from horn.calibration import remove_dc
from horn.config import Config
from horn.frontend import read_audio
from horn.geometry import circular_array
from horn.surround import (AcousticEvidence, Association, Bearing, Detection, FrozenPanns, Rig, associate, bearing_candidates,
                           coverage_grid, form_beam, propagate_audio, route_cameras,
                           select_bearing, spatial_response, vehicle_box, wrap_angle)


PROTO = Path(__file__).resolve().parent


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def coverage_summary(rig, cameras=None):
    points, visible = coverage_grid(rig, cameras=cameras)
    counts = visible.sum(axis=1)
    return dict(cells=len(points), covered=int((counts > 0).sum()),
                fraction=float(np.mean(counts > 0)), overlap=float(np.mean(counts > 1)),
                definition="equal-area 1 m ground-cell centers; geometry only, no occlusion or pixel-size test")


def coverage_svg(path, rig, cameras):
    points, visible = coverage_grid(rig, cameras=cameras)
    counts = visible.sum(axis=1)
    scale = 620 / (2 * rig.maximum_range)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="800" height="760" viewBox="0 0 800 760">',
             '<rect width="800" height="760" fill="white"/>',
             '<text x="24" y="28" font-size="18">SIMULATED GROUND COVERAGE: geometric visibility only</text>']
    colors = ["#d94b4b", "#89ba73", "#467bb5"]
    for point, count in zip(points, counts):
        x, y = 400 + point[0] * scale, 370 - point[1] * scale
        parts.append(f'<rect x="{x-scale/2:.2f}" y="{y-scale/2:.2f}" width="{scale:.2f}" height="{scale:.2f}" fill="{colors[min(count,2)]}"/>')
    parts.extend(['<circle cx="400" cy="370" r="4" fill="black"/>',
                  '<text x="25" y="710" font-size="15">Red: blind | Green: 1 camera | Blue: 2+ cameras</text>',
                  f'<text x="25" y="735" font-size="15">n={rig.count}, h={rig.height:g}m, pitch={rig.pitch:g}deg, radius={rig.maximum_range:g}m</text>', '</svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")


def annotation_svg(path, camera, detections, result):
    matches = {item["identity"] for group in result["groups"] for item in group}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{camera.width}" height="{camera.height + 60}" viewBox="0 0 {camera.width} {camera.height + 60}">',
             f'<rect width="{camera.width}" height="{camera.height + 60}" fill="#e9edf1"/>',
             '<text x="12" y="24" font-size="17">SYNTHETIC BOX INPUT: no real image or vehicle detector</text>']
    for detection in detections:
        if detection.camera != camera.identity:
            continue
        x, y, right, bottom = detection.box
        matched = detection.identity in matches
        color = ("#24813b" if result["state"] == "unique" else "#b76700") if matched else "#728092"
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{right-x:.2f}" height="{bottom-y:.2f}" fill="#a8b2bc" fill-opacity="0.3" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<circle cx="{detection.foot[0]:.2f}" cy="{detection.foot[1]:.2f}" r="4" fill="{color}"/>')
        parts.append(f'<text x="{x:.2f}" y="{max(42,y-6):.2f}" font-size="16" fill="{color}">{escape(detection.identity)}</text>')
    parts.extend([f'<text x="12" y="{camera.height+35}" font-size="18">Camera {escape(camera.identity)} | geometric association: {escape(result["state"])}</text>', '</svg>'])
    path.write_text("\n".join(parts), encoding="utf-8")


def observations(rig, cameras, centers, velocities, exposure, generator, pixel_sigma=0.0,
                 miss_probability=0.0, drop_target=False):
    """Generate cuboid boxes and noisy ground anchors; infer velocity from two anchors."""
    detections, truth = [], {}
    for vehicle, (center, velocity) in enumerate(zip(centers, velocities)):
        velocity = np.asarray(velocity)
        present = np.asarray(center) + velocity * exposure
        previous = present - velocity * 0.1
        for camera in cameras:
            if (drop_target and vehicle == 0) or generator.random() < miss_probability:
                continue
            projected = vehicle_box(camera, present)
            if projected is None:
                continue
            box, foot = projected
            foot = np.asarray(foot) + generator.normal(0, pixel_sigma, 2)
            previous_pixel, previous_visible = camera.project([[*previous, 0]])
            estimated_velocity = np.zeros(2)
            if previous_visible[0]:
                previous_pixel += generator.normal(0, pixel_sigma, (1, 2))
                points, valid = camera.ground([previous_pixel[0], foot])
                if valid.all():
                    estimated_velocity = (points[1, :2] - points[0, :2]) / 0.1
            identity = f"observation-{len(detections)}"
            detections.append(Detection(identity, camera.identity, box, tuple(foot), exposure,
                                        tuple(estimated_velocity)))
            truth[identity] = vehicle
    return detections, truth


def evaluate_match(result, truth):
    groups = [{truth[item["identity"]] for item in group} for group in result["groups"]]
    unique = result["state"] == "unique"
    return dict(correct_unique=unique and groups == [{0}],
                incorrect_unique=unique and groups != [{0}],
                target_retained=any(0 in group for group in groups),
                mixed_cluster=any(len(group) > 1 for group in groups))


def summarize_visual(rows):
    count = len(rows)
    return dict(count=count, correct_unique=sum(row["evaluation"]["correct_unique"] for row in rows),
                incorrect_unique=sum(row["evaluation"]["incorrect_unique"] for row in rows),
                ambiguous=sum(row["association"]["state"] == "ambiguous" for row in rows),
                unmatched=sum(row["association"]["state"] == "unmatched" for row in rows),
                target_retained=sum(row["evaluation"]["target_retained"] for row in rows),
                mixed_clusters=sum(row["evaluation"]["mixed_cluster"] for row in rows),
                routing_hits=sum(row["routing_hit"] for row in rows),
                visually_available=sum(row["visually_available"] for row in rows))


def analytic_evidence(azimuth, depression, broad=False):
    """Inject a labeled analytic response, not a measured acoustic localization result."""
    azimuths = np.arange(-180, 180, 2.0)
    depressions = np.arange(0, 91, 3.0)
    response = np.exp(-0.5 * (wrap_angle(azimuths[:, None] - azimuth) / 2) ** 2
                      -0.5 * ((depressions[None, :] - depression) / (12 if broad else 2)) ** 2)
    return AcousticEvidence(Bearing(azimuth, depression, float(response.max())),
                            (azimuths, depressions, response))


def compare_association(rig, azimuth, arrival, detections, cameras, evidence, settings, truth):
    results = {}
    for method in ("azimuth", "depression", "spatial"):
        result = associate(rig, azimuth, arrival, detections, replace(settings, method=method), cameras, evidence)
        results[method] = dict(association=result, evaluation=evaluate_match(result, truth))
    return results


def visual_trials(rig, cameras, seed, output, association):
    generator = np.random.default_rng(seed)
    modes = ["clean", "perturbed", "boundary", "near", "collinear", "missed_collinear", "stale", "moving", "calibration",
             "broad_response", "missing_evidence"]
    rows, examples = [], []
    for mode in modes:
        for index in range(16):
            angle = float(generator.uniform(-180, 180))
            radius = float(generator.uniform(8, 30))
            if mode == "boundary":
                angle = float(wrap_angle(rig.yaw + (index % rig.count + 0.5) * 360 / rig.count))
            if mode == "near":
                radius = 2.0
            if mode in ("collinear", "missed_collinear"):
                radius = 12.0
            angle_rad = np.deg2rad(angle)
            target = radius * np.array([np.cos(angle_rad), np.sin(angle_rad)])
            other_angle = angle_rad if "collinear" in mode else angle_rad + np.pi / 2
            other = 24 * np.array([np.cos(other_angle), np.sin(other_angle)])
            centers = [target, other]
            velocity = 12 * np.array([-np.sin(angle_rad), np.cos(angle_rad)]) if mode == "moving" else np.zeros(2)
            exposure = 1.2 if mode == "stale" else 0.25 if mode == "moving" else 0.0
            detections, truth = observations(rig, cameras, centers, [velocity, np.zeros(2)], exposure,
                                              generator, pixel_sigma=2 if mode == "perturbed" else 0,
                                              miss_probability=0.1 if mode == "perturbed" else 0,
                                              drop_target=mode == "missed_collinear")
            measured_angle = float(wrap_angle(angle + (generator.normal(0, 3) if mode == "perturbed" else 0)))
            depression = float(np.rad2deg(np.arctan2(rig.height - rig.horn_height, radius)))
            measured_depression = float(np.clip(depression + (generator.normal(0, 1.5) if mode == "perturbed" else 0), 0, 90))
            evidence = None if mode == "missing_evidence" else analytic_evidence(measured_angle, measured_depression, mode == "broad_response")
            selected = route_cameras(rig, measured_angle, cameras=cameras)
            available = {item.camera for item in detections if truth[item.identity] == 0}
            selected_detections = [item for item in detections if item.camera in selected]
            calibrated = [replace(camera, yaw=camera.yaw + 2) for camera in cameras] if mode == "calibration" else cameras
            arrival = np.hypot(radius, rig.height - rig.horn_height) / 343
            comparisons = compare_association(rig, measured_angle, arrival, selected_detections,
                                               calibrated, evidence, association, truth)
            result = comparisons[association.method]["association"]
            row = dict(mode=mode, index=index, truth_angle=angle, measured_angle=measured_angle,
                       source_position=target.tolist(), selected_cameras=selected,
                       visually_available=bool(available), routing_hit=bool(available.intersection(selected)),
                       measured_depression=measured_depression, evidence="analytic injected response" if evidence else "missing",
                       comparisons=comparisons, association=result, evaluation=evaluate_match(result, truth))
            rows.append(row)
            if index == 0 and selected_detections:
                camera = next(item for item in cameras if item.identity == selected_detections[0].camera)
                name = f"annotation-{mode}.svg"
                annotation_svg(output / name, camera, selected_detections, result)
                examples.append(name)
    summary = {mode: summarize_visual([row for row in rows if row["mode"] == mode]) for mode in modes}
    write_json(output / "visual-cases.json", rows)
    write_json(output / "visual-report.json", summary)
    comparison = {method: {mode: summarize_visual([dict(row, **row["comparisons"][method])
                                                   for row in rows if row["mode"] == mode]) for mode in modes}
                  for method in ("azimuth", "depression", "spatial")}
    write_json(output / "visual-comparison.json", comparison)
    return summary, examples


def source_waveform(record):
    checksum = digest(Path(record["path"]))
    if record.get("sha256") != checksum:
        raise ValueError("A recording no longer matches the frozen source manifest")
    audio = read_audio(record["path"], 48000, channel=0)
    audio, _ = remove_dc(audio)
    signal = audio[:, 0]
    if len(signal) < 48000:
        raise ValueError("Simulation recordings must contain one complete second")
    starts = sorted(set(range(0, len(signal) - 48000 + 1, 24000)) | {len(signal) - 48000})
    start = max(starts, key=lambda offset: np.mean(signal[offset:offset + 48000] ** 2))
    signal = signal[start:start + 48000]
    rms = float(np.sqrt(np.mean(signal ** 2)))
    if rms <= 0:
        raise ValueError("Source recording has no energy")
    return signal / rms, dict(file=record["file"], category=record["category"], dataset=record["dataset"],
                              split=record["split"], sha256=checksum, crop_start=start,
                              selection="highest-energy completed 1 s crop; simulation stimulus only")


def acoustic_trials(rig, cameras, settings, seed, output, association):
    import torch
    torch.set_num_threads(2)
    base = PROTO / "outputs" / "horn_detector_v1"
    checkpoint = base / "panns-horn.pt"
    records_path = base / "clips.json"
    frozen = {str(path.relative_to(PROTO)): digest(path) for path in (checkpoint, records_path)}
    records = json.loads(records_path.read_text(encoding="utf-8"))
    selected = sorted([item for item in records if item["dataset"] == "ESC50" and item["split"] == "test"],
                      key=lambda item: item["file"])
    horns = [item for item in selected if item["label"] == 1][:2]
    backgrounds = [next(item for item in selected if item["category"] == category) for category in ("engine", "siren")]
    sources = [source_waveform(item) for item in horns + backgrounds]
    if len(horns) != 2:
        raise ValueError("Two existing ESC50 test horn recordings are required")
    network = FrozenPanns(checkpoint)
    config = Config(microphones=14, diameter=0.24)
    coordinates = circular_array(config)
    generator = np.random.default_rng(seed)
    rows = []
    positions = [(-179, 60), (179, -60), (-90, 40), (0, 120), (90, -20), (45, -135)]
    plan = [(True, source, radius, angle, other, "standard") for source in range(2)
            for radius in (4, 15, 35) for angle, other in positions]
    plan += [(False, 2, 15, angle, other, "standard") for angle, other in positions]
    plan += [(True, source, 12, angle, angle, mode) for source in range(2)
             for angle in (-179, -45, 30, 120) for mode in ("collinear", "missed_collinear")]
    for identity, (positive, source_index, radius, angle, other_angle, mode) in enumerate(plan):
        center = radius * np.array([np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
        other_radius = 24 if "collinear" in mode else radius + 3
        other = other_radius * np.array([np.cos(np.deg2rad(other_angle)), np.sin(np.deg2rad(other_angle))])
        source = np.r_[center, rig.horn_height - rig.height]
        interferer = np.r_[other, rig.horn_height - rig.height]
        audio, propagation = propagate_audio(sources[source_index][0], source, coordinates, config)
        other_index = 2 + identity % 2 if positive else 3
        noise, _ = propagate_audio(sources[other_index][0], interferer, coordinates, config)
        audio += noise * 10 ** (5 / 20)
        audio += generator.normal(0, 10 ** (-25 / 20), audio.shape)
        began = time.perf_counter()
        grid = spatial_response(audio, coordinates, config)
        candidates = bearing_candidates(*grid, count=settings["candidate_count"])
        beams = [form_beam(audio, coordinates, bearing, config)[0] for bearing in candidates]
        scores = network.score(beams)
        decision = select_bearing(candidates, scores, settings["threshold"], settings["margin"])
        elapsed = (time.perf_counter() - began) * 1000
        chosen = candidates[decision["index"]] if decision["index"] is not None else None
        estimate = chosen.azimuth if chosen is not None else None
        ordinary = float(grid[0][np.argmax(grid[2][:, 0])])
        selected_cameras = route_cameras(rig, estimate, cameras=cameras) if decision["valid"] else []
        detections, truth = observations(rig, cameras, [center, other], [np.zeros(2), np.zeros(2)],
                                          0.5, generator, drop_target=mode == "missed_collinear")
        selected_detections = [item for item in detections if item.camera in selected_cameras]
        evidence = AcousticEvidence(chosen, grid) if decision["valid"] and chosen is not None else None
        comparisons = compare_association(rig, estimate, 0.5 + propagation["flight_seconds"],
                                           selected_detections, cameras, evidence, association, truth)
        result = comparisons[association.method]["association"]
        evaluation = evaluate_match(result, truth) if positive else None
        error = float(abs(wrap_angle(estimate - angle))) if positive and estimate is not None else None
        rows.append(dict(id=identity, mode=mode, positive=positive, source=sources[source_index][1]["file"],
                         interferer=sources[other_index][1]["file"], radius=radius, truth_angle=angle if positive else None,
                         source_position=source.tolist(), propagation=propagation, candidates=[asdict(item) for item in candidates],
                         scores=scores, decision=decision, angle_error=error, baseline_angle=ordinary,
                         baseline_error=float(abs(wrap_angle(ordinary - angle))) if positive else None,
                         candidate_hit=any(item.azimuth is not None and abs(wrap_angle(item.azimuth-angle)) < 10 for item in candidates) if positive else None,
                         selected_cameras=selected_cameras, comparisons=comparisons, association=result, evaluation=evaluation,
                         processing_ms=elapsed))
        print(f"acoustic {identity + 1}/{len(plan)} accepted={decision['valid']} score={decision['score']} angle={estimate}", flush=True)
    positives = [row for row in rows if row["positive"]]
    negatives = [row for row in rows if not row["positive"]]
    accepted = [row for row in positives if row["decision"]["valid"]]
    summary = dict(positive_count=len(positives), negative_count=len(negatives),
                   candidate_hits=sum(row["candidate_hit"] for row in positives),
                   accepted_positive=len(accepted),
                   correct_direction=sum(row["decision"]["valid"] and row["angle_error"] is not None and row["angle_error"] < 10 for row in positives),
                   baseline_correct=sum(row["baseline_error"] < 10 for row in positives),
                   accepted_negative=sum(row["decision"]["valid"] for row in negatives),
                   correct_unique_vehicle=sum(row["evaluation"]["correct_unique"] for row in positives),
                   incorrect_unique_vehicle=sum(row["evaluation"]["incorrect_unique"] for row in positives),
                   median_ms=float(np.median([row["processing_ms"] for row in rows])),
                   p90_ms=float(np.percentile([row["processing_ms"] for row in rows], 90)),
                   timing_scope="one completed 1 s window: STFT + full 360/elevation SRP + beams + PANNs; includes first-call warmup; excludes synthesis, I/O, routing and association",
                   settings=settings, sources=[item[1] for item in sources], frozen_hashes=frozen,
                   caution="58 constructed stationary scenes from only 2 horn and 2 nonhorn recordings; inherited/unvalidated gates, no threshold tuning, no deployment claims")
    summary["association_comparison"] = {
        method: {mode: dict(count=len(selected),
                           correct_unique=sum(row["comparisons"][method]["evaluation"]["correct_unique"] for row in selected),
                           incorrect_unique=sum(row["comparisons"][method]["evaluation"]["incorrect_unique"] for row in selected),
                           ambiguous=sum(row["comparisons"][method]["association"]["state"] == "ambiguous" for row in selected),
                           unmatched=sum(row["comparisons"][method]["association"]["state"] == "unmatched" for row in selected))
                 for mode in ("standard", "collinear", "missed_collinear")
                 for selected in [[row for row in positives if row["mode"] == mode]]}
        for method in ("azimuth", "depression", "spatial")}
    for path, checksum in frozen.items():
        if digest(PROTO / path) != checksum:
            raise ValueError("A frozen source manifest or checkpoint changed")
    write_json(output / "acoustic-cases.json", rows)
    write_json(output / "acoustic-report.json", summary)
    return summary


def report(output, config, coverage, visual, acoustic, examples):
    rig = config["rig"]
    lines = ["# 可调多相机鸣笛定位与车辆关联仿真", "",
             "本报告是可复现的工程原型验证。图像输入是几何生成的车辆框与地面锚点，不含真实车辆检测、真实跟踪或实际相机采集；唯一几何匹配不等于已证明该车鸣笛。", "",
             f"参数：环形相机 {rig['count']} 个＋下视 1 个，高度 {rig['height']} m，向下俯角 {rig['pitch']}°，视场 {rig['hfov']}°×{rig['vfov']}°，监控距离 {rig['minimum_range']}～{rig['maximum_range']} m。配置不是硬件推荐或冻结值。", "",
             "## 地面覆盖", "",
             f"1 m 等面积网格中心共 {coverage['cells']} 个，几何覆盖 {coverage['fraction']:.2%}，至少两个相机覆盖 {coverage['overlap']:.2%}。未计遮挡、畸变、目标像素尺寸；采样不能证明连续地面无缝覆盖。", "",
             "[覆盖图](coverage.svg)；[参数扫描](coverage-sweep.json)；[完整配置](configuration.json)。默认扫描 36 组；若指定较高喇叭高度，则跳过低于喇叭的安装高度。扫描使用理想均匀布局，不应用逐相机安装覆盖项。", "",
             "## 独立声画关联检查", "",
             "本组注入已知方向/俯角或指定误差，并用方位标准差2°、俯角标准差2°的解析高斯响应检查接口；不运行声音定位或分类器，不能把注入真值当成实测俯角。车辆框来自 4.5×1.8×1.5 m 长方体，速度由相隔0.1 s的两个模拟地面锚点反投影求差。", "",
             "| 条件 | 数量 | 正确唯一 | 错误唯一 | 歧义 | 未匹配 | 保留目标 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for mode, value in visual.items():
        lines.append(f"| {mode} | {value['count']} | {value['correct_unique']} | {value['incorrect_unique']} | {value['ambiguous']} | {value['unmatched']} | {value['target_retained']} |")
    lines.extend(["", "clean：无扰动；perturbed：方位标准差3°、俯角标准差1.5°、锚点标准差2 px、逐相机漏框率10%；boundary：相机分界；near：距离2 m；collinear：12/24 m同方向双车；missed_collinear：额外漏掉目标全部框；stale：图像晚1.2 s；moving：12 m/s切向运动、图像晚0.25 s；calibration：标定朝向偏差2°；broad_response：解析俯角响应标准差12°；missing_evidence：不给声学证据。", "",
                  "[三种关联方式对照](visual-comparison.json)使用完全相同的框、时间和输入证据。azimuth为显式旧基线，depression增加高度/位置区间的俯角检查，spatial再检查SRP支持及峰宽。默认spatial缺少证据时不回退到只看角度。", "",
                  f"当前门限（未在新道路数据校准）：{json.dumps(config['association'], ensure_ascii=False)}。SRP支持相对PANNs选中的鸣笛候选峰，而非可能属于发动机的全局最高峰；接受后仍只能称为声学兼容候选，不能视作违规归属证据。", "",
                  "[逐场景记录](visual-cases.json)；[分条件指标](visual-report.json)。以下均为合成框示意图：", ""])
    lines.extend([f"- [{name}]({name})" for name in examples])
    lines.extend(["", "## 真实录音合成阵列闭环", ""])
    if acoustic is None:
        lines.append("本次使用 --geometry-only，未执行 PANNs/声学仿真。")
    else:
        count = acoustic["positive_count"]
        lines.extend([f"使用冻结PANNs、2条ESC50测试鸣笛与2条测试干扰录音，构造{count}个有鸣笛、{acoustic['negative_count']}个无鸣笛场景。标准组36正例目标距离4/15/35 m；另有16个12/24 m同方向双车压力场景，其中8个隐藏全部目标车框。干扰比目标强5 dB，白噪声为单位目标RMS下25 dB。俯角与响应全部从合成阵列音频估计，没有将真实车辆距离传入关联证据。", "",
                      f"- 水平来声基线最强峰正确方向：{acoustic['baseline_correct']}/{count}。",
                      f"- 考虑俯角的前 {acoustic['settings']['candidate_count']} 个候选包含正确方向：{acoustic['candidate_hits']}/{count}。",
                      f"- 波束＋PANNs 接受正例：{acoustic['accepted_positive']}/{count}；正确接受方向（误差＜10°）：{acoustic['correct_direction']}/{count}。",
                      f"- 无鸣笛场景接受：{acoustic['accepted_negative']}/{acoustic['negative_count']}；不能换算误报/小时。",
                      f"- 正例闭环正确唯一车辆：{acoustic['correct_unique_vehicle']}/{count}；错误唯一车辆：{acoustic['incorrect_unique_vehicle']}/{count}。",
                      f"- 声学处理耗时中位数 {acoustic['median_ms']:.1f} ms，P90 {acoustic['p90_ms']:.1f} ms；包含首次调用，非板上性能或鸣笛起点延迟。", "",
                      "方向正确不要求距离估计正确；地面位置来自模拟视觉锚点。传播使用精确球面相对延时及 1/r 通道幅度差，不加入反射，波形去除公共飞行时间并保留滤波缓冲，绝对飞行时间在关联时间中显式处理。阵列为理想 14 路、直径 0.24 m，48 kHz；SRP 为 2°方位×3°向下俯角，PHAT 逐帧归一化后平均。", "",
                      "分类阈值 0.2、分差 0 继承旧实验，仅用于接通流程；未在本 360°高位场景验证或调参。两条鸣笛录音反复使用，裁剪为原录音最高能量的一秒，不含事件起止标注，不能推断短鸣召回或事件延迟。整车静止；移动只在独立视觉关联组验证，未完成动态声场闭环。", "",
                      "[声学指标与来源哈希](acoustic-report.json)；[逐场景声学与匹配输出](acoustic-cases.json)。"])
        lines.extend(["", "三种关联方式在同一音频、同一候选、同一框上的对照（正确唯一／错误唯一／歧义／未匹配）：", "",
                      "| 方法 | 标准正例36 | 同方向双车8 | 目标漏检8 |", "|---|---|---|---|"])
        for method, groups in acoustic["association_comparison"].items():
            values = [" / ".join(str(groups[mode][key]) for key in ("correct_unique", "incorrect_unique", "ambiguous", "unmatched"))
                      for mode in ("standard", "collinear", "missed_collinear")]
            lines.append(f"| {method} | {' | '.join(values)} |")
        lines.extend(["", "目标漏检与双车可见组的分类/定位结果可能受相干干扰影响；必须同时查看错误唯一与正确唯一损失。若两车在当前阵列分辨能力内不可区分，或PANNs选择了含鸣笛泄漏的错误波束，本方法仍不能保证排除错配，不得针对这组测试调阈值隐藏失败。"])
    lines.extend(["", "## 运行与边界", "", "```powershell",
                  r".\prototype\.venv\Scripts\python.exe -B prototype/simulate_surround.py --output prototype/outputs/surround_v2",
                  r".\prototype\.venv\Scripts\python.exe -B prototype/simulate_surround.py --geometry-only --cameras 8 --height 8 --pitch 45 --output prototype/outputs/surround_custom",
                  "```", "", "不下载数据或模型，不训练模型，不新增依赖。完整声学运行要求已有 outputs/horn_detector_v1/panns-horn.pt、clips.json 及其引用的本地真实录音；缺失则明确失败，不用随机权重替代。", "",
                  "关联与声学测试使用固定的场景距离；自定义监控范围不包含这些位置时，越界导致的未匹配仍保留。需要比较配置时使用不同输出目录，防止混入其他配置的旧产物。", "",
                  "尚待验证：真实相机内外参/畸变、真实车辆检测与跟踪、遮挡与路面坡度、硬件时钟同步、连续声场/视频缓存、真实道路声学误差、角度及分数门限、摄像头 I/O、FPGA 部署。", ""])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROTO / "outputs" / "surround_v2")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20261017)
    for option, kind in [("cameras", int), ("height", float), ("pitch", float), ("hfov", float),
                         ("vfov", float), ("down-hfov", float), ("down-vfov", float),
                         ("minimum-range", float), ("maximum-range", float), ("horn-height", float),
                         ("yaw", float), ("width", int), ("image-height", int)]:
        parser.add_argument("--" + option, type=kind)
    arguments = parser.parse_args()
    profile = json.loads(arguments.profile.read_text(encoding="utf-8")) if arguments.profile else {}
    if set(profile) - {"rig", "camera_overrides", "acoustics", "association"}:
        raise ValueError("Unknown profile sections")
    values = dict(profile.get("rig", {}))
    for key in asdict(Rig()):
        value = getattr(arguments, "cameras" if key == "count" else key, None)
        if value is not None:
            values[key] = value
    rig = Rig(**values)
    cameras = rig.cameras()
    overrides = profile.get("camera_overrides", {})
    if set(overrides) - {camera.identity for camera in cameras} or any("identity" in value for value in overrides.values()):
        raise ValueError("Camera overrides must reference existing immutable identities")
    cameras = [replace(camera, **overrides.get(camera.identity, {})) for camera in cameras]
    settings = dict(candidate_count=2, threshold=0.2, margin=0.0)
    if set(profile.get("acoustics", {})) - settings.keys():
        raise ValueError("Unknown acoustic settings")
    settings.update(profile.get("acoustics", {}))
    if type(settings["candidate_count"]) is not int or settings["candidate_count"] < 1:
        raise ValueError("Candidate count must be a positive integer")
    select_bearing([], [], settings["threshold"], settings["margin"])
    association = Association(**profile.get("association", {}))
    output = arguments.output.resolve()
    config = dict(schema="surround-v2", rig=asdict(rig), cameras=[asdict(camera) for camera in cameras], acoustics=settings,
                  association=asdict(association),
                  seed=arguments.seed, geometry_only=arguments.geometry_only,
                  source_hashes={str(path.relative_to(PROTO)): digest(path) for path in
                                 [Path(__file__).resolve(), PROTO / "src" / "horn" / "surround.py"]})
    if output.exists() and any(output.iterdir()):
        previous_path = output / "configuration.json"
        if not previous_path.is_file():
            raise ValueError("Use an empty output directory or one created by this simulator")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        normalized = json.loads(json.dumps(config))
        if any(previous.get(key) != normalized[key] for key in ("schema", "rig", "cameras", "acoustics", "association", "seed", "geometry_only")):
            raise ValueError("Configuration changed; use a separate output directory to avoid stale artifacts")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "configuration.json", config)
    coverage = coverage_summary(rig, cameras)
    write_json(output / "coverage.json", coverage)
    points, visible = coverage_grid(rig, cameras=cameras)
    np.savez_compressed(output / "coverage-grid.npz", points=points, visible=visible)
    coverage_svg(output / "coverage.svg", rig, cameras)
    sweep = []
    for count in (3, 4, 6, 8):
        for height in (3, 6, 10):
            if height <= rig.horn_height:
                continue
            for pitch in (25, 40, 60):
                variant = replace(rig, count=count, height=height, pitch=pitch)
                sweep.append(dict(count=count, height=height, pitch=pitch, **coverage_summary(variant)))
    write_json(output / "coverage-sweep.json", sweep)
    visual, examples = visual_trials(rig, cameras, arguments.seed, output, association)
    acoustic = None if arguments.geometry_only else acoustic_trials(rig, cameras, settings, arguments.seed, output, association)
    report(output, config, coverage, visual, acoustic, examples)
    for path in output.glob("*.svg"):
        element_tree.parse(path)
    manifest = {path.name: digest(path) for path in output.iterdir() if path.is_file() and path.name != "artifact-manifest.json"}
    write_json(output / "artifact-manifest.json", manifest)
    print(json.dumps(dict(output=str(output), coverage=coverage, visual=visual, acoustic=acoustic), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
