"""核心转换逻辑与标定加载"""

import json
import os

import cv2
import numpy as np
import yaml


def load_source_fisheye_json(json_path, aspect_mode="fy_over_fx"):
    """
    读取源鱼眼相机参数（OpenCV fisheye 模型）
    JSON 结构示例：
    {
      "image_width": 2704,
      "image_height": 2028,
      "intrinsic_type": "FISHEYE",
      "intrinsics": {
        "aspect_ratio": 1.0029788958491257,
        "focal_length": 796.8544625226342,
        "principal_pt_x": 1354.4265245977356,
        "principal_pt_y": 1011.4847310011687,
        "radial_distortion_1": -0.02196117964405394,
        "radial_distortion_2": -0.018959717016668237,
        "radial_distortion_3": 0.001693880829392453,
        "radial_distortion_4": -0.00016807228608000285
      }
    }

    aspect_mode:
      - "fy_over_fx": fy = fx * aspect_ratio
      - "fx_over_fy": fx = fy * aspect_ratio
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intrinsic_type = data.get("intrinsic_type", "").upper()
    if intrinsic_type != "FISHEYE":
        raise ValueError(f"Expected intrinsic_type='FISHEYE', got '{intrinsic_type}'")

    intr = data["intrinsics"]
    width = int(data["image_width"])
    height = int(data["image_height"])

    focal_length = float(intr["focal_length"])
    aspect_ratio = float(intr.get("aspect_ratio", 1.0))
    cx = float(intr["principal_pt_x"])
    cy = float(intr["principal_pt_y"])
    k1 = float(intr["radial_distortion_1"])
    k2 = float(intr["radial_distortion_2"])
    k3 = float(intr["radial_distortion_3"])
    k4 = float(intr["radial_distortion_4"])
    skew = float(intr.get("skew", 0.0))

    if aspect_mode == "fy_over_fx":
        fx = focal_length
        fy = focal_length * aspect_ratio
    elif aspect_mode == "fx_over_fy":
        fy = focal_length
        fx = focal_length * aspect_ratio
    else:
        raise ValueError(f"Unsupported aspect_mode: {aspect_mode}")

    K = np.array([
        [fx, skew, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)

    return {
        "K": K,
        "D": D,
        "resolution": (width, height),
        "raw": data,
    }


def load_target_zed_yaml(yaml_path, camera_key="left"):
    """
    读取目标 ZED calibration
    支持之前导出的格式，例如：

    left:
      camera_model: pinhole
      distortion_model: radtan
      intrinsics: [fx, fy, cx, cy]
      distortion_coeffs: [k1, k2, p1, p2, k3]
      resolution: [1280, 720]
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if camera_key not in data:
        raise KeyError(f"Cannot find '{camera_key}' in {yaml_path}")

    cam = data[camera_key]

    intrinsics = cam["intrinsics"]
    resolution = cam["resolution"]

    if len(intrinsics) < 4:
        raise ValueError("Target intrinsics must be [fx, fy, cx, cy]")

    fx, fy, cx, cy = [float(v) for v in intrinsics[:4]]
    width, height = [int(v) for v in resolution]

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return {
        "K": K,
        "resolution": (width, height),
        "raw": data,
    }


def default_zed_mini_target():
    """
    一个常见的 ZED Mini rectified 近似目标。
    更准确时请改用真实 ZED YAML。
    """
    width, height = 1280, 720
    fx, fy = 736.0, 528.0
    cx, cy = 640.0, 360.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)

    return {
        "K": K,
        "resolution": (width, height),
        "raw": {
            "name": "default_zed_mini_rectified"
        }
    }


def fisheye_to_target_pinhole(
    image_path,
    output_path,
    source_K,
    source_D,
    target_K,
    target_size,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_CONSTANT,
    use_fast=False,
):
    """将鱼眼图像转换为目标针孔视角。

    use_fast: 使用 INTER_NEAREST 插值，速度更快但画质略降。
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    R = np.eye(3, dtype=np.float64)

    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K=source_K,
        D=source_D,
        R=R,
        P=target_K,
        size=target_size,
        m1type=cv2.CV_32FC1,
    )

    interp = cv2.INTER_NEAREST if use_fast else interpolation
    out = cv2.remap(
        image,
        map1,
        map2,
        interpolation=interp,
        borderMode=border_mode,
    )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ok = cv2.imwrite(output_path, out)
    if not ok:
        raise IOError(f"Failed to save image to: {output_path}")

    return out
