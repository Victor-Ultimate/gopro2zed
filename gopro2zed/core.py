"""核心转换逻辑与标定加载"""

import json
import os

import cv2
import numpy as np
import yaml

_CUDA_AVAILABLE = None
_TORCH_AVAILABLE = None


def _torch_available():
    """检测 PyTorch 是否已安装。"""
    global _TORCH_AVAILABLE
    if _TORCH_AVAILABLE is not None:
        return _TORCH_AVAILABLE
    try:
        import torch
        _TORCH_AVAILABLE = True
        return True
    except ImportError:
        _TORCH_AVAILABLE = False
        return False


def cuda_available():
    """检测 CUDA 加速是否可用。优先使用 PyTorch（pip install torch 即可），其次 OpenCV CUDA。"""
    global _CUDA_AVAILABLE
    if _CUDA_AVAILABLE is not None:
        return _CUDA_AVAILABLE
    # 1. 优先 PyTorch CUDA（安装简单：pip install torch）
    if _torch_available():
        try:
            import torch
            if torch.cuda.is_available():
                _CUDA_AVAILABLE = True
                return True
        except Exception:
            pass
    # 2. 其次 OpenCV CUDA（需自定义编译）
    try:
        if not hasattr(cv2, "cuda"):
            _CUDA_AVAILABLE = False
            return False
        GpuMat = getattr(cv2.cuda, "GpuMat", None) or getattr(cv2, "cuda_GpuMat", None)
        if GpuMat is None:
            _CUDA_AVAILABLE = False
            return False
        gpu = GpuMat()
        gpu.upload(np.zeros((2, 2), dtype=np.uint8))
        _CUDA_AVAILABLE = True
        return True
    except Exception:
        _CUDA_AVAILABLE = False
        return False


def _remap_pytorch(image, map1, map2, interpolation):
    """使用 PyTorch grid_sample 在 GPU 上执行 remap。"""
    import torch
    import torch.nn.functional as F

    h_src, w_src = image.shape[:2]
    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]

    # (H, W, C) -> (1, C, H, W)
    img_t = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # map1=map_x(列), map2=map_y(行); 转为 PyTorch 归一化坐标 [-1,1]
    # 避免除零
    w_safe = max(w_src - 1, 1)
    h_safe = max(h_src - 1, 1)
    grid_x = (2.0 * map1.astype(np.float32) / w_safe) - 1.0
    grid_y = (2.0 * map2.astype(np.float32) / h_safe) - 1.0

    grid = np.stack([grid_x, grid_y], axis=-1)
    grid_t = torch.from_numpy(grid).unsqueeze(0).float()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_t = img_t.to(device)
    grid_t = grid_t.to(device)

    mode = "bilinear" if interpolation == cv2.INTER_LINEAR else "nearest"
    out = F.grid_sample(
        img_t,
        grid_t,
        mode=mode,
        padding_mode="zeros",
        align_corners=False,
    )

    out = (out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return out


def _remap_opencv_cuda(image, map1, map2, interpolation, border_mode):
    """使用 OpenCV CUDA 执行 remap，失败时回退到 CPU。"""
    GpuMat = getattr(cv2.cuda, "GpuMat", None) or getattr(cv2, "cuda_GpuMat", None)
    if GpuMat is None:
        return cv2.remap(image, map1, map2, interpolation=interpolation, borderMode=border_mode)
    try:
        gpu_src = GpuMat()
        gpu_src.upload(image)
        gpu_map1 = GpuMat()
        gpu_map1.upload(map1.astype(np.float32))
        gpu_map2 = GpuMat()
        gpu_map2.upload(map2.astype(np.float32))
        gpu_dst = cv2.cuda.remap(
            gpu_src, gpu_map1, gpu_map2,
            interpolation=interpolation,
            borderMode=border_mode,
        )
        return gpu_dst.download()
    except Exception:
        return cv2.remap(
            image, map1, map2,
            interpolation=interpolation,
            borderMode=border_mode,
        )


def _remap_cuda(image, map1, map2, interpolation, border_mode):
    """CUDA remap：优先 PyTorch，其次 OpenCV CUDA，失败则 CPU。"""
    if _torch_available():
        try:
            import torch
            if torch.cuda.is_available():
                return _remap_pytorch(image, map1, map2, interpolation)
        except Exception:
            pass
    # OpenCV CUDA
    if hasattr(cv2, "cuda"):
        return _remap_opencv_cuda(image, map1, map2, interpolation, border_mode)
    return cv2.remap(image, map1, map2, interpolation=interpolation, borderMode=border_mode)


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
    use_cuda=False,
):
    """将鱼眼图像转换为目标针孔视角。"""
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

    if use_cuda and cuda_available():
        out = _remap_cuda(image, map1, map2, interpolation, border_mode)
    else:
        out = cv2.remap(
            image,
            map1,
            map2,
            interpolation=interpolation,
            borderMode=border_mode,
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ok = cv2.imwrite(output_path, out)
    if not ok:
        raise IOError(f"Failed to save image to: {output_path}")

    return out
