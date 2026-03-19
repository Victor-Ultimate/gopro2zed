"""
gopro2zed: 将 GoPro 鱼眼图像转换为 ZED Mini 风格的针孔图像

支持从 JSON 标定读取源鱼眼参数，从 YAML 读取目标 ZED 参数。
"""

__version__ = "0.1.0"

from .core import (
    load_source_fisheye_json,
    load_target_zed_yaml,
    default_zed_mini_target,
    fisheye_to_target_pinhole,
)

__all__ = [
    "__version__",
    "load_source_fisheye_json",
    "load_target_zed_yaml",
    "default_zed_mini_target",
    "fisheye_to_target_pinhole",
]
