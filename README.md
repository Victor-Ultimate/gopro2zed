# gopro2zed

将 GoPro 鱼眼图像转换为 ZED Mini 风格的针孔（rectified）图像。

## 安装

```bash
# 从源码安装（在 gopro2zed 目录下）
cd gopro2zed
pip install .
```

或从父目录安装：

```bash
pip install ./gopro2zed
```

## 用法

### 命令行

```bash
gopro2zed --image input.jpg --source-json calibration.json --output output.png
```

可选参数：

- `--target-zed-yaml`：目标 ZED 标定 YAML，不指定则使用内置 ZED Mini 默认参数
- `--target-camera-key`：YAML 中的相机 key，默认 `left`
- `--aspect-mode`：`fy_over_fx` 或 `fx_over_fy`，默认 `fy_over_fx`
- `--cuda`：启用 CUDA GPU 加速（使用 PyTorch，`pip install torch` 即可）

示例（使用自定义 ZED 标定）：

```bash
gopro2zed --image test.jpg --source-json gopro_calib.json \
  --target-zed-yaml zed_calibration.yaml --target-camera-key left \
  --output out.png
```

### Python API

```python
from gopro2zed import (
    load_source_fisheye_json,
    load_target_zed_yaml,
    default_zed_mini_target,
    fisheye_to_target_pinhole,
)

# 加载标定
source = load_source_fisheye_json("gopro_calib.json")
target = load_target_zed_yaml("zed_calibration.yaml", camera_key="left")
# 或使用默认: target = default_zed_mini_target()

# 转换（可选 use_cuda=True 启用 GPU 加速）
fisheye_to_target_pinhole(
    image_path="input.jpg",
    output_path="output.png",
    source_K=source["K"],
    source_D=source["D"],
    target_K=target["K"],
    target_size=target["resolution"],
    use_cuda=True,  # 需 pip install torch（自动检测 CUDA）
)
```

## 依赖

- numpy
- opencv-python
- PyYAML

**CUDA 加速**（可选）：`--cuda` 使用 PyTorch 的 `grid_sample` 在 GPU 上加速，安装简单：

```bash
pip install torch
# 或安装包时一并安装：pip install gopro2zed[cuda]
```

若未安装 PyTorch 或未检测到 CUDA，将自动回退到 CPU。
