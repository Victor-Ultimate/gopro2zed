"""命令行入口"""

import argparse
import time

from .core import (
    load_source_fisheye_json,
    load_target_zed_yaml,
    default_zed_mini_target,
    fisheye_to_target_pinhole,
    cuda_available,
    warmup_cuda,
)


def main():
    parser = argparse.ArgumentParser(
        description="Convert fisheye image (from JSON calibration) to ZED Mini-like pinhole image."
    )
    parser.add_argument("--image", required=True, help="Input fisheye image")
    parser.add_argument("--source-json", required=True, help="Source fisheye calibration JSON")
    parser.add_argument("--output", required=True, help="Output image path")

    parser.add_argument(
        "--target-zed-yaml",
        default=None,
        help="Optional target ZED calibration YAML; if omitted, use built-in default ZED Mini parameters"
    )
    parser.add_argument(
        "--target-camera-key",
        default="left",
        help="Camera key inside target ZED YAML, default: left"
    )

    parser.add_argument(
        "--aspect-mode",
        choices=["fy_over_fx", "fx_over_fy"],
        default="fy_over_fx",
        help="Interpretation of aspect_ratio in source JSON"
    )
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA GPU acceleration via PyTorch (pip install torch)",
    )

    args = parser.parse_args()

    source = load_source_fisheye_json(args.source_json, aspect_mode=args.aspect_mode)

    if args.target_zed_yaml:
        target = load_target_zed_yaml(args.target_zed_yaml, camera_key=args.target_camera_key)
        target_name = f"yaml:{args.target_zed_yaml}:{args.target_camera_key}"
    else:
        target = default_zed_mini_target()
        target_name = "default_zed_mini_target"

    if args.cuda and not cuda_available():
        print("Warning: --cuda requested but CUDA not available, falling back to CPU. "
              "Install PyTorch with CUDA: pip install torch (or pip install gopro2zed[cuda])")
    elif args.cuda:
        warmup_cuda()  # 提前加载 torch/CUDA，不计入转换耗时

    start_time = time.time()
    fisheye_to_target_pinhole(
        image_path=args.image,
        output_path=args.output,
        source_K=source["K"],
        source_D=source["D"],
        target_K=target["K"],
        target_size=target["resolution"],
        use_cuda=args.cuda,
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("Done.")
    print("\n[Source Fisheye]")
    print("resolution:", source["resolution"])
    print("K:\n", source["K"])
    print("D:\n", source["D"].reshape(-1))

    print("\n[Target ZED-like Pinhole]")
    print("target:", target_name)
    print("resolution:", target["resolution"])
    print("K:\n", target["K"])

    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
