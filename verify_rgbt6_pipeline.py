"""
Verify VIS/IR batch concatenation and 6-channel YOLO input.

Run from the root of your modified YOLO/Ultralytics repository, for example:

python C:/Users/youyk/Documents/Codex/2026-07-07/w/outputs/verify_rgbt6_pipeline.py ^
  --vis-dir D:/datasets/M3FD/vis/train ^
  --ir-dir D:/datasets/M3FD/ir/train ^
  --output-dir D:/datasets/M3FD_6ch/images/train ^
  --model yolov8n-6ch.yaml ^
  --target-size 640 640 ^
  --limit 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify RGBT 6-channel preprocessing and YOLO forward.")
    parser.add_argument("--vis-dir", type=Path, required=True, help="Directory containing visible/RGB images.")
    parser.add_argument("--ir-dir", type=Path, required=True, help="Directory containing infrared images.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save fused .npy/.npz files.")
    parser.add_argument("--model", default="yolov8n-6ch.yaml", help="YOLO model yaml/pt path configured for 6 channels.")
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("W", "H"), default=None, help="Resize size.")
    parser.add_argument("--limit", type=int, default=8, help="Max image pairs to verify. Use 0 for all pairs.")
    parser.add_argument("--save-mode", choices=("npy", "npz"), default="npy", help="Fused file format.")
    parser.add_argument("--prefix", default="", help="Optional prefix for fused filenames.")
    parser.add_argument("--device", default="cpu", help="Torch device for YOLO forward, e.g. cpu or cuda:0.")
    parser.add_argument("--skip-yolo", action="store_true", help="Only verify concatenation; skip YOLO forward.")
    parser.add_argument(
        "--auto-patch-first-conv",
        action="store_true",
        help="For forward smoke test only: patch the first Conv2d from 3 input channels to 6 if needed.",
    )
    return parser.parse_args()


def collect_by_stem(directory: Path) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    files = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files[path.stem] = path
    return files


def find_pairs(vis_dir: Path, ir_dir: Path, limit: int) -> tuple[list[Path], list[Path]]:
    vis = collect_by_stem(vis_dir)
    ir = collect_by_stem(ir_dir)
    common = sorted(set(vis) & set(ir))
    if not common:
        raise RuntimeError(f"No matched VIS/IR image stems found in {vis_dir} and {ir_dir}")
    if limit > 0:
        common = common[:limit]

    missing_vis = sorted(set(ir) - set(vis))[:5]
    missing_ir = sorted(set(vis) - set(ir))[:5]
    if missing_vis:
        print(f"[WARN] IR files without VIS match, first examples: {missing_vis}")
    if missing_ir:
        print(f"[WARN] VIS files without IR match, first examples: {missing_ir}")

    return [vis[s] for s in common], [ir[s] for s in common]


def load_fused(path: Path) -> np.ndarray:
    import numpy as np

    if path.suffix == ".npy":
        arr = np.load(path)
    elif path.suffix == ".npz":
        data = np.load(path)
        key = "arr_0" if "arr_0" in data.files else data.files[0]
        arr = data[key]
    else:
        raise ValueError(f"Unsupported fused file suffix: {path.suffix}")
    if arr.ndim != 3:
        raise ValueError(f"Expected fused array with shape (H, W, C), got {arr.shape} from {path}")
    if arr.shape[2] != 6:
        raise ValueError(f"Expected 6 channels, got shape {arr.shape} from {path}")
    return arr


def verify_saved_arrays(paths: list[Path]) -> list[np.ndarray]:
    arrays = []
    print("\n[1/3] Checking saved fused arrays")
    for path in paths:
        arr = load_fused(path)
        arrays.append(arr)
        print(
            f"  OK {path.name}: shape={arr.shape}, dtype={arr.dtype}, "
            f"min={arr.min()}, max={arr.max()}"
        )
    return arrays


def arrays_to_tensor(arrays: list[np.ndarray], device: str) -> torch.Tensor:
    import numpy as np
    import torch

    tensors = []
    for arr in arrays:
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).float()
        if x.max() > 1.5:
            x = x / 255.0
        tensors.append(x)
    shapes = {tuple(t.shape) for t in tensors}
    if len(shapes) != 1:
        raise ValueError(f"Fused arrays must share one shape for batching, got {sorted(shapes)}")
    return torch.stack(tensors, dim=0).to(device)


def find_first_conv(module: torch.nn.Module) -> torch.nn.Conv2d:
    import torch

    for child in module.modules():
        if isinstance(child, torch.nn.Conv2d):
            return child
    raise RuntimeError("No Conv2d layer found in YOLO model")


def patch_first_conv_to_6ch(conv: torch.nn.Conv2d) -> None:
    import torch

    if conv.in_channels == 6:
        return
    if conv.in_channels != 3:
        raise RuntimeError(f"Refusing to auto-patch first conv with in_channels={conv.in_channels}")

    new_conv = torch.nn.Conv2d(
        6,
        conv.out_channels,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.dilation,
        conv.groups,
        conv.bias is not None,
        conv.padding_mode,
    ).to(conv.weight.device)
    with torch.no_grad():
        new_conv.weight[:, :3] = conv.weight
        new_conv.weight[:, 3:] = conv.weight
        new_conv.weight.mul_(0.5)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    conv.in_channels = new_conv.in_channels
    conv.weight = new_conv.weight
    conv.bias = new_conv.bias


def verify_yolo_forward(model_path: str, arrays: list[np.ndarray], device: str, auto_patch: bool) -> None:
    import torch

    print("\n[2/3] Building YOLO model and checking first convolution")
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ultralytics. Run this script from your YOLO repository root "
            "or install your modified package first."
        ) from exc

    yolo = YOLO(model_path)
    model = yolo.model.to(device).eval()
    first_conv = find_first_conv(model)
    print(f"  First Conv2d weight shape: {tuple(first_conv.weight.shape)}")

    if first_conv.in_channels != 6:
        message = (
            f"Model first Conv2d expects {first_conv.in_channels} channels, not 6. "
            "Use a 6-channel model yaml/config, or pass --auto-patch-first-conv for a smoke test only."
        )
        if not auto_patch:
            raise RuntimeError(message)
        print(f"  [WARN] {message}")
        patch_first_conv_to_6ch(first_conv)
        print(f"  Patched first Conv2d weight shape: {tuple(first_conv.weight.shape)}")

    print("\n[3/3] Running YOLO forward with real fused .npy tensor")
    x = arrays_to_tensor(arrays, device)
    print(f"  Input tensor shape: {tuple(x.shape)}")
    with torch.no_grad():
        _ = model(x)
    print("  OK YOLO forward accepted [B, 6, H, W] input.")


def main() -> int:
    args = parse_args()
    target_size = tuple(args.target_size) if args.target_size else None

    print("[0/3] Finding matched VIS/IR pairs")
    vis_paths, ir_paths = find_pairs(args.vis_dir, args.ir_dir, args.limit)
    print(f"  Matched pairs: {len(vis_paths)}")
    for vis_path, ir_path in zip(vis_paths[:3], ir_paths[:3]):
        print(f"  pair: {vis_path.name} + {ir_path.name}")

    try:
        from ultralytics.data.concat_images import batch_concat_vis_ir
    except Exception as exc:
        raise RuntimeError(
            "Failed to import ultralytics.data.concat_images.batch_concat_vis_ir. "
            "Run this script from your modified YOLO repository root."
        ) from exc

    saved = batch_concat_vis_ir(
        vis_paths,
        ir_paths,
        args.output_dir,
        target_size=target_size,
        ir_to_3ch=True,
        save_mode=args.save_mode,
        prefix=args.prefix,
    )
    saved = [Path(p) for p in saved]
    if not saved:
        raise RuntimeError("No fused files were saved. Check warnings from batch_concat_vis_ir.")

    arrays = verify_saved_arrays(saved)
    if not args.skip_yolo:
        verify_yolo_forward(args.model, arrays, args.device, args.auto_patch_first_conv)

    print("\nDONE: RGBT 6-channel preprocessing and YOLO input verification passed.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1)
