"""Prepare a 6-channel M3FD detection dataset for YOLO training.

This script reads the original M3FD_Detection layout:

    M3FD_Detection/
      vi/
      ir/
      labels/

and writes a YOLO dataset with VIS+IR 6-channel .npy images:

    datasets/M3FD_6ch/
      rgbt6.yaml
      images/train/*.npy
      images/val/*.npy
      labels/train/*.txt
      labels/val/*.txt

Labels keep the same stem as images, e.g. images/train/00000.npy maps to
labels/train/00000.txt.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from ultralytics.data.concat_images import batch_concat_vis_ir

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_KEEP_CLASSES = [0, 1]
DEFAULT_NAMES = ["person", "car"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert M3FD VIS/IR pairs to a YOLO 6-channel .npy dataset.")
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path(r"E:\BaiduNetdiskDownload\M3FD_Detection"),
        help="Original M3FD_Detection root containing vi/, ir/, and labels/.",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=Path("datasets/M3FD_6ch"),
        help="Output dataset root. Relative paths are resolved from the current working directory.",
    )
    parser.add_argument("--target-size", type=int, nargs=2, metavar=("W", "H"), default=(640, 640), help="Resize size.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random split seed.")
    parser.add_argument("--limit", type=int, default=0, help="Limit converted pairs for smoke testing. 0 means all.")
    parser.add_argument("--overwrite", action="store_true", help="Delete an existing output dataset before writing.")
    parser.add_argument("--save-mode", choices=("npy", "npz"), default="npy", help="Output array format.")
    parser.add_argument(
        "--keep-classes",
        type=int,
        nargs="+",
        default=DEFAULT_KEEP_CLASSES,
        help="Original class IDs to keep. Kept classes are remapped to 0..N-1 in the output labels.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        default=DEFAULT_NAMES,
        help="Output class names matching --keep-classes order.",
    )
    return parser.parse_args()


def collect_files(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    return {p.stem: p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in suffixes}


def parse_label(label_path: Path) -> list[list[float]]:
    rows = []
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_no} expected 5 YOLO columns, got {len(parts)}")
        rows.append([float(x) for x in parts])
    return rows


def write_filtered_label(src_label: Path, dst_label: Path, class_map: dict[int, int]) -> tuple[int, int]:
    kept = []
    dropped = 0
    for row in parse_label(src_label):
        original_cls = int(row[0])
        if original_cls not in class_map:
            dropped += 1
            continue
        row[0] = class_map[original_cls]
        kept.append(row)

    dst_label.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(f"{int(row[0])} {row[1]:.16g} {row[2]:.16g} {row[3]:.16g} {row[4]:.16g}" for row in kept)
    dst_label.write_text(text + ("\n" if text else ""), encoding="utf-8")
    return len(kept), dropped


def find_matched_stems(src_root: Path) -> tuple[list[str], dict[str, Path], dict[str, Path], dict[str, Path]]:
    vi_files = collect_files(src_root / "vi", IMAGE_EXTS)
    ir_files = collect_files(src_root / "ir", IMAGE_EXTS)
    label_files = collect_files(src_root / "labels", {".txt"})
    stems = sorted(set(vi_files) & set(ir_files) & set(label_files))

    if not stems:
        raise RuntimeError(f"No matched vi/ir/labels stems found under {src_root}")

    missing_ir = sorted(set(vi_files) - set(ir_files))[:5]
    missing_vi = sorted(set(ir_files) - set(vi_files))[:5]
    missing_label = sorted((set(vi_files) & set(ir_files)) - set(label_files))[:5]
    if missing_ir:
        print(f"[WARN] VIS files without IR match, examples: {missing_ir}")
    if missing_vi:
        print(f"[WARN] IR files without VIS match, examples: {missing_vi}")
    if missing_label:
        print(f"[WARN] VIS/IR pairs without label, examples: {missing_label}")

    return stems, vi_files, ir_files, label_files


def split_stems(stems: list[str], train_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    stems = stems.copy()
    random.Random(seed).shuffle(stems)
    n_train = max(1, min(len(stems) - 1, round(len(stems) * train_ratio))) if len(stems) > 1 else len(stems)
    return sorted(stems[:n_train]), sorted(stems[n_train:])


def reset_output(dst_root: Path, overwrite: bool) -> None:
    if dst_root.exists() and any(dst_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {dst_root}. Use --overwrite to replace it.")
        shutil.rmtree(dst_root)
    for split in ("train", "val"):
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_yaml(dst_root: Path, names: list[str]) -> None:
    yaml_path = dst_root / "rgbt6.yaml"
    names_yaml = "\n".join(f"  {i}: {name}" for i, name in enumerate(names))
    content = (
        f"path: {dst_root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "channels: 6\n"
        f"nc: {len(names)}\n\n"
        "names:\n"
        f"{names_yaml}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def convert_split(
    split: str,
    stems: list[str],
    vi_files: dict[str, Path],
    ir_files: dict[str, Path],
    label_files: dict[str, Path],
    dst_root: Path,
    target_size: tuple[int, int],
    save_mode: str,
    class_map: dict[int, int],
) -> tuple[int, int]:
    image_dir = dst_root / "images" / split
    label_dir = dst_root / "labels" / split
    print(f"[{split}] converting {len(stems)} pairs...")

    saved = batch_concat_vis_ir(
        [vi_files[s] for s in stems],
        [ir_files[s] for s in stems],
        image_dir,
        target_size=(target_size[1], target_size[0]),  # concat_vis_ir expects (H, W)
        ir_to_3ch=True,
        save_mode=save_mode,
        name_suffix="",
        prefix=f"{split}: ",
    )
    if len(saved) != len(stems):
        raise RuntimeError(f"{split}: converted {len(saved)}/{len(stems)} pairs")

    kept_total = dropped_total = 0
    for stem in stems:
        kept, dropped = write_filtered_label(label_files[stem], label_dir / f"{stem}.txt", class_map)
        kept_total += kept
        dropped_total += dropped
    print(f"[{split}] labels kept/dropped: {kept_total}/{dropped_total}")
    return kept_total, dropped_total


def main() -> int:
    args = parse_args()
    src_root = args.src_root.resolve()
    dst_root = args.dst_root.resolve()
    target_size = tuple(args.target_size)
    if len(args.keep_classes) != len(args.names):
        raise ValueError("--keep-classes and --names must have the same length")
    class_map = {source_cls: idx for idx, source_cls in enumerate(args.keep_classes)}

    print(f"Source: {src_root}")
    print(f"Output: {dst_root}")
    print(f"Keeping classes: {class_map} -> {args.names}")

    stems, vi_files, ir_files, label_files = find_matched_stems(src_root)
    if args.limit > 0:
        stems = stems[: args.limit]
        print(f"[INFO] limit enabled: using first {len(stems)} matched pairs")

    train_stems, val_stems = split_stems(stems, args.train_ratio, args.seed)
    print(f"Matched pairs: {len(stems)}")
    print(f"Train/val: {len(train_stems)}/{len(val_stems)}")

    reset_output(dst_root, args.overwrite)
    train_kept, train_dropped = convert_split(
        "train", train_stems, vi_files, ir_files, label_files, dst_root, target_size, args.save_mode, class_map
    )
    val_kept, val_dropped = convert_split(
        "val", val_stems, vi_files, ir_files, label_files, dst_root, target_size, args.save_mode, class_map
    )
    write_yaml(dst_root, args.names)

    print("\nDONE")
    print(f"Total labels kept/dropped: {train_kept + val_kept}/{train_dropped + val_dropped}")
    print(f"Dataset YAML: {dst_root / 'rgbt6.yaml'}")
    print("Next smoke train command:")
    print(
        f"  yolo detect train model=yolov8n-6ch.yaml data={dst_root / 'rgbt6.yaml'} imgsz=640 batch=2 epochs=1 workers=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
