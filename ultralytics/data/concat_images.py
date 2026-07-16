# Ultralytics ? AGPL-3.0 License - https://ultralytics.com/license

"""Utilities for concatenating VIS (visible) and IR (infrared) images into multi-channel format."""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple

from ultralytics.utils import LOGGER
from ultralytics.utils.patches import imread


def concat_vis_ir(
    vis_path: str | Path,
    ir_path: str | Path,
    output_path: str | Path | None = None,
    target_size: Tuple[int, int] | None = None,
    ir_to_3ch: bool = True,
    save_mode: str = "npy",
) -> np.ndarray:
    """Concatenate VIS (visible) and IR (infrared) images into a single multi-channel array.

    This function reads a visible-spectrum image and an infrared image, aligns them to the same
    dimensions, and concatenates them along the channel axis to create a 6-channel image
    (3-channel VIS + 3-channel IR) or other combinations depending on input.

    Args:
        vis_path (str | Path): Path to the visible-spectrum image.
        ir_path (str | Path): Path to the infrared image.
        output_path (str | Path | None): Path to save the concatenated image. If None, only returns
            the array without saving. Default: None.
        target_size (Tuple[int, int] | None): Target (height, width) for resizing both images.
            If None, matches IR size to VIS size. Default: None.
        ir_to_3ch (bool): If True and IR is single-channel, replicate it to 3 channels.
            If False, keep IR as single channel (result will be 4-channel). Default: True.
        save_mode (str): Format to save: 'npy' for numpy binary, 'npz' for compressed. Default: 'npy'.

    Returns:
        (np.ndarray): Concatenated image as (H, W, C) where C is typically 6 (vis 3ch + ir 3ch)
            or 4 (vis 3ch + ir 1ch if ir_to_3ch=False).

    Raises:
        FileNotFoundError: If vis_path or ir_path does not exist.
        ValueError: If images cannot be read or concatenated.

    Examples:
        >>> # Concatenate and save to .npy
        >>> concat_vis_ir("visible.jpg", "thermal.jpg", "fused.npy")
        >>> # Concatenate multiple pairs in batch
        >>> for vis, ir in zip(vis_files, ir_files):
        ...     concat_vis_ir(vis, ir, f"{output_dir}/{Path(vis).stem}.npy")
    """
    vis_path = Path(vis_path)
    ir_path = Path(ir_path)

    if not vis_path.exists():
        raise FileNotFoundError(f"VIS image not found: {vis_path}")
    if not ir_path.exists():
        raise FileNotFoundError(f"IR image not found: {ir_path}")

    # Read images
    vis = imread(str(vis_path), flags=cv2.IMREAD_COLOR)  # BGR, shape (H, W, 3)
    ir = imread(str(ir_path), flags=cv2.IMREAD_GRAYSCALE)  # grayscale, shape (H, W)

    if vis is None:
        raise ValueError(f"Failed to read VIS image: {vis_path}")
    if ir is None:
        raise ValueError(f"Failed to read IR image: {ir_path}")

    # Determine target size
    if target_size is None:
        target_size = (vis.shape[0], vis.shape[1])  # Use VIS size as reference

    # Resize if necessary
    if vis.shape[:2] != target_size:
        vis = cv2.resize(vis, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)
    if ir.shape[:2] != target_size:
        ir = cv2.resize(ir, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    # Ensure IR is 3-channel if required
    if ir_to_3ch:
        if ir.ndim == 2:
            ir = np.stack([ir, ir, ir], axis=2)  # Replicate single channel to 3 channels
    else:
        # Keep IR as 1-channel but ensure it's (H, W, 1) format for concatenation
        if ir.ndim == 2:
            ir = ir[..., np.newaxis]

    # Concatenate: VIS (H,W,3) + IR (H,W,3 or H,W,1) -> (H,W,6 or H,W,4)
    concatenated = np.concatenate([vis, ir], axis=2)

    # Save if output_path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if save_mode.lower() == "npy":
                np.save(output_path.with_suffix(".npy"), concatenated, allow_pickle=False)
            elif save_mode.lower() == "npz":
                np.savez_compressed(output_path.with_suffix(".npz"), data=concatenated)
            else:
                raise ValueError(f"Unsupported save_mode: {save_mode}")
            LOGGER.info(f"Concatenated image saved: {output_path}")
        except Exception as e:
            LOGGER.warning(f"Failed to save concatenated image to {output_path}: {e}")

    return concatenated


def batch_concat_vis_ir(
    vis_paths: list[str | Path],
    ir_paths: list[str | Path],
    output_dir: str | Path,
    target_size: Tuple[int, int] | None = None,
    ir_to_3ch: bool = True,
    save_mode: str = "npy",
    name_suffix: str = "_fused",
    prefix: str = "",
) -> list[Path]:
    """Batch process multiple VIS-IR image pairs.

    Args:
        vis_paths (list[str | Path]): List of paths to VIS images.
        ir_paths (list[str | Path]): List of paths to IR images (must match length of vis_paths).
        output_dir (str | Path): Directory to save concatenated images.
        target_size (Tuple[int, int] | None): Target (height, width). Default: None.
        ir_to_3ch (bool): Replicate single-channel IR to 3 channels. Default: True.
        save_mode (str): Format to save ('npy' or 'npz'). Default: 'npy'.
        name_suffix (str): Suffix appended to each VIS stem before the file extension. Use an empty string to preserve
            source stems, e.g. '00000.npy' for direct YOLO label matching. Default: '_fused'.
        prefix (str): Prefix for logging. Default: "".

    Returns:
        (list[Path]): List of saved output paths.

    Raises:
        ValueError: If lengths of vis_paths and ir_paths don't match.
    """
    if len(vis_paths) != len(ir_paths):
        raise ValueError(f"VIS and IR path lists must have same length: {len(vis_paths)} vs {len(ir_paths)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for i, (vis_path, ir_path) in enumerate(zip(vis_paths, ir_paths)):
        try:
            vis_path = Path(vis_path)
            output_path = output_dir / f"{vis_path.stem}{name_suffix}.{save_mode.lower()}"
            concat_vis_ir(vis_path, ir_path, output_path, target_size, ir_to_3ch, save_mode)
            output_paths.append(output_path)
        except Exception as e:
            LOGGER.warning(f"{prefix}Failed to process pair {i} ({vis_path}, {ir_path}): {e}")

    LOGGER.info(f"{prefix}Batch processing complete: {len(output_paths)}/{len(vis_paths)} pairs processed")
    return output_paths
