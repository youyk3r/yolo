# -*- coding: utf-8 -*-

"""
Standalone demo: VIS-IR 6-channel concatenation.
This does not require full ultralytics import (avoids torch DLL issues).
"""

import cv2
import numpy as np
from pathlib import Path
import tempfile


def concat_vis_ir_demo(vis_path, ir_path, output_path, target_size=(480, 640)):
    """Simple standalone concatenation demo."""
    vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
    ir = cv2.imread(str(ir_path), cv2.IMREAD_GRAYSCALE)
    
    if vis is None or ir is None:
        raise ValueError("Failed to read images")
    
    if vis.shape[:2] != target_size:
        vis = cv2.resize(vis, (target_size[1], target_size[0]))
    if ir.shape[:2] != target_size:
        ir = cv2.resize(ir, (target_size[1], target_size[0]))
    
    ir_3ch = np.stack([ir, ir, ir], axis=2)
    concatenated = np.concatenate([vis, ir_3ch], axis=2)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), concatenated, allow_pickle=False)
    
    return concatenated


def main():
    print("\n" + "=" * 60)
    print("VIS-IR 6-Channel Concatenation Demo")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        print("\n[Step 1: Creating synthetic test images]")
        vis_dir = tmp_path / "vis"
        ir_dir = tmp_path / "ir"
        vis_dir.mkdir(exist_ok=True)
        ir_dir.mkdir(exist_ok=True)
        
        height, width = 480, 640
        
        vis_img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        vis_img[100:150, 200:250] = [0, 255, 0]
        
        ir_img = np.random.randint(100, 200, (height, width), dtype=np.uint8)
        ir_img[100:150, 200:250] = 255
        
        vis_path = vis_dir / "test_vis.jpg"
        ir_path = ir_dir / "test_ir.jpg"
        
        cv2.imwrite(str(vis_path), vis_img)
        cv2.imwrite(str(ir_path), ir_img)
        print(f"Created synthetic images: {vis_path.name}, {ir_path.name}")
        
        print("\n[Step 2: Concatenating VIS and IR]")
        output_path = tmp_path / "fused.npy"
        result = concat_vis_ir_demo(vis_path, ir_path, output_path)
        print(f"Concatenation result shape: {result.shape}")
        print(f"Expected: (480, 640, 6)")
        assert result.shape == (480, 640, 6), "Shape mismatch!"
        print("Shape check: PASSED")
        
        print("\n[Step 3: Verifying saved file]")
        loaded = np.load(output_path)
        print(f"Loaded shape: {loaded.shape}")
        print(f"Matches saved: {np.array_equal(result, loaded)}")
        assert np.array_equal(result, loaded), "Data mismatch!"
        print("Data integrity: PASSED")
        
        print("\n[Step 4: Understanding the channel layout]")
        print(f"Channels 0-2 (VIS BGR): shape {loaded[:, :, 0:3].shape}")
        print(f"Channels 3-5 (IR 3ch):  shape {loaded[:, :, 3:6].shape}")
        print(f"VIS sample pixel [0,0]: {loaded[0, 0, 0:3]}")
        print(f"IR  sample pixel [0,0]: {loaded[0, 0, 3:6]}")
        
        print("\n[Step 5: Shape transformation for model]")
        batch_hwc = np.expand_dims(loaded, 0)
        print(f"After adding batch dim (HWC): {batch_hwc.shape}")
        
        batch_chw = np.transpose(batch_hwc, (0, 3, 1, 2))
        print(f"After transpose to CHW: {batch_chw.shape}")
        print(f"Model expects: (B, C, H, W) = (1, 6, 480, 640)")
        assert batch_chw.shape == (1, 6, 480, 640), "Model input shape mismatch!"
        print("Shape transformation: PASSED")
        
    print("\n" + "=" * 60)
    print("All checks passed successfully!")
    print("=" * 60)
    print("\nUsage in your code:")
    print("  1. For each VIS-IR pair, call concat_vis_ir(vis_path, ir_path, out.npy)")
    print("  2. List all .npy paths in a text file")
    print("  3. Pass that file to BaseDataset with channels=6")
    print("  4. Ensure your model can accept 6-channel input")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
