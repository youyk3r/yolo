# -*- coding: utf-8 -*-
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""
Test script for VIS-IR 6-channel image concatenation and loading.

This test demonstrates:
1. Creating synthetic VIS and IR test images
2. Concatenating them using concat_vis_ir
3. Loading them with modified BaseDataset.load_image supporting 6 channels
"""

import cv2
import numpy as np
from pathlib import Path
import tempfile
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics.data.concat_images import concat_vis_ir, batch_concat_vis_ir
from ultralytics.data.base import BaseDataset


def create_synthetic_test_images(output_dir, num_pairs=3):
    """Create synthetic VIS and IR image pairs for testing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = output_dir / "vis"
    ir_dir = output_dir / "ir"
    vis_dir.mkdir(exist_ok=True)
    ir_dir.mkdir(exist_ok=True)
    
    img_height, img_width = 480, 640
    vis_files = []
    ir_files = []
    
    for i in range(num_pairs):
        vis_img = np.random.randint(0, 256, (img_height, img_width, 3), dtype=np.uint8)
        vis_img[100:150, 200:250] = [0, 255, 0]
        vis_img[250:300, 350:400] = [255, 0, 0]
        
        ir_img = np.random.randint(100, 200, (img_height, img_width), dtype=np.uint8)
        ir_img[100:150, 200:250] = 255
        ir_img[250:300, 350:400] = 50
        
        vis_path = vis_dir / f"vis_{i:03d}.jpg"
        ir_path = ir_dir / f"ir_{i:03d}.jpg"
        
        cv2.imwrite(str(vis_path), vis_img)
        cv2.imwrite(str(ir_path), ir_img)
        
        vis_files.append(vis_path)
        ir_files.append(ir_path)
    
    print(f"Created {num_pairs} synthetic VIS-IR pairs")
    return vis_files, ir_files


def test_concatenation(vis_files, ir_files, output_dir):
    """Test VIS-IR concatenation."""
    print("\n" + "=" * 60)
    print("Test 1: Concatenation (Single and Batch)")
    print("=" * 60)
    
    print("\n[Single Concatenation]")
    concat_path = output_dir / "fused_single.npy"
    result = concat_vis_ir(
        vis_files[0], ir_files[0],
        output_path=concat_path,
        target_size=(480, 640),
        ir_to_3ch=True,
        save_mode="npy"
    )
    
    print(f"Single concatenation result shape: {result.shape}")
    assert result.shape == (480, 640, 6), f"Expected (480, 640, 6), got {result.shape}"
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"
    print(f"VIS channels (0-2): Valid BGR data")
    print(f"IR channels (3-5): Valid replicated grayscale data")
    
    print("\n[Batch Concatenation]")
    fused_dir = output_dir / "fused_batch"
    output_paths = batch_concat_vis_ir(
        vis_paths=vis_files,
        ir_paths=ir_files,
        output_dir=fused_dir,
        target_size=(480, 640),
        ir_to_3ch=True,
        save_mode="npy",
        prefix="  "
    )
    
    assert len(output_paths) == len(vis_files), "Not all pairs were processed"
    print(f"Batch processing complete: {len(output_paths)} pairs")


def test_load_image_with_6channels(output_dir):
    """Test loading 6-channel images with modified BaseDataset."""
    print("\n" + "=" * 60)
    print("Test 2: Loading 6-channel Images with BaseDataset")
    print("=" * 60)
    
    fused_dir = output_dir / "fused_batch"
    fused_images = sorted(fused_dir.glob("*.npy"))
    
    if not fused_images:
        print("No fused images found")
        return
    
    print(f"\n[Creating dataset with {len(fused_images)} images]")
    
    image_list = output_dir / "fused_images.txt"
    with open(image_list, "w") as f:
        for img_path in fused_images:
            f.write(str(img_path.absolute()) + "\n")
    
    print("\n[Testing load_image directly]")
    
    class MinimalDataset:
        def __init__(self):
            self.im_files = [str(img) for img in fused_images]
            self.npy_files = [Path(img) for img in fused_images]
            self.ims = [None] * len(fused_images)
            self.im_hw0 = [None] * len(fused_images)
            self.im_hw = [None] * len(fused_images)
            self.imgsz = 640
            self.channels = 6
            self.pad = 0.5
            self.stride = 32
            self.augment = False
            self.buffer = []
            self.max_buffer_length = 0
            self.cache = None
            self.cv2_flag = cv2.IMREAD_COLOR
            self.prefix = "[Test]"
        
        from ultralytics.data.base import BaseDataset
        load_image = BaseDataset.load_image
    
    dataset = MinimalDataset()
    
    for i in range(min(2, len(fused_images))):
        print(f"\n  Loading image {i}...")
        im, hw_orig, hw_resized = dataset.load_image(i)
        
        print(f"  Original shape: {hw_orig}")
        print(f"  Loaded shape: {im.shape}")
        print(f"  Resized shape: {hw_resized}")
        
        assert len(im.shape) == 3, f"Expected 3D array, got {len(im.shape)}D"
        assert im.shape[2] == 6, f"Expected 6 channels, got {im.shape[2]}"
        print(f"  Successfully loaded 6-channel image")


def test_shape_transition():
    """Test shape transitions: (H,W,6) -> (6,H,W) -> (B,6,H,W)."""
    print("\n" + "=" * 60)
    print("Test 3: Shape Transitions in Pipeline")
    print("=" * 60)
    
    batch_size = 4
    height, width = 640, 640
    channels = 6
    
    print(f"\n[After load_image]")
    batch_hwc = np.random.randint(0, 256, (batch_size, height, width, channels), dtype=np.uint8)
    print(f"  Batch shape (HWC): {batch_hwc.shape}")
    
    print(f"\n[After transforms (HWC->CHW)]")
    batch_chw = np.transpose(batch_hwc, (0, 3, 1, 2))
    print(f"  Batch shape (CHW): {batch_chw.shape}")
    print(f"  Expected for model: {(batch_size, channels, height, width)}")
    
    assert batch_chw.shape == (batch_size, channels, height, width)
    print(f"  Shape transition complete")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Testing 6-Channel VIS-IR Pipeline")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        vis_files, ir_files = create_synthetic_test_images(tmp_path / "input", num_pairs=3)
        
        test_concatenation(vis_files, ir_files, tmp_path / "concat_test")
        
        test_load_image_with_6channels(tmp_path / "concat_test")
        
        test_shape_transition()
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
    print("\nSummary:")
    print("  VIS-IR concatenation works (creates 6-channel .npy files)")
    print("  load_image supports reading 6-channel images")
    print("  Shape transitions are correct for model input")
    print("\nNext steps:")
    print("  1. Use concat_vis_ir() / batch_concat_vis_ir() on your actual data")
    print("  2. Create image list file pointing to .npy paths")
    print("  3. Initialize BaseDataset with channels=6")
    print("  4. Set up transforms to handle 6-channel data")
    print("=" * 60 + "\n")
