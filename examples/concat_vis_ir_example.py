# Ultralytics ? AGPL-3.0 License - https://ultralytics.com/license

"""
Example script: How to concatenate VIS and IR images for multi-modal dataset.

This script demonstrates:
1. Single pair concatenation
2. Batch processing of multiple pairs
3. Creating an image list file for BaseDataset
"""

from pathlib import Path
from ultralytics.data.concat_images import concat_vis_ir, batch_concat_vis_ir


def example_single_concat():
    """Example: Concatenate a single VIS-IR pair."""
    print("=" * 60)
    print("Example 1: Single VIS-IR Concatenation")
    print("=" * 60)

    # Paths (replace with your actual paths)
    vis_file = "path/to/vis_image.jpg"  # Visible spectrum image
    ir_file = "path/to/ir_image.jpg"    # Infrared image
    output_file = "path/to/output/fused.npy"

    # Concatenate and save
    concatenated = concat_vis_ir(
        vis_path=vis_file,
        ir_path=ir_file,
        output_path=output_file,
        target_size=(640, 640),  # Optionally resize to fixed size
        ir_to_3ch=True,           # Replicate single-channel IR to 3 channels
        save_mode="npy"           # Save as .npy (binary numpy format)
    )

    print(f"? Concatenated shape: {concatenated.shape}")
    print(f"  Expected: (H, W, 6) where first 3 channels are VIS (BGR)")
    print(f"  and last 3 channels are IR (replicated from single channel)")
    print()


def example_batch_concat():
    """Example: Batch process multiple VIS-IR pairs."""
    print("=" * 60)
    print("Example 2: Batch VIS-IR Concatenation")
    print("=" * 60)

    # Prepare lists of VIS and IR file paths
    # You can generate these from directory scanning or text files
    vis_list = [
        "dataset/vis/image_001.jpg",
        "dataset/vis/image_002.jpg",
        "dataset/vis/image_003.jpg",
    ]
    
    ir_list = [
        "dataset/ir/thermal_001.jpg",
        "dataset/ir/thermal_002.jpg",
        "dataset/ir/thermal_003.jpg",
    ]

    output_dir = "dataset/fused_6ch"

    # Batch process
    output_paths = batch_concat_vis_ir(
        vis_paths=vis_list,
        ir_paths=ir_list,
        output_dir=output_dir,
        target_size=(640, 640),
        ir_to_3ch=True,
        save_mode="npy",
        prefix="[Batch] "
    )

    print(f"? Processed {len(output_paths)} pairs")
    print(f"  Output directory: {output_dir}")
    print()


def example_dataset_integration():
    """Example: How to use concatenated images with BaseDataset."""
    print("=" * 60)
    print("Example 3: Integration with BaseDataset")
    print("=" * 60)

    # Step 1: Create concatenated images (as shown above)
    # Step 2: Create an image list file pointing to concatenated .npy files
    
    output_dir = "dataset/fused_6ch"
    image_list_file = "dataset/fused_images.txt"
    
    # List all .npy files in output directory
    npy_files = sorted(Path(output_dir).glob("*.npy"))

    # Write to file (one path per line)
    with open(image_list_file, "w") as f:
        for npy_file in npy_files:
            f.write(str(npy_file.absolute()) + "\n")

    print(f"? Created image list file: {image_list_file}")
    print(f"  Total concatenated images: {len(npy_files)}")
    print()

    # Step 3: Use with BaseDataset
    print("Usage with BaseDataset:")
    print("-" * 60)
    print("""
    from ultralytics.data.base import BaseDataset
    
    # Load dataset using the image list
    dataset = BaseDataset(
        img_path="dataset/fused_images.txt",  # Point to .txt file with .npy paths
        imgsz=640,
        cache=False,
        augment=True,
        channels=6,  # IMPORTANT: Set to 6 for 6-channel images!
        ...
    )
    
    # load_image will now:
    # 1. First check if img_file ends with .npy
    # 2. If so, load with np.load() -> shape (H, W, 6)
    # 3. Verify channels match self.channels
    # 4. Return the 6-channel image
    """)
    print()


def example_dataset_from_directory():
    """Example: Auto-discovery from directory pairs."""
    print("=" * 60)
    print("Example 4: Auto-detect VIS-IR pairs from directories")
    print("=" * 60)

    from pathlib import Path

    # Assume directory structure:
    # dataset/
    #   ©À©¤©¤ vis/
    #   ©¦   ©À©¤©¤ frame_001.jpg
    #   ©¦   ©À©¤©¤ frame_002.jpg
    #   ©¦   ©¸©¤©¤ ...
    #   ©¸©¤©¤ ir/
    #       ©À©¤©¤ frame_001.jpg
    #       ©À©¤©¤ frame_002.jpg
    #       ©¸©¤©¤ ...

    vis_dir = Path("dataset/vis")
    ir_dir = Path("dataset/ir")
    output_dir = Path("dataset/fused_6ch")

    # Collect matched pairs
    vis_files = sorted(vis_dir.glob("*.jpg"))
    vis_stems = [f.stem for f in vis_files]
    
    ir_files = sorted(ir_dir.glob("*.jpg"))
    ir_dict = {f.stem: f for f in ir_files}

    # Match and concatenate
    matched_pairs = []
    for vis_file, vis_stem in zip(vis_files, vis_stems):
        if vis_stem in ir_dict:
            ir_file = ir_dict[vis_stem]
            matched_pairs.append((vis_file, ir_file))

    if matched_pairs:
        print(f"? Found {len(matched_pairs)} matched VIS-IR pairs")
        
        # Batch process
        output_paths = batch_concat_vis_ir(
            vis_paths=[p[0] for p in matched_pairs],
            ir_paths=[p[1] for p in matched_pairs],
            output_dir=output_dir,
            target_size=(640, 640),
            ir_to_3ch=True,
            save_mode="npy",
            prefix="[Auto] "
        )
    else:
        print("? No matched VIS-IR pairs found")


if __name__ == "__main__":
    print("\n")
    print("¨X" + "=" * 58 + "¨[")
    print("¨U" + " VIS-IR Concatenation Examples ".center(58) + "¨U")
    print("¨^" + "=" * 58 + "¨a")
    print()

    # Note: These are demonstration examples.
    # Uncomment and modify paths to run with your own data.

    # example_single_concat()
    example_batch_concat()  # Placeholder - will fail if paths don't exist
    # example_dataset_integration()
    # example_dataset_from_directory()

    print("\n" + "=" * 60)
    print("For more information, see concat_images.py docstrings")
    print("=" * 60 + "\n")
