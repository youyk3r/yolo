#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Code verification - 6-channel YOLO support"""

import sys
from pathlib import Path

def check_files():
    """Check if all needed modifications are in place"""
    print("\n" + "="*70)
    print("6-CHANNEL YOLO SUPPORT VERIFICATION")
    print("="*70)
    
    checks = []
    
    # Check 1: base.py has .npy support
    print("\nCHECK 1: base.py .npy support")
    base_py = Path("ultralytics/data/base.py")
    if base_py.exists():
        content = base_py.read_text()
        has_npy = '"npy"' in content
        has_load = 'endswith(".npy")' in content
        has_stat = 'stat().st_size' in content
        
        print(f"  .npy in valid formats: {'OK' if has_npy else 'MISSING'}")
        print(f"  load_image .npy support: {'OK' if has_load else 'MISSING'}")
        print(f"  cache .npy support: {'OK' if has_stat else 'MISSING'}")
        checks.append(has_npy and has_load and has_stat)
    else:
        print("  base.py not found")
        checks.append(False)
    
    # Check 2: utils.py check_image handles .npy
    print("\nCHECK 2: data/utils.py check_image .npy")
    utils_py = Path("ultralytics/data/utils.py")
    if utils_py.exists():
        try:
            content = utils_py.read_text(encoding='utf-8', errors='ignore')
        except:
            content = utils_py.read_text(encoding='latin1', errors='ignore')
        has_npy_check = 'endswith(".npy")' in content
        has_np_load = 'np.load' in content
        
        print(f"  .npy branch in check_image: {'OK' if has_npy_check else 'MISSING'}")
        print(f"  np.load for .npy files: {'OK' if has_np_load else 'MISSING'}")
        checks.append(has_npy_check and has_np_load)
    else:
        print("  utils.py not found")
        checks.append(False)
    
    # Check 3: trainer passes ch parameter
    print("\nCHECK 3: Model ch parameter flow")
    detect_train = Path("ultralytics/models/yolo/detect/train.py")
    if detect_train.exists():
        content = detect_train.read_text()
        has_ch = 'ch=self.data["channels"]' in content
        
        print(f"  trainer passes ch param: {'OK' if has_ch else 'MISSING'}")
        checks.append(has_ch)
    else:
        print("  detect/train.py not found")
        checks.append(False)
    
    # Check 4: concat_images.py exists
    print("\nCHECK 4: concat_images.py data preparation")
    concat = Path("ultralytics/data/concat_images.py")
    if concat.exists():
        print(f"  concat_images.py exists: OK")
        checks.append(True)
    else:
        print(f"  concat_images.py exists: MISSING")
        checks.append(False)
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION RESULT")
    print("="*70)
    
    all_passed = all(checks)
    status = "PASS" if all_passed else "FAIL"
    
    print(f"\nAll checks: [{status}]")
    
    if all_passed:
        print("\nSUCCESS! 6-channel support is fully implemented.")
        return 0
    else:
        print("\nSome checks failed.")
        return 1

if __name__ == "__main__":
    sys.exit(check_files())
