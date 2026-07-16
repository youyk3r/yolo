# VIS-IR 6通道图像拼接方案 (离线预处理)

## 概述

本方案实现了将可见光(VIS)和红外(IR)图像拼接成6通道格式 `[B, 6, H, W]` 的完整流程。采用**离线预处理**策略，即先将VIS和IR图像配对拼接并保存为 *.npy 文件，然后通过修改 `BaseDataset.load_image` 来加载这些预处理好的图像。

## 核心改动

### 1. 新增模块: `ultralytics/data/concat_images.py`

实现了两个主要函数:

#### `concat_vis_ir(vis_path, ir_path, output_path, target_size, ir_to_3ch, save_mode)`

单个VIS-IR对的拼接函数。

**参数:**
- `vis_path`: 可见光图像路径 (BGR, 3通道)
- `ir_path`: 红外图像路径 (通常为单通道灰度)
- `output_path`: 输出 *.npy 文件路径 (可选)
- `target_size`: 目标 (高, 宽) 尺寸 (默认用VIS尺寸)
- `ir_to_3ch`: 是否将单通道IR复制为3通道 (默认True)
- `save_mode`: 保存格式 ('npy' 或 'npz', 默认'npy')

**返回:**
- `np.ndarray`: 拼接后的图像，形状为 `(H, W, 6)` (HWC格式)
  - 通道 0-2: VIS (BGR)
  - 通道 3-5: IR (复制的灰度值或原始单通道)

**示例:**
```python
from ultralytics.data.concat_images import concat_vis_ir

result = concat_vis_ir(
    "path/to/visible.jpg",
    "path/to/thermal.jpg", 
    "path/to/output/fused.npy",
    target_size=(640, 640),
    ir_to_3ch=True
)
# result.shape = (640, 640, 6)
np.save("fused.npy", result)
```

#### `batch_concat_vis_ir(vis_paths, ir_paths, output_dir, ...)`

批量处理多个VIS-IR对。

**参数:**
- `vis_paths`: VIS图像路径列表
- `ir_paths`: IR图像路径列表 (长度必须与vis_paths相同)
- `output_dir`: 输出目录
- `target_size`, `ir_to_3ch`, `save_mode`: 同上

**返回:**
- `list[Path]`: 保存的输出文件路径列表

**示例:**
```python
from ultralytics.data.concat_images import batch_concat_vis_ir

output_paths = batch_concat_vis_ir(
    vis_paths=["dataset/vis/001.jpg", "dataset/vis/002.jpg"],
    ir_paths=["dataset/ir/001.jpg", "dataset/ir/002.jpg"],
    output_dir="dataset/fused_6ch",
    target_size=(640, 640)
)
```

### 2. 修改: `ultralytics/data/base.py` 中的 `load_image` 方法

**关键改动:** 修改了单通道到3通道的转换逻辑，使其支持 6 通道图像。

**原逻辑:**
```python
if im.ndim == 2:
    im = im[..., None]  # (H, W) -> (H, W, 1)
```

**新逻辑:**
```python
# Only convert 2D (grayscale) to 3D if not multi-channel format
# For 6-channel images from concatenated VIS-IR, skip this conversion
if im.ndim == 2 and self.channels <= 3:
    im = im[..., None]
```

这样:
- 当 `channels=6` 时，不会对6通道图像进行不必要的维度转换
- .npy 文件中的6通道数据被正确识别和加载
- .npy 文件的通道数必须与 `BaseDataset` 的 `channels` 参数匹配

## 使用流程

### Step 1: 离线拼接准备

首先，整理你的VIS和IR图像，确保它们能够配对:

```python
from pathlib import Path
from ultralytics.data.concat_images import batch_concat_vis_ir

# 假设目录结构:
# dataset/
#   ├── vis/
#   │   ├── frame_001.jpg
#   │   ├── frame_002.jpg
#   │   └── ...
#   └── ir/
#       ├── frame_001.jpg
#       ├── frame_002.jpg
#       └── ...

vis_dir = Path("dataset/vis")
ir_dir = Path("dataset/ir")

vis_files = sorted(vis_dir.glob("*.jpg"))
ir_files = sorted(ir_dir.glob("*.jpg"))

# 确保名称对应
assert len(vis_files) == len(ir_files), "VIS和IR文件数量不匹配"

# 批量拼接
output_paths = batch_concat_vis_ir(
    vis_paths=vis_files,
    ir_paths=ir_files,
    output_dir="dataset/fused_6ch",
    target_size=(640, 640),
    ir_to_3ch=True,
    save_mode="npy"
)
print(f"Successfully fused {len(output_paths)} image pairs")
```

### Step 2: 创建图像列表文件

将所有拼接后的 .npy 文件路径写入一个文本文件:

```python
from pathlib import Path

fused_dir = Path("dataset/fused_6ch")
image_list_file = "dataset/fused_images.txt"

npy_files = sorted(fused_dir.glob("*.npy"))
with open(image_list_file, "w") as f:
    for npy_file in npy_files:
        f.write(str(npy_file.absolute()) + "\n")

print(f"Created image list: {image_list_file}")
print(f"Total images: {len(npy_files)}")
```

### Step 3: 创建 BaseDataset 实例

使用修改后的 `BaseDataset`，**一定要设置 `channels=6`**:

```python
from ultralytics.data.base import BaseDataset
from ultralytics.models.yolo.detect import DetectionDataset  # 或你的具体任务

# 对于检测任务，使用 DetectionDataset (继承自 BaseDataset)
dataset = DetectionDataset(
    img_path="dataset/fused_images.txt",  # 指向拼接的.npy路径列表
    imgsz=640,
    cache=False,  # 或 'ram'/'disk' 根据需要
    augment=True,
    channels=6,   # IMPORTANT: 设置为6通道
    batch_size=16,
    stride=32,
    ...
)

# 验证
for i in range(3):
    sample = dataset[i]
    print(f"Sample {i}: img shape = {sample['img'].shape}")
    # 应该输出 torch tensor shape, 如 (6, 640, 640)
```

### Step 4: 模型适配

确保你的模型能接收 6 通道输入。例如，对于YOLO检测:

```python
# 修改模型的第一层以接收6通道输入
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 模型头部应该支持6通道输入
# 你可能需要微调模型的第一层:
# 将 model.model[0].conv 从 (3,H,W) -> (h,w) 改为支持 (6,H,W)
```

## 架构图

```
离线阶段:
  VIS (xxx.jpg) --\
                   --> concat_vis_ir() --> fused.npy
  IR  (yyy.jpg) --/

在线加载阶段:
  fused_images.txt (包含.npy路径列表)
         |
         v
  BaseDataset(..., channels=6)
         |
         v
  load_image(i)
         |
         v
  从 .npy 加载 (H, W, 6) ndarray
         |
         v
  transforms (HWC -> CHW)
         |
         v
  Model input (B, 6, H, W)
```

## 数据格式详解

### 存储格式 (*.npy)

```
Shape: (H, W, 6)  [所有操作都是HWC格式]
Dtype: uint8
通道分布:
  [0] - VIS Red   (来自BGR的R)
  [1] - VIS Green (来自BGR的G)  
  [2] - VIS Blue  (来自BGR的B)
  [3] - IR Red    (复制的灰度值或原始值)
  [4] - IR Green  (复制的灰度值或原始值)
  [5] - IR Blue   (复制的灰度值或原始值)
```

### Batch 变换流程

```
load_image 输出:      (H, W, 6)         # HWC
                       |
                    Compose transforms
                       | (ToTensor + 其他)
                       v
Dataset.__getitem__:  (6, H, W)         # CHW, torch.Tensor
                       |
                  DataLoader batch
                       |
                       v
Model input:          (B, 6, H, W)      # Batch of CHW
```

## 完整示例脚本

参考以下文件获取完整的可运行示例:

- [examples/concat_vis_ir_example.py](examples/concat_vis_ir_example.py) - 详细的使用示例
- [demo_vis_ir_concat.py](demo_vis_ir_concat.py) - 独立演示脚本

## FAQs

### Q1: 为什么用离线预处理而不是在线拼接?

**答:** 
- ? 离线: 磁盘读取只需一次 `.npy` 加载，更快；兼容现有缓存机制
- ? 在线: 每次读取时都要打开两个文件并拼接，性能较低

### Q2: IR 是单通道的，为什么要复制成3通道?

**答:** 
- 保持输入均匀性: 总是6通道 (VIS 3ch + IR 3ch)
- 简化下游处理: transforms 可以统一处理
- 如需不同处理，可设置 `ir_to_3ch=False` 得到4通道 (VIS 3ch + IR 1ch)

### Q3: 如何处理不同分辨率的VIS和IR?

**答:**  
`concat_vis_ir()` 的 `target_size` 参数会自动对齐。例如:
```python
concat_vis_ir(vis_path, ir_path, out_path, target_size=(640, 640))
# VIS和IR都会被resize到 640x640
```

### Q4: 能否跳过npy预处理，直接在load_image中拼接?

**答:** 可以，但不推荐。那样会变成在线拼接，每次读取需要:
1. 打开两个文件
2. 执行拼接操作  
3. 返回结果

相比一次 `.npy` 读取会慢很多。如果您确实需要在线拼接，可以修改 `load_image` 使其根据文件后缀判断是否需要拼接。

### Q5: 会占用多少磁盘空间?

**答:** 每张 640x640 的 6通道图像约 2.4MB (.npy格式):
- VIS: 1.2MB (640×640×3×1字节)
- IR:  1.2MB (640×640×3×1字节)  
- 总计: 2.4MB

1000张图像集合约 2.4GB (可用npz压缩至~600MB)。

## 调试技巧

### 验证拼接结果

```python
import numpy as np

# 加载并检查
fused = np.load("dataset/fused_6ch/image_001.npy")
print(f"Shape: {fused.shape}")
print(f"Dtype: {fused.dtype}")
print(f"VIS sample: {fused[100, 100, 0:3]}")  # 应该是真实BGR值
print(f"IR sample:  {fused[100, 100, 3:6]}")  # 应该相同(复制的)
```

### 验证DataLoader输出

```python
dataset = DetectionDataset(..., channels=6)
loader = torch.utils.data.DataLoader(dataset, batch_size=4)

for batch in loader:
    img = batch['img']  # torch.Tensor
    print(f"Batch shape: {img.shape}")  # 应该是 (B, 6, H, W)
    print(f"Dtype: {img.dtype}")  # 通常是 float32
    break
```

## 相关文件

| 文件 | 说明 |
|------|------|
| [ultralytics/data/concat_images.py](ultralytics/data/concat_images.py) | 拼接工具模块 |
| [ultralytics/data/base.py](ultralytics/data/base.py) | 修改了 `load_image` 方法 |
| [examples/concat_vis_ir_example.py](examples/concat_vis_ir_example.py) | 详细使用示例 |
| [tests/test_vis_ir_concat.py](tests/test_vis_ir_concat.py) | 单元测试 |
| [demo_vis_ir_concat.py](demo_vis_ir_concat.py) | 独立演示脚本 |

## 后续扩展

1. **更灵活的IR处理**: 支持不同的IR单通道映射方式 (例如直方图均衡化)
2. **在线增强**: 在transforms中实现VIS-IR特定的增强策略
3. **模型适配**: 为不同模型框架(YOLO/RT-DETR等)提供6通道适配层
4. **性能优化**: 考虑使用 HDF5 或 zarr 格式加快大规模数据访问

---

**最后更新**: 2026-07-09
