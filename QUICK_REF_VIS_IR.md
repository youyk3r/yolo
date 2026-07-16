# VIS-IR 6通道快速参考卡

## 一句话总结
**将VIS和IR图像离线拼接成6通道.npy文件，通过 `channels=6` 参数加载，实现 `[B, 6, H, W]` 的多模态输入。**

---

## 快速开始 (3步)

### 1?? 拼接 VIS-IR 对

```python
from ultralytics.data.concat_images import batch_concat_vis_ir

batch_concat_vis_ir(
    vis_paths=["dataset/vis/001.jpg", ...],
    ir_paths=["dataset/ir/001.jpg", ...],
    output_dir="dataset/fused",
    target_size=(640, 640)
)
```

**生成:** `dataset/fused/` 中的 *.npy 文件 (形状: H×W×6)

---

### 2?? 创建图像列表

```python
from pathlib import Path

npy_files = sorted(Path("dataset/fused").glob("*.npy"))
with open("dataset/image_list.txt", "w") as f:
    for f_path in npy_files:
        f.write(str(f_path.absolute()) + "\n")
```

**生成:** `dataset/image_list.txt` (每行一个.npy路径)

---

### 3?? 加载到 Dataset

```python
dataset = MyDataset(
    img_path="dataset/image_list.txt",
    channels=6,  # ← 关键!
    imgsz=640,
    ...
)
```

**结果:** 自动加载6通道图像 `(B, 6, H, W)`

---

## 核心 API

### 单个拼接

```python
from ultralytics.data.concat_images import concat_vis_ir

im = concat_vis_ir(
    vis_path="path/to/visible.jpg",      # BGR, 3通道
    ir_path="path/to/thermal.jpg",       # 单通道灰度
    output_path="out/fused.npy",
    target_size=(640, 640),
    ir_to_3ch=True
)
# im.shape = (640, 640, 6)
# 通道 0-2: VIS (BGR)
# 通道 3-5: IR (复制的灰度)
```

### 批量拼接

```python
from ultralytics.data.concat_images import batch_concat_vis_ir

paths = batch_concat_vis_ir(
    vis_paths=vis_list,
    ir_paths=ir_list,
    output_dir="output",
    target_size=(640, 640),
    ir_to_3ch=True
)
# 返回保存的.npy文件路径列表
```

---

## 关键参数表

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vis_path` | str/Path | - | 可见光图像(BGR) |
| `ir_path` | str/Path | - | 红外图像(灰度) |
| `output_path` | str/Path | None | 输出.npy路径 |
| `target_size` | Tuple[h,w] | VIS原始尺寸 | 统一后的图像尺寸 |
| `ir_to_3ch` | bool | True | 是否复制IR为3通道 |
| `save_mode` | str | 'npy' | 保存格式('npy'或'npz') |

---

## 通道编码

```
6通道 (H, W, 6):
  [0-2] ← VIS BGR
  [3-5] ← IR RGB (实际都是相同灰度值)

元素值范围: 0-255 (uint8)
存储格式:  HWC (NumPy默认)
```

---

## 数据流

```
VIS (H?,W?,3) ─┐
               ├─→ Resize to (H,W) ─→ Concat ─→ (H,W,6) ─→ Save .npy
IR  (H?,W?)   ─┘                                         &
                                                       Return
```

---

## BaseDataset 改动

**文件:** `ultralytics/data/base.py` → `load_image` 方法

**改动:**
```python
# OLD:
if im.ndim == 2:
    im = im[..., None]  # (H,W) → (H,W,1)

# NEW:
if im.ndim == 2 and self.channels <= 3:
    im = im[..., None]  # 只对≤3通道做转换
```

**效果:**
- ? 6通道图像可以直接加载
- ? 向后兼容3通道及以下的图像
- ? .npy 文件通道数需匹配 `channels` 参数

---

## 常见配置

### 640×640 高清多模态

```python
batch_concat_vis_ir(
    vis_paths=vis_list,
    ir_paths=ir_list,
    output_dir="fused",
    target_size=(640, 640),  # Full HD
    ir_to_3ch=True
)

dataset = DetectionDataset(
    img_path="image_list.txt",
    channels=6,
    imgsz=640
)
```

### 轻量级预览 (320×320)

```python
batch_concat_vis_ir(
    ...,
    target_size=(320, 320),
    save_mode="npz"  # 压缩格式
)
```

### 4通道输出 (VIS 3ch + IR 1ch)

```python
batch_concat_vis_ir(
    ...,
    ir_to_3ch=False  # IR保持单通道
)
# 输出形状: (H, W, 4)，dataset中用 channels=4
```

---

## 验证清单

```
□ VIS和IR图像正确配对
□ 运行了 batch_concat_vis_ir()
□ 创建了 image_list.txt 文件
□ dataset 初始化时设置 channels=6
□ model 的输入层支持6通道
□ 加载一个batch检查形状 (B, 6, H, W)
```

---

## 性能指标

| 操作 | 耗时 | 备注 |
|------|------|------|
| 拼接 + 存储 1张 640? | ~50ms | 含磁盘I/O |
| 加载 1张 .npy 640? | ~5ms | 内存很快 |
| 1000张预处理 | ~1min | 单线程 |
| 磁盘占用 1张 | 2.4MB | .npy 格式 |

---

## 故障排除

| 问题 | 原因 | 解决 |
|------|------|------|
| `FileNotFoundError` | VIS/IR路径不存在 | 检查路径是否正确 |
| Shape mismatch | 通道数不匹配 | 检查 `channels` 参数是否为6 |
| 内存溢出 | 图像太大或batch太大 | 减小 `target_size` 或 `batch_size` |
| 加载很慢 | 可能在SD卡上 | 考虑用 `cache='ram'` |
| NaN or异常值 | IR数据不正常 | 检查IR图像的像素值范围 |

---

## 文件清单

| 文件 | 用途 |
|------|------|
| `ultralytics/data/concat_images.py` | 拼接工具 |
| `ultralytics/data/base.py` | 改动: `load_image()` |
| `VIS_IR_6CHANNEL_GUIDE.md` | 详细文档 |
| `examples/concat_vis_ir_example.py` | 使用示例 |
| `demo_vis_ir_concat.py` | 演示脚本 |

---

## 下一步

1. **调整transforms** - 确保处理6通道数据
2. **模型适配** - 改造输入层接收6通道
3. **性能优化** - 考虑多进程预处理
4. **数据集划分** - 生成train/val/test列表

---

**需要帮助?** 参考 [VIS_IR_6CHANNEL_GUIDE.md](VIS_IR_6CHANNEL_GUIDE.md) 获取完整文档。
