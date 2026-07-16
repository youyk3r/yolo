#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive 6-channel input verification script
Steps: Generate test data -> Create config -> Verify data loading -> Build model -> Check first layer channels
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import cv2
import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_dataset(output_dir: Path, num_images: int = 5, img_size: tuple = (480, 640)):
    """创建合成VIS-IR数据对和6通道.npy文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    vis_dir = output_dir / "vis"
    ir_dir = output_dir / "ir"
    fused_dir = output_dir / "fused_6ch"
    
    vis_dir.mkdir(exist_ok=True)
    ir_dir.mkdir(exist_ok=True)
    fused_dir.mkdir(exist_ok=True)
    
    logger.info(f"生成 {num_images} 个合成VIS-IR图像对...")
    
    image_list_path = fused_dir / "image_list.txt"
    with open(image_list_path, "w") as f:
        for i in range(num_images):
            # 生成合成VIS（BGR，3通道）
            vis = np.random.randint(0, 255, size=(img_size[0], img_size[1], 3), dtype=np.uint8)
            
            # 生成合成IR（灰度，1通道）
            ir = np.random.randint(100, 200, size=(img_size[0], img_size[1]), dtype=np.uint8)
            
            # 保存原始图像
            vis_path = vis_dir / f"image_{i:04d}.jpg"
            ir_path = ir_dir / f"image_{i:04d}.png"
            cv2.imwrite(str(vis_path), vis)
            cv2.imwrite(str(ir_path), ir)
            
            # 创建6通道.npy文件：[B, G, R, IR, IR, IR]
            ir_3ch = np.stack([ir, ir, ir], axis=-1)
            fused = np.concatenate([vis, ir_3ch], axis=-1)  # (H, W, 6)
            
            npy_path = fused_dir / f"image_{i:04d}.npy"
            np.save(str(npy_path), fused.astype(np.uint8))
            
            f.write(f"{npy_path}\n")
            logger.info(f"  ? 创建 {npy_path.name} - 形状: {fused.shape}")
    
    logger.info(f"? 测试数据集创建完成，位置: {fused_dir}")
    return fused_dir, image_list_path

def create_data_yaml(output_dir: Path, image_list_path: Path):
    """创建6通道的data.yaml配置"""
    data_yaml_path = output_dir / "data_6ch.yaml"
    
    yaml_content = f"""# 6通道VIS-IR数据集
path: {output_dir}
train: {image_list_path}
val: {image_list_path}
test: {image_list_path}

nc: 1  # 类别数（示例：1个类别）
names: ['object']  # 类别名称

# 关键配置：指定6个通道
channels: 6
"""
    
    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)
    
    logger.info(f"? 数据配置文件创建: {data_yaml_path}")
    logger.info(f"  关键参数: channels: 6")
    
    return data_yaml_path

def create_labels(output_dir: Path, num_images: int = 5):
    """创建简单的检测标签（YOLO格式）"""
    labels_dir = output_dir / "labels"
    labels_dir.mkdir(exist_ok=True)
    
    logger.info(f"生成 {num_images} 个标签文件...")
    for i in range(num_images):
        # YOLO格式: <class_id> <x_center> <y_center> <width> <height>（归一化坐标）
        label_content = "0 0.5 0.5 0.4 0.4\n"  # 1个目标
        label_path = labels_dir / f"image_{i:04d}.txt"
        with open(label_path, "w") as f:
            f.write(label_content)
    
    logger.info(f"? 标签文件创建完成: {labels_dir}")
    return labels_dir

def verify_data_loading(data_yaml_path: Path):
    """验证数据加载：检查dataset是否能识别和加载.npy文件"""
    logger.info("\n" + "="*60)
    logger.info("步骤 1: 验证数据加载（.npy文件扫描和加载）")
    logger.info("="*60)
    
    try:
        from ultralytics.data.dataset import YOLODataset
        from ultralytics.cfg import get_cfg
        
        # 加载配置
        cfg = get_cfg(cfg=data_yaml_path)
        logger.info(f"? 配置加载成功")
        logger.info(f"  channels参数: {cfg.get('channels', 3)}")
        
        # 创建Dataset实例
        dataset = YOLODataset(
            img_path=str(data_yaml_path).replace("data_6ch.yaml", "fused_6ch/image_list.txt"),
            imgsz=640,
            batch_size=2,
            augment=False,
            hyp=None,
            rect=False,
            cache=False,
            single_cls=False,
            stride=32,
            pad=0,
            prefix="",
            data=cfg
        )
        
        logger.info(f"? Dataset创建成功")
        logger.info(f"  数据集大小: {len(dataset)}")
        logger.info(f"  数据集通道数: {dataset.channels}")
        
        # 加载第一个样本
        if len(dataset) > 0:
            sample = dataset[0]
            if isinstance(sample, dict):
                img = sample.get('img')
            else:
                img = sample[0]
            
            logger.info(f"? 样本加载成功")
            logger.info(f"  样本形状: {img.shape if hasattr(img, 'shape') else 'N/A'}")
            if hasattr(img, 'shape') and len(img.shape) > 0:
                expected_ch = cfg.get('channels', 3)
                if img.shape[0] == expected_ch:
                    logger.info(f"  ? 通道数正确: {img.shape[0]} == {expected_ch}")
                else:
                    logger.warning(f"  ? 通道数不匹配: {img.shape[0]} != {expected_ch}")
        
        return True
    except Exception as e:
        logger.error(f"? 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_model_construction(data_yaml_path: Path):
    """验证模型构建：检查首层Conv是否为6通道输入"""
    logger.info("\n" + "="*60)
    logger.info("步骤 2: 验证模型构建（首层通道配置）")
    logger.info("="*60)
    
    try:
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        
        # 加载配置获取channels参数
        cfg = get_cfg(cfg=data_yaml_path)
        channels = cfg.get('channels', 3)
        
        logger.info(f"? 配置加载: channels={channels}")
        
        # 创建YOLO模型（不加载权重）
        model = YOLO("ultralytics/cfg/models/v8/yolov8n.yaml")
        
        # 检查首层Conv的输入通道
        first_module = None
        first_module_name = None
        for name, module in model.model.named_modules():
            if hasattr(module, 'in_channels'):
                first_module = module
                first_module_name = name
                break
        
        if first_module is not None:
            logger.info(f"? 模型首层识别成功")
            logger.info(f"  首层模块: {first_module_name}")
            logger.info(f"  首层类型: {type(first_module).__name__}")
            logger.info(f"  当前输入通道: {first_module.in_channels}")
            
            if first_module.in_channels == 3:
                logger.warning(f"? 首层仍为3通道输入，需要修改模型配置")
                logger.info(f"  解决方案: 在training时传递 channels={channels} 参数")
            else:
                logger.info(f"? 可调整当前模型以支持 {first_module.in_channels} 通道")
        
        # 尝试用6通道参数构建模型
        logger.info(f"\n尝试用 ch={channels} 重新构建模型...")
        model_6ch = None
        try:
            from ultralytics.models.yolo.detect import DetectionModel
            model_6ch = DetectionModel(cfg="ultralytics/cfg/models/v8/yolov8n.yaml", ch=channels, nc=1, verbose=False)
            
            logger.info(f"? 6通道模型构建成功")
            
            # 检查重建后的首层
            first_module_6ch = None
            for name, module in model_6ch.model.named_modules():
                if hasattr(module, 'in_channels'):
                    first_module_6ch = module
                    break
            
            if first_module_6ch is not None:
                logger.info(f"? 构建后首层输入通道: {first_module_6ch.in_channels}")
                if first_module_6ch.in_channels == channels:
                    logger.info(f"  ?? 首层通道成功设置为 {channels}")
                    return True
        except Exception as e:
            logger.error(f"? 6通道模型构建失败: {e}")
            return False
        
        return False
    except Exception as e:
        logger.error(f"? 模型验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_forward_pass(data_yaml_path: Path, img_size: tuple = (640, 640)):
    """验证前向传递：6通道数据能否通过模型"""
    logger.info("\n" + "="*60)
    logger.info("步骤 3: 验证前向传播（6通道数据通过模型）")
    logger.info("="*60)
    
    try:
        from ultralytics.cfg import get_cfg
        from ultralytics.models.yolo.detect import DetectionModel
        
        cfg = get_cfg(cfg=data_yaml_path)
        channels = cfg.get('channels', 3)
        
        # 创建6通道模型
        model = DetectionModel(cfg="ultralytics/cfg/models/v8/yolov8n.yaml", ch=channels, nc=1, verbose=False)
        model.eval()
        
        # 创建6通道输入张量
        batch_size = 2
        x = torch.randn(batch_size, channels, img_size[0], img_size[1], dtype=torch.float32)
        
        logger.info(f"? 创建输入张量: shape={x.shape}")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  通道数: {channels}")
        logger.info(f"  空间尺寸: {img_size[0]}x{img_size[1]}")
        
        # 前向传递
        with torch.no_grad():
            y = model(x)
        
        logger.info(f"? 前向传播成功")
        logger.info(f"  输出形状: {len(y)} 个检测头")
        
        return True
    except Exception as e:
        logger.error(f"? 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主验证流程"""
    logger.info("\n" + "="*60)
    logger.info("YOLO 6通道输入完整验证")
    logger.info("="*60)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # 步骤1: 生成测试数据
        logger.info("\n[步骤 0] 生成测试数据...")
        fused_dir, image_list_path = create_synthetic_dataset(tmpdir / "test_data", num_images=3)
        create_labels(tmpdir / "test_data", num_images=3)
        data_yaml_path = create_data_yaml(tmpdir / "test_data", image_list_path)
        
        # 步骤2: 验证数据加载
        logger.info("\n[步骤 1] 验证数据加载...")
        data_ok = verify_data_loading(data_yaml_path)
        
        # 步骤3: 验证模型构建
        logger.info("\n[步骤 2] 验证模型构建...")
        model_ok = verify_model_construction(data_yaml_path)
        
        # 步骤4: 验证前向传播
        logger.info("\n[步骤 3] 验证前向传播...")
        forward_ok = verify_forward_pass(data_yaml_path)
        
        # 汇总结果
        logger.info("\n" + "="*60)
        logger.info("验证总结")
        logger.info("="*60)
        
        results = {
            "? 数据加载": data_ok,
            "? 模型构建": model_ok,
            "? 前向传播": forward_ok,
        }
        
        for name, result in results.items():
            status = "? 通过" if result else "? 失败"
            logger.info(f"{name}: {status}")
        
        if all(results.values()):
            logger.info("\n? 所有验证通过！6通道输入已可用。")
            logger.info("\n接下来可以运行:")
            logger.info(f"  yolo detect train data={data_yaml_path} model=yolov8n.yaml epochs=5")
            return 0
        else:
            logger.error("\n? 某些验证失败，请检查上述错误信息")
            return 1

if __name__ == "__main__":
    sys.exit(main())
