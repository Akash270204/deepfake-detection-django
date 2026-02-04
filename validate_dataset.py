#!/usr/bin/env python
"""
Dataset Validation Script
Checks if dataset structure is correct and provides statistics
"""

import os
from pathlib import Path

project_root = Path(__file__).resolve().parent

def validate_dataset():
    print("\n" + "="*70)
    print("📊 DATASET VALIDATION")
    print("="*70 + "\n")
    
    dataset_dir = project_root / 'dataset'
    
    if not dataset_dir.exists():
        print("❌ Dataset directory not found!")
        print(f"   Please create: {dataset_dir}")
        return False
    
    splits = ['train', 'val', 'test']
    classes = ['real', 'fake']
    
    total_images = 0
    all_valid = True
    
    for split in splits:
        split_dir = dataset_dir / split
        
        if not split_dir.exists():
            print(f"⚠️  {split.upper()}: Directory not found")
            all_valid = False
            continue
        
        print(f"📁 {split.upper()}:")
        
        for cls in classes:
            cls_dir = split_dir / cls
            
            if not cls_dir.exists():
                print(f"   ❌ {cls}: Directory not found")
                all_valid = False
                continue
            
            # Count images
            images = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                images.extend(list(cls_dir.glob(ext)))
            
            count = len(images)
            total_images += count
            
            status = "✅" if count >= 100 else "⚠️ "
            print(f"   {status} {cls}: {count} images")
            
            if count < 100:
                print(f"      ⚠️  Minimum 100 images recommended")
    
    print(f"\n📊 Total images: {total_images}")
    
    if total_images < 600:
        print(f"⚠️  WARNING: Limited dataset size")
        print(f"   Recommended: 1,000+ images per class")
        print(f"   Your dataset: {total_images} total images")
    
    print("\n" + "="*70)
    
    if all_valid and total_images >= 200:
        print("✅ Dataset structure is valid!")
        print("   Ready to train.")
        return True
    else:
        print("❌ Dataset validation failed!")
        print("\n📚 Required structure:")
        print("   /app/dataset/")
        print("   ├── train/")
        print("   │   ├── real/  (at least 100 images)")
        print("   │   └── fake/  (at least 100 images)")
        print("   ├── val/")
        print("   │   ├── real/")
        print("   │   └── fake/")
        print("   └── test/")
        print("       ├── real/")
        print("       └── fake/")
        return False

if __name__ == '__main__':
    validate_dataset()
