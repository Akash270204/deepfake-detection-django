import os
import shutil
import random
from pathlib import Path

def split_dataset(real_source, fake_source):
    """Split dataset into train/val/test"""
    
    print("📂 Splitting dataset...\n")
    
    def get_images(directory):
        """Get all image files"""
        extensions = ('.jpg', '.jpeg', '.png')
        return [f for f in Path(directory).glob('*') if f.suffix.lower() in extensions]
    
    def copy_files(files, dest_dir):
        """Copy files to destination"""
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        for file in files:
            shutil.copy2(file, dest_dir)
    
    # Get all files
    real_files = get_images(real_source)
    fake_files = get_images(fake_source)
    
    print(f"Found {len(real_files)} real images")
    print(f"Found {len(fake_files)} fake images\n")
    
    # Shuffle
    random.seed(42)
    random.shuffle(real_files)
    random.shuffle(fake_files)
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.2
    
    # Real images
    real_train_idx = int(len(real_files) * train_ratio)
    real_val_idx = int(len(real_files) * (train_ratio + val_ratio))
    
    real_train = real_files[:real_train_idx]
    real_val = real_files[real_train_idx:real_val_idx]
    real_test = real_files[real_val_idx:]
    
    # Fake images
    fake_train_idx = int(len(fake_files) * train_ratio)
    fake_val_idx = int(len(fake_files) * (train_ratio + val_ratio))
    
    fake_train = fake_files[:fake_train_idx]
    fake_val = fake_files[fake_train_idx:fake_val_idx]
    fake_test = fake_files[fake_val_idx:]
    
    # Copy files
    print("Copying files...")
    copy_files(real_train, 'dataset/train/real')
    copy_files(real_val, 'dataset/val/real')
    copy_files(real_test, 'dataset/test/real')
    
    copy_files(fake_train, 'dataset/train/fake')
    copy_files(fake_val, 'dataset/val/fake')
    copy_files(fake_test, 'dataset/test/fake')
    
    print("\n✅ Dataset split complete!")
    print(f"Train: {len(real_train)} real, {len(fake_train)} fake")
    print(f"Val: {len(real_val)} real, {len(fake_val)} fake")
    print(f"Test: {len(real_test)} real, {len(fake_test)} fake")

if __name__ == '__main__':
    # Update these paths to your source folders
    REAL_SOURCE = 'path/to/your/real/images'
    FAKE_SOURCE = 'path/to/your/fake/images'
    
    split_dataset(REAL_SOURCE, FAKE_SOURCE)