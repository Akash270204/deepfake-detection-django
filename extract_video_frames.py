"""
Video Frame Extraction for Deepfake Training Dataset
Extracts frames from real and fake videos into proper directory structure.

Usage:
    python extract_video_frames.py

Directory structure (before):
    dataset/
    ├── videos/
    │   ├── real/           # Real videos (.mp4, .avi, .mov)
    │   └── fake/           # Fake videos (.mp4, .avi, .mov)

Directory structure (after):
    dataset/
    ├── train/
    │   ├── real/           # 70% of real video frames
    │   └── fake/           # 70% of fake video frames
    ├── val/
    │   ├── real/           # 15% of real video frames
    │   └── fake/           # 15% of fake video frames
    └── test/
        ├── real/           # 15% of real video frames
        └── fake/           # 15% of fake video frames
"""

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import shutil
import argparse

class VideoFrameExtractor:
    """Extract frames from videos for deepfake training"""
    
    def __init__(self, 
                 video_dir='dataset/videos',
                 output_dir='dataset',
                 sample_rate=30,
                 max_frames_per_video=50,
                 train_split=0.70,
                 val_split=0.15,
                 test_split=0.15,
                 min_resolution=(224, 224),
                 quality_threshold=100.0):
        """
        Args:
            video_dir: Directory containing real/ and fake/ video folders
            output_dir: Output directory for train/val/test splits
            sample_rate: Extract 1 frame every N frames (30 = 1 frame per second at 30fps)
            max_frames_per_video: Maximum frames to extract per video
            train_split: Percentage of frames for training (0.70 = 70%)
            val_split: Percentage for validation
            test_split: Percentage for testing
            min_resolution: Minimum (width, height) for extracted frames
            quality_threshold: Minimum Laplacian variance (blur detection)
        """
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.sample_rate = sample_rate
        self.max_frames_per_video = max_frames_per_video
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.min_resolution = min_resolution
        self.quality_threshold = quality_threshold
        
        # Video extensions to process
        self.video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        
        # Statistics
        self.stats = {
            'real': {'total': 0, 'train': 0, 'val': 0, 'test': 0, 'rejected': 0},
            'fake': {'total': 0, 'train': 0, 'val': 0, 'test': 0, 'rejected': 0}
        }
    
    def check_frame_quality(self, frame):
        """
        Check if frame meets quality requirements.
        Returns: (is_valid, reason)
        """
        # Check resolution
        h, w = frame.shape[:2]
        if w < self.min_resolution[0] or h < self.min_resolution[1]:
            return False, f"Low resolution: {w}x{h}"
        
        # Check blur (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        if laplacian_var < self.quality_threshold:
            return False, f"Too blurry: {laplacian_var:.1f}"
        
        # Check if frame is mostly black/white
        mean_intensity = np.mean(gray)
        if mean_intensity < 10 or mean_intensity > 245:
            return False, "Extreme brightness"
        
        return True, "OK"
    
    def extract_frames_from_video(self, video_path, output_dir, class_label):
        """
        Extract frames from a single video.
        Returns: number of frames extracted
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"⚠️  Could not open video: {video_path.name}")
            return 0
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        saved_count = 0
        video_id = video_path.stem
        
        while cap.isOpened() and saved_count < self.max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified rate
            if frame_count % self.sample_rate == 0:
                # Quality check
                is_valid, reason = self.check_frame_quality(frame)
                
                if is_valid:
                    # Save frame
                    frame_filename = f'{video_id}_frame_{saved_count:04d}.jpg'
                    frame_path = output_dir / frame_filename
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_count += 1
                else:
                    self.stats[class_label]['rejected'] += 1
            
            frame_count += 1
        
        cap.release()
        return saved_count
    
    def split_frames(self, frames_list, class_label):
        """
        Split extracted frames into train/val/test sets.
        Args:
            frames_list: List of frame file paths
            class_label: 'real' or 'fake'
        """
        # Shuffle frames
        random.shuffle(frames_list)
        
        total = len(frames_list)
        train_count = int(total * self.train_split)
        val_count = int(total * self.val_split)
        
        train_frames = frames_list[:train_count]
        val_frames = frames_list[train_count:train_count + val_count]
        test_frames = frames_list[train_count + val_count:]
        
        # Create directories
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split / class_label
            split_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy frames to respective directories
        print(f"   Splitting {class_label} frames...")
        
        for frame in tqdm(train_frames, desc="     Train", leave=False):
            shutil.copy2(frame, self.output_dir / 'train' / class_label / frame.name)
            self.stats[class_label]['train'] += 1
        
        for frame in tqdm(val_frames, desc="     Val", leave=False):
            shutil.copy2(frame, self.output_dir / 'val' / class_label / frame.name)
            self.stats[class_label]['val'] += 1
        
        for frame in tqdm(test_frames, desc="     Test", leave=False):
            shutil.copy2(frame, self.output_dir / 'test' / class_label / frame.name)
            self.stats[class_label]['test'] += 1
    
    def process_videos(self, class_label):
        """
        Process all videos for a given class (real or fake).
        Args:
            class_label: 'real' or 'fake'
        """
        video_class_dir = self.video_dir / class_label
        
        if not video_class_dir.exists():
            print(f"⚠️  Directory not found: {video_class_dir}")
            return
        
        # Find all videos
        video_files = []
        for ext in self.video_extensions:
            video_files.extend(video_class_dir.glob(f'*{ext}'))
        
        if not video_files:
            print(f"⚠️  No videos found in {video_class_dir}")
            return
        
        print(f"\n{'='*70}")
        print(f"Processing {class_label.upper()} videos ({len(video_files)} videos)")
        print(f"{'='*70}\n")
        
        # Create temporary extraction directory
        temp_dir = self.output_dir / 'temp_frames' / class_label
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract frames from each video
        all_frames = []
        
        for video_path in tqdm(video_files, desc=f"Extracting {class_label} frames"):
            frames_extracted = self.extract_frames_from_video(
                video_path, temp_dir, class_label
            )
            self.stats[class_label]['total'] += frames_extracted
        
        # Collect all extracted frames
        all_frames = list(temp_dir.glob('*.jpg'))
        
        print(f"\n   Total frames extracted: {len(all_frames)}")
        print(f"   Frames rejected (quality): {self.stats[class_label]['rejected']}")
        
        # Split into train/val/test
        if all_frames:
            self.split_frames(all_frames, class_label)
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
    
    def run(self):
        """Main extraction pipeline"""
        print("\n" + "="*70)
        print("🎬 VIDEO FRAME EXTRACTION FOR DEEPFAKE TRAINING")
        print("="*70)
        print(f"\n📂 Configuration:")
        print(f"   Video directory: {self.video_dir}")
        print(f"   Output directory: {self.output_dir}")
        print(f"   Sample rate: 1 frame every {self.sample_rate} frames")
        print(f"   Max frames per video: {self.max_frames_per_video}")
        print(f"   Train/Val/Test split: {self.train_split:.0%}/{self.val_split:.0%}/{self.test_split:.0%}")
        print(f"   Min resolution: {self.min_resolution[0]}x{self.min_resolution[1]}")
        print(f"   Blur threshold: {self.quality_threshold}")
        
        # Check if video directories exist
        if not self.video_dir.exists():
            print(f"\n❌ Error: Video directory not found: {self.video_dir}")
            print("\n💡 Expected structure:")
            print("   dataset/videos/real/    <- Put real videos here")
            print("   dataset/videos/fake/    <- Put fake videos here")
            return
        
        # Process both real and fake videos
        for class_label in ['real', 'fake']:
            self.process_videos(class_label)
        
        # Print final statistics
        self.print_summary()
    
    def print_summary(self):
        """Print extraction summary"""
        print("\n" + "="*70)
        print("✅ EXTRACTION COMPLETE")
        print("="*70)
        
        print("\n📊 Summary:")
        print(f"\n   REAL Videos:")
        print(f"      Total frames extracted: {self.stats['real']['total']}")
        print(f"      Train: {self.stats['real']['train']}")
        print(f"      Val:   {self.stats['real']['val']}")
        print(f"      Test:  {self.stats['real']['test']}")
        print(f"      Rejected (quality): {self.stats['real']['rejected']}")
        
        print(f"\n   FAKE Videos:")
        print(f"      Total frames extracted: {self.stats['fake']['total']}")
        print(f"      Train: {self.stats['fake']['train']}")
        print(f"      Val:   {self.stats['fake']['val']}")
        print(f"      Test:  {self.stats['fake']['test']}")
        print(f"      Rejected (quality): {self.stats['fake']['rejected']}")
        
        total_train = self.stats['real']['train'] + self.stats['fake']['train']
        total_val = self.stats['real']['val'] + self.stats['fake']['val']
        total_test = self.stats['real']['test'] + self.stats['fake']['test']
        
        print(f"\n   TOTAL:")
        print(f"      Train: {total_train}")
        print(f"      Val:   {total_val}")
        print(f"      Test:  {total_test}")
        
        # Class balance check
        if self.stats['real']['train'] > 0 and self.stats['fake']['train'] > 0:
            balance_ratio = min(self.stats['real']['train'], self.stats['fake']['train']) / \
                          max(self.stats['real']['train'], self.stats['fake']['train'])
            print(f"\n   Balance ratio: {balance_ratio:.2%}")
            
            if balance_ratio < 0.5:
                print(f"   ⚠️  WARNING: Dataset is imbalanced!")
                print(f"      Recommendation: Add more videos to minority class")
            else:
                print(f"   ✅ Dataset is well balanced")
        
        print("\n💡 Next Steps:")
        print(f"   1. Verify dataset structure:")
        print(f"      {self.output_dir}/train/real/")
        print(f"      {self.output_dir}/train/fake/")
        print(f"      {self.output_dir}/val/real/")
        print(f"      {self.output_dir}/val/fake/")
        print(f"      {self.output_dir}/test/real/")
        print(f"      {self.output_dir}/test/fake/")
        print(f"\n   2. Start training:")
        print(f"      python train_efficientnet_b1.py")
        print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Extract frames from deepfake videos for training'
    )
    parser.add_argument('--video-dir', default='dataset/videos',
                       help='Directory containing real/ and fake/ video folders')
    parser.add_argument('--output-dir', default='dataset',
                       help='Output directory for train/val/test splits')
    parser.add_argument('--sample-rate', type=int, default=30,
                       help='Extract 1 frame every N frames (default: 30)')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Maximum frames per video (default: 50)')
    parser.add_argument('--train-split', type=float, default=0.70,
                       help='Training set ratio (default: 0.70)')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test set ratio (default: 0.15)')
    parser.add_argument('--min-width', type=int, default=224,
                       help='Minimum frame width (default: 224)')
    parser.add_argument('--min-height', type=int, default=224,
                       help='Minimum frame height (default: 224)')
    parser.add_argument('--quality', type=float, default=100.0,
                       help='Blur threshold - Laplacian variance (default: 100.0)')
    
    args = parser.parse_args()
    
    extractor = VideoFrameExtractor(
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        max_frames_per_video=args.max_frames,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        min_resolution=(args.min_width, args.min_height),
        quality_threshold=args.quality
    )
    
    extractor.run()


if __name__ == '__main__':
    main()