import cv2
import os
import argparse
import uuid

def extract_frames_from_video(video_path, output_folder, frame_interval=1, max_frames=None, video_name=None):
    """Extract frames from a single video file with unique IDs."""
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Could not open: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0

    print(f"  FPS: {fps:.2f} | Frames: {total_frames} | Duration: {duration:.2f}s")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            # Unique filename: videoname_frameNUMBER_uuid.jpg
            unique_id = str(uuid.uuid4())[:8]
            frame_filename = os.path.join(
                output_folder,
                f"{video_name}_frame{frame_count:06d}_{unique_id}.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

            if saved_count % 100 == 0:
                print(f"  Saved {saved_count} frames so far...")

            if max_frames and saved_count >= max_frames:
                break

        frame_count += 1

    cap.release()
    print(f"  ✅ Saved {saved_count} frames from '{video_name}'")
    return saved_count


def extract_all_videos(videos_folder, output_folder, frame_interval=1, max_frames=None):
    """Process all videos and save all frames into a single folder."""

    VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v')

    video_files = [
        f for f in os.listdir(videos_folder)
        if f.lower().endswith(VIDEO_EXTENSIONS)
    ]

    if not video_files:
        print(f"❌ No video files found in '{videos_folder}'")
        return

    print(f"Found {len(video_files)} video(s) in '{videos_folder}'")
    print(f"All frames → single folder: '{output_folder}'")
    print(f"Frame interval: every {frame_interval} frame(s)")
    print("=" * 50)

    total_saved = 0

    for idx, video_file in enumerate(video_files, 1):
        video_path = os.path.join(videos_folder, video_file)
        video_name = os.path.splitext(video_file)[0]  # filename without extension

        print(f"\n[{idx}/{len(video_files)}] Processing: {video_file}")

        saved = extract_frames_from_video(
            video_path=video_path,
            output_folder=output_folder,   # ← same single folder for all
            frame_interval=frame_interval,
            max_frames=max_frames,
            video_name=video_name
        )
        total_saved += saved

    print("\n" + "=" * 50)
    print(f"✅ All done! Extracted {total_saved} total frames from {len(video_files)} videos")
    print(f"📁 All frames saved in: '{output_folder}'")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from all videos into a single folder")
    parser.add_argument("videos_folder", help="Folder containing input videos")
    parser.add_argument("output_folder", help="Single folder to save all extracted frames")
    parser.add_argument("-i", "--interval", type=int, default=1,
                        help="Extract every nth frame (default: 1)")
    parser.add_argument("-m", "--max", type=int, default=None,
                        help="Max frames per video (default: all)")

    args = parser.parse_args()

    extract_all_videos(
        videos_folder=args.videos_folder,
        output_folder=args.output_folder,
        frame_interval=args.interval,
        max_frames=args.max
    )


if __name__ == "__main__":
    main()
