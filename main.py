

import cv2
import numpy as np
import os
import argparse
import subprocess
from pathlib import Path


class TennisFrameExtractor:
    def __init__(self, output_dir="extracted_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def download_video(self, url, output_path="downloaded_video.mp4"):
        """Download video from YouTube using yt-dlp."""
        print(f"Downloading video from: {url}")
        cmd = [
            "yt-dlp",
            "-f", "best[height<=720]",  
            "-o", output_path,
            url
        ]
        try:
            subprocess.run(cmd, check=True)
            print(f"Downloaded to: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"Error downloading video: {e}")
            return None
    
    def compute_motion_score(self, prev_frame, curr_frame):
        """Compute motion score between two frames."""
        if prev_frame is None:
            return 0
        
        
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
       
        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
        curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)
        
       
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
      
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        motion_score = np.sum(thresh > 0) / thresh.size
        
        return motion_score
    
    def is_duplicate(self, frame, last_saved_frame, threshold=0.98):
        """Check if frame is too similar to last saved frame."""
        if last_saved_frame is None:
            return False
     
        small_curr = cv2.resize(frame, (64, 64))
        small_last = cv2.resize(last_saved_frame, (64, 64))
   
        gray_curr = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)
        gray_last = cv2.cvtColor(small_last, cv2.COLOR_BGR2GRAY)
   
        similarity = cv2.matchTemplate(gray_curr, gray_last, cv2.TM_CCOEFF_NORMED)[0][0]
        
        return similarity > threshold
    
    def detect_court_region(self, frame):
        """
        Attempt to detect the tennis court region.
        Returns a mask or bounding box for the court area.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_lower = np.array([35, 40, 40])
        green_upper = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        blue_lower = np.array([90, 40, 40])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        court_mask = cv2.bitwise_or(green_mask, blue_mask)
        
        
        contours, _ = cv2.findContours(court_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
      
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h)
        
        return None
    
    def extract_frames_uniform(self, video_path, interval=30, max_frames=500):
        """
        Extract frames at uniform intervals.
        
        Args:
            video_path: Path to video file
            interval: Extract every Nth frame
            max_frames: Maximum number of frames to extract
        """
        print(f"Extracting frames uniformly (every {interval} frames)...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {fps:.1f} FPS, {total_frames} total frames")
        
        saved_frames = []
        frame_count = 0
        last_saved_frame = None
        
        while len(saved_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % interval == 0:
                # Skip duplicates
                if not self.is_duplicate(frame, last_saved_frame):
                    frame_name = f"frame_{frame_count:06d}.jpg"
                    frame_path = self.output_dir / frame_name
                    cv2.imwrite(str(frame_path), frame)
                    saved_frames.append(frame_path)
                    last_saved_frame = frame.copy()
                    
                    if len(saved_frames) % 50 == 0:
                        print(f"Extracted {len(saved_frames)} frames...")
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(saved_frames)} frames to {self.output_dir}")
        return saved_frames
    
    def extract_frames_motion(self, video_path, motion_threshold=0.01, 
                              min_interval=5, max_frames=500):
        """
        Extract frames based on motion detection.
        Prioritizes frames during active play (rallies).
        
        Args:
            video_path: Path to video file
            motion_threshold: Minimum motion score to consider frame
            min_interval: Minimum frames between extractions
            max_frames: Maximum number of frames to extract
        """
        print(f"Extracting frames based on motion (threshold={motion_threshold})...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {fps:.1f} FPS, {total_frames} total frames")
        
        saved_frames = []
        frame_count = 0
        frames_since_save = min_interval
        prev_frame = None
        last_saved_frame = None
        
        while len(saved_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            motion_score = self.compute_motion_score(prev_frame, frame)

            if motion_score > motion_threshold and frames_since_save >= min_interval:
                if not self.is_duplicate(frame, last_saved_frame):
                    frame_name = f"frame_{frame_count:06d}_motion_{motion_score:.3f}.jpg"
                    frame_path = self.output_dir / frame_name
                    cv2.imwrite(str(frame_path), frame)
                    saved_frames.append(frame_path)
                    last_saved_frame = frame.copy()
                    frames_since_save = 0
                    
                    if len(saved_frames) % 50 == 0:
                        print(f"Extracted {len(saved_frames)} frames...")
            
            prev_frame = frame.copy()
            frame_count += 1
            frames_since_save += 1
        
        cap.release()
        print(f"Extracted {len(saved_frames)} frames to {self.output_dir}")
        return saved_frames
    
    def extract_frames_hybrid(self, video_path, motion_threshold=0.008,
                              min_interval=5, uniform_interval=60, max_frames=500):
        """
        Hybrid extraction: motion-based during rallies + uniform sampling as backup.
        Best for tennis ball detection training.
        
        Args:
            video_path: Path to video file
            motion_threshold: Minimum motion score for motion-based extraction
            min_interval: Minimum frames between motion-based extractions
            uniform_interval: Fallback uniform interval for low-motion periods
            max_frames: Maximum number of frames to extract
        """
        print(f"Extracting frames (hybrid mode)...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {fps:.1f} FPS, {total_frames} total frames")
        
        saved_frames = []
        frame_count = 0
        frames_since_motion_save = min_interval
        frames_since_uniform_save = 0
        prev_frame = None
        last_saved_frame = None
        
        while len(saved_frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            motion_score = self.compute_motion_score(prev_frame, frame)
            should_save = False
            save_reason = ""
    
            if motion_score > motion_threshold and frames_since_motion_save >= min_interval:
                should_save = True
                save_reason = f"motion_{motion_score:.3f}"
                frames_since_motion_save = 0
          
            elif frames_since_uniform_save >= uniform_interval:
                should_save = True
                save_reason = "uniform"
                frames_since_uniform_save = 0
            
            if should_save and not self.is_duplicate(frame, last_saved_frame):
                frame_name = f"frame_{frame_count:06d}_{save_reason}.jpg"
                frame_path = self.output_dir / frame_name
                cv2.imwrite(str(frame_path), frame)
                saved_frames.append(frame_path)
                last_saved_frame = frame.copy()
                
                if len(saved_frames) % 50 == 0:
                    print(f"Extracted {len(saved_frames)} frames...")
            
            prev_frame = frame.copy()
            frame_count += 1
            frames_since_motion_save += 1
            frames_since_uniform_save += 1
        
        cap.release()
        print(f"Extracted {len(saved_frames)} frames to {self.output_dir}")
        return saved_frames
    
    def extract_with_ball_likelihood(self, video_path, max_frames=500):
  
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Video: {fps:.1f} FPS, {total_frames} total frames")
    
        print("Pass 1: Analyzing motion patterns...")
        motion_scores = []
        frame_count = 0
        prev_frame = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            score = self.compute_motion_score(prev_frame, frame)
            motion_scores.append((frame_count, score))
            prev_frame = frame.copy()
            frame_count += 1
            
            if frame_count % 500 == 0:
                print(f"Analyzed {frame_count}/{total_frames} frames...")
      
        avg_motion = np.mean([s[1] for s in motion_scores])
        rally_threshold = avg_motion * 1.5

        print(f"Pass 2: Extracting frames (rally threshold: {rally_threshold:.4f})...")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        saved_frames = []
        last_saved_frame = None
        frame_count = 0
        
        for idx, score in motion_scores:
            if len(saved_frames) >= max_frames:
                break
                
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            in_rally = score > rally_threshold
            
            if in_rally and idx % 5 == 0:
                if not self.is_duplicate(frame, last_saved_frame):
                    frame_name = f"frame_{idx:06d}_rally.jpg"
                    frame_path = self.output_dir / frame_name
                    cv2.imwrite(str(frame_path), frame)
                    saved_frames.append(frame_path)
                    last_saved_frame = frame.copy()
            elif not in_rally and idx % 30 == 0:  # Every 30 frames otherwise
                if not self.is_duplicate(frame, last_saved_frame):
                    frame_name = f"frame_{idx:06d}_other.jpg"
                    frame_path = self.output_dir / frame_name
                    cv2.imwrite(str(frame_path), frame)
                    saved_frames.append(frame_path)
                    last_saved_frame = frame.copy()
            
            if len(saved_frames) % 50 == 0 and len(saved_frames) > 0:
                print(f"Extracted {len(saved_frames)} frames...")
        
        cap.release()
        print(f"Extracted {len(saved_frames)} frames to {self.output_dir}")
        return saved_frames


def main():
    parser = argparse.ArgumentParser(description="Extract frames from tennis videos")
    parser.add_argument("--source", type=str, required=True,
                        help="Video file path or YouTube URL")
    parser.add_argument("--output", type=str, default="extracted_frames",
                        help="Output directory for frames")
    parser.add_argument("--mode", type=str, default="hybrid",
                        choices=["uniform", "motion", "hybrid", "smart"],
                        help="Extraction mode")
    parser.add_argument("--max-frames", type=int, default=500,
                        help="Maximum number of frames to extract")
    parser.add_argument("--interval", type=int, default=30,
                        help="Frame interval for uniform mode")
    
    args = parser.parse_args()
    
    extractor = TennisFrameExtractor(output_dir=args.output)

    video_path = args.source
    if args.source.startswith("http"):
        video_path = extractor.download_video(args.source)
        if video_path is None:
            return

    if args.mode == "uniform":
        frames = extractor.extract_frames_uniform(video_path, 
                                                   interval=args.interval,
                                                   max_frames=args.max_frames)
    elif args.mode == "motion":
        frames = extractor.extract_frames_motion(video_path,
                                                  max_frames=args.max_frames)
    elif args.mode == "hybrid":
        frames = extractor.extract_frames_hybrid(video_path,
                                                  max_frames=args.max_frames)
    elif args.mode == "smart":
        frames = extractor.extract_with_ball_likelihood(video_path,
                                                         max_frames=args.max_frames)
    
    print(f"\nDone! Extracted {len(frames)} frames.")
    print(f"Frames saved to: {args.output}/")
    print("\nNext steps:")
    print("1. Review and label frames using a tool like LabelImg or Roboflow")
    print("2. Create YOLO format annotations")
    print("3. Train your model!")


if __name__ == "__main__":
    main()
