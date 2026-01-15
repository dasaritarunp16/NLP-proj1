import cv2
import os

def read(video_path, max_frames=200):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while count < max_frames:
        reg, frame = cap.read()
        if reg:
            frames.append(frame)
            count += 1
        else:
            break
    cap.release()
    return frames

def save(output_frames, output_path):
    if len(output_frames) == 0:
        print("No frames to save")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    height, width = output_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Changed to mp4v
    out = cv2.VideoWriter(output_path, fourcc, 24, (width, height))
    
    for frame in output_frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {output_path}")