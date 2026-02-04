import cv2
import sys
sys.path.append('/home/user/NLP-proj1/cv-proj1')

from utils.court_line_detector_hough import CLDH

# Load a frame from the video
video_path = '/home/user/NLP-proj1/cv-proj1/IMG_5391.mov'
cap = cv2.VideoCapture(video_path)

# Skip to frame 50 (more likely to have good court view)
cap.set(cv2.CAP_PROP_POS_FRAMES, 50)
ret, frame = cap.read()
cap.release()

if not ret:
    print("Failed to read video frame")
    exit(1)

print(f"Frame shape: {frame.shape}")

# Initialize CLDH with default parameters
cldh = CLDH(
    c_low=50,      # Canny low threshold
    c_high=150,    # Canny high threshold
    h_thresh=50,   # Hough threshold (lower = more lines)
    min_len=50,    # Minimum line length
    max_gap=10,    # Maximum gap
    w_thresh=200   # White threshold
)

# Step by step to see what's happening
print("\n1. Getting court edges...")
court_edges = cldh.get_lines(frame)
cv2.imwrite('debug_1_edges.jpg', court_edges)
print(f"   Saved: debug_1_edges.jpg")

print("\n2. Detecting lines with Hough Transform...")
lines = cldh.HLT(court_edges)
if lines is not None:
    print(f"   Found {len(lines)} lines")
else:
    print("   No lines found!")
    exit(1)

print("\n3. Classifying lines...")
h_lines, v_lines = cldh.label_lines(lines)
print(f"   Horizontal: {len(h_lines)}, Vertical: {len(v_lines)}")

# Draw just the lines
lines_image = cldh.draw_lines(frame, h_lines, v_lines)
cv2.imwrite('debug_2_lines.jpg', lines_image)
print(f"   Saved: debug_2_lines.jpg")

print("\n4. Finding intersections...")
keypoints = cldh.find_intersections(h_lines, v_lines, frame.shape[1], frame.shape[0])
print(f"   Found {len(keypoints)} keypoints")

# Draw everything
if len(keypoints) > 0:
    final_image = cldh.draw_all(frame, h_lines, v_lines, keypoints)
    cv2.imwrite('debug_3_final.jpg', final_image)
    print(f"   Saved: debug_3_final.jpg")

    print("\n5. Keypoint coordinates:")
    for i, (x, y) in enumerate(keypoints):
        print(f"   Point {i}: ({x}, {y})")
else:
    print("   No keypoints found - try adjusting parameters")

print("\nDone! Check the debug_*.jpg files.")
