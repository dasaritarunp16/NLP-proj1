import sys
import numpy as np
import cv2
from utils.video import read, save
from utils.player_utils import PT
from utils.ball_utils import BT
from utils.court_line_detector import CLD
from utils.homography import homography


def main():
    input_video = "test_video.mp4"
    if len(sys.argv) > 1:
        input_video = sys.argv[1]

    print(f"Reading video: {input_video}")
    vid_frames = read(input_video)
    print(f"Loaded {len(vid_frames)} frames")

    if len(vid_frames) == 0:
        print("ERROR: No frames loaded. Check that the video file exists and is a valid video.")
        return

    # Player detection
    Player_tracker = PT(model="yolo12n.pt")
    p_detect = Player_tracker.detect_frames(vid_frames)

    # Ball detection
    ball_tracker = BT(model="tennis_ball_best.pt")
    b_detect = ball_tracker.detect_frames(vid_frames)

    # Court line detection
    court_model_path = "keypoints_model_50.pth"
    court_line_detector = CLD(court_model_path)
    court_keypoints = court_line_detector.predict(vid_frames[0])
    court_keypoints_r = court_keypoints.reshape(-1, 2)

    # Homography: map pixel coords to real-world court coords
    H_points = homography(court_keypoints_r)[0]

    # Find balls that landed in court
    ball_count = sum(1 for b in b_detect if len(b) > 0)
    print(f"Ball detected in {ball_count}/{len(vid_frames)} frames")

    landed_balls = []
    for frame_count, balls in enumerate(b_detect):
        if 1 in balls:
            box = balls[1]
            x, y = ball_tracker.ball_center(box)
            ball_frame = np.array([[[x, y]]], dtype=np.float32)
            ball_homography = cv2.perspectiveTransform(ball_frame, H_points)
            rx, ry = ball_homography[0][0][0], ball_homography[0][0][1]

            if ball_tracker.balls_in_court(rx, ry):
                landed_balls.append({
                    'frame': frame_count,
                    'x_coord': rx,
                    'y_coord': ry,
                })

    print(f"Balls landed in court: {len(landed_balls)}")
    for b in landed_balls:
        print(f"  Frame {b['frame']}: ({b['x_coord']:.2f}, {b['y_coord']:.2f})")

    # Draw detections on video
    o_vid_frames = Player_tracker.draw_boxes(vid_frames, p_detect)
    o_vid_frames = ball_tracker.draw_boxes(o_vid_frames, b_detect)
    o_vid_frames = court_line_detector.draw_keypoints_on_video(o_vid_frames, court_keypoints)

    save(o_vid_frames, "output_video/output.mp4")
    print("Done! Output saved to output_video/output.mp4")


if __name__ == "__main__":
    main()
