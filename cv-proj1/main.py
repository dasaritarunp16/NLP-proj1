import sys
from utils.video import read, save
from utils.player_utils import PT
from utils.ball_utils import BT
from utils.court_line_detector import CLD


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

    Player_tracker = PT(model="yolo12n.pt")
    p_detect = Player_tracker.detect_frames(vid_frames)

    ball_tracker = BT(model="tennis_ball_best.pt")
    b_detect = ball_tracker.detect_frames(vid_frames)

    court_model_path = "keypoints_model_50.pth"
    court_line_detector = CLD(court_model_path)
    court_keypoints = court_line_detector.predict(vid_frames[0])

    # Count frames where ball was detected
    ball_count = sum(1 for b in b_detect if len(b) > 0)
    print(f"Ball detected in {ball_count}/{len(vid_frames)} frames")

    o_vid_frames = Player_tracker.draw_boxes(vid_frames, p_detect)
    o_vid_frames = ball_tracker.draw_boxes(o_vid_frames, b_detect)
    o_vid_frames = court_line_detector.draw_keypoints_on_video(o_vid_frames, court_keypoints)

    save(o_vid_frames, "output_video/output.mp4")
    print("Done! Output saved to output_video/output.mp4")


if __name__ == "__main__":
    main()
