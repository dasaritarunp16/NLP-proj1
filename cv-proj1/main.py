
from utils.video import read, save
from utils.player_utils import PT
from utils.ball_utils import BT
from utils.court_line_detector import CLD


def main():
    #input = "IMG_5391.mov"
    input = "test_video.mp4"
    vid_frames = read(input)
    
    Player_tracker = PT(model = "yolo12n.pt")
    
    p_detect = Player_tracker.detect_frames(vid_frames)
    
    ball_tracker = BT(model = "tennis_ball_best.pt")
    
    court_model_path = "models/keypoints_model_50.pth"
    court_line_detector = CLD(court_model_path)
    court_keypoints = court_line_detector.predict(vid_frames[0])
    
    b_detect = ball_tracker.detect_frames(vid_frames)
    
    o_vid_frames = Player_tracker.draw_boxes(vid_frames, p_detect)
    o_vid_frames = ball_tracker.draw_boxes(o_vid_frames, b_detect)
    o_vid_frames = court_line_detector.draw_keypoints_on_video(o_vid_frames, court_keypoints)
    
    save(o_vid_frames, "output_video/output.mp4")



if __name__ == "__main__":
    main()
