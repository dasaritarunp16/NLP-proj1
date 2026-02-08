
from utils.video import read, save
from utils.player_utils import PT
from utils.ball_utils import BT
from utils.court_line_detector import CLD
from utils.homography import homography
from utils.court_zones import CourtZones


def main():
    input = "test_video.mp4"
    vid_frames = read(input)

    Player_tracker = PT(model = "yolo12n.pt")

    p_detect = Player_tracker.detect_frames(vid_frames)

    ball_tracker = BT(model = "tennis_ball_best.pt")

    court_model_path = "keypoints_model_50.pth"
    court_line_detector = CLD(court_model_path)
    court_keypoints = court_line_detector.predict(vid_frames[0])
    court_keypoints_r = court_keypoints.reshape(-1,2)



    b_detect = ball_tracker.detect_frames(vid_frames)

    o_vid_frames = Player_tracker.draw_boxes(vid_frames, p_detect)
    o_vid_frames = ball_tracker.draw_boxes(o_vid_frames, b_detect)
    o_vid_frames = court_line_detector.draw_keypoints_on_video(o_vid_frames, court_keypoints)

    save(o_vid_frames, "output_video/output.mp4")


    print(f"Type: {type(court_keypoints_r[0])}")
    print(f"Shape : {court_keypoints_r.shape}")
    print(f"Content: \n{court_keypoints_r}")

    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    court_keypoints_reshaped = court_keypoints.reshape(-1, 2)

    frame_with_kp = vid_frames[0].copy()

    for i in range(len(court_keypoints_reshaped)):

        x = int(float(court_keypoints_reshaped[i, 0]))
        y = int(float(court_keypoints_reshaped[i, 1]))


        cv2.circle(frame_with_kp, (x, y), 8, (0, 255, 0), -1)

        cv2.putText(frame_with_kp, str(i), (x+10, y-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(frame_with_kp, cv2.COLOR_BGR2RGB))
        plt.title("Court Keypoints - Numbered")
        plt.show()

    H_points = homography(court_keypoints_reshaped)[0]

    print(f"Homography: {H_points}")

    # Create court zones from keypoints (pixel-space rectangles)
    court_zones = CourtZones(court_keypoints_reshaped)

    landed_balls = []
    for frame_count, balls in enumerate(b_detect):
        if 1 in balls:
            box = balls[1]

            x,y = ball_tracker.ball_center(box)

            # Classify zone directly in pixel space
            zone = court_zones.classify(x, y)

            ball_frame = np.array([[[x,y]]], dtype= np.float32)

            ball_homography = cv2.perspectiveTransform(ball_frame, H_points)


            if(ball_tracker.balls_in_court(ball_homography[0][0][0], ball_homography[0][0][1])):
                landed_balls.append({
                    'frame' : frame_count,
                    'x_coord' : ball_homography[0][0][0],
                    'y_coord' : ball_homography[0][0][1],
                    'zone' : zone,
                })



    print(f"balls_in:")
    for b in landed_balls:
        print(f"  Frame {b['frame']}: ({b['x_coord']:.2f}, {b['y_coord']:.2f}) -> {b['zone']}")


if __name__ == "__main__":
    main()
