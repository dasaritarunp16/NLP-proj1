
from utils.video import read, save
from utils.player_utils import PT
from utils.ball_utils import BT
from utils.ball_tracker_tracknet import BallTrackerTN
from utils.court_line_detector import CLD
from utils.homography import homography
from utils.court_zones import CourtZones
from utils.court_visualizer import CourtVisualizer


def main():
    input = "test_video.mp4"
    vid_frames = read(input, start_time=476)  # start at 7:56

    Player_tracker = PT(model = "yolo12n.pt")

    p_detect = Player_tracker.detect_frames(vid_frames)

    # Use TrackNet for ball detection (3-frame context, specialized for broadcast tennis)
    ball_tracker = BallTrackerTN(model_path="tracknet_weights.pt")

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

    # Build pixel-space court boundary polygon from the 4 corner keypoints
    # kp 0=far-left, 1=far-right, 3=near-right, 2=near-left
    court_boundary = np.array([
        court_keypoints_reshaped[0],
        court_keypoints_reshaped[1],
        court_keypoints_reshaped[3],
        court_keypoints_reshaped[2],
    ], dtype=np.float32).reshape(-1, 1, 2)

    # Build list of ball positions in real-world coords per frame
    ball_trajectory = []
    frames_with_ball = 0
    frames_out_of_court = 0
    frames_outside_pixel_court = 0
    frames_static = 0
    STATIC_THRESHOLD = 10    # pixels - if ball moves less than this, it's "static"
    STATIC_MAX_COUNT = 3     # consecutive static frames before we start filtering
    prev_px, prev_py = None, None
    static_count = 0
    for frame_count, balls in enumerate(b_detect):
        if 1 in balls:
            frames_with_ball += 1
            box = balls[1]
            x, y = ball_tracker.ball_center(box)

            # Filter: reject static false positives (scoreboard, ball basket, etc.)
            if prev_px is not None:
                dist = ((x - prev_px)**2 + (y - prev_py)**2)**0.5
                if dist < STATIC_THRESHOLD:
                    static_count += 1
                else:
                    static_count = 0
            prev_px, prev_py = x, y
            if static_count >= STATIC_MAX_COUNT:
                frames_static += 1
                continue

            # Filter: ball pixel position must be inside the court polygon
            if cv2.pointPolygonTest(court_boundary, (float(x), float(y)), False) < 0:
                frames_outside_pixel_court += 1
                continue

            ball_frame = np.array([[[x, y]]], dtype=np.float32)
            ball_homography = cv2.perspectiveTransform(ball_frame, H_points)
            rx, ry = ball_homography[0][0][0], ball_homography[0][0][1]

            if ball_tracker.balls_in_court(rx, ry):
                ball_trajectory.append({
                    'frame': frame_count,
                    'px': x, 'py': y,
                    'rx': rx, 'ry': ry,
                })
            else:
                frames_out_of_court += 1
                if frames_out_of_court <= 10:
                    print(f"  OUT: frame {frame_count} pixel=({x:.0f},{y:.0f}) -> real=({rx:.2f},{ry:.2f})")

    print(f"\n--- DEBUG ---")
    print(f"Total frames: {len(b_detect)}")
    print(f"Frames with ball detected: {frames_with_ball}")
    print(f"Frames static (false positive): {frames_static}")
    print(f"Frames outside pixel court: {frames_outside_pixel_court}")
    print(f"Frames out of real court (filtered): {frames_out_of_court}")
    print(f"Frames in court (ball_trajectory before sampling): {len(ball_trajectory)}")

    # Sample every Nth point for a cleaner ball path
    SAMPLE_EVERY = 5
    ball_trajectory = ball_trajectory[::SAMPLE_EVERY]
    print(f"Frames after sampling every {SAMPLE_EVERY}: {len(ball_trajectory)}")

    if len(ball_trajectory) > 0:
        print(f"First detection: frame {ball_trajectory[0]['frame']} ({ball_trajectory[0]['rx']:.2f}, {ball_trajectory[0]['ry']:.2f})")
        print(f"Last detection: frame {ball_trajectory[-1]['frame']} ({ball_trajectory[-1]['rx']:.2f}, {ball_trajectory[-1]['ry']:.2f})")
    print(f"--- END DEBUG ---\n")

    # --- Shot detection: furthest y-point in each pass ---
    # Track the ball back and forth. Each time it reverses direction,
    # the furthest y-point reached = where the ball landed for that shot.
    MIN_Y_TRAVEL = 3.0        # minimum y-distance between consecutive shots
    REVERSAL_THRESHOLD = 2.0  # ball must move this far back before we confirm reversal

    shot_landings = []
    if len(ball_trajectory) >= 2:
        extreme_pt = ball_trajectory[0]
        going_near = ball_trajectory[1]['ry'] > ball_trajectory[0]['ry']

        for i in range(1, len(ball_trajectory)):
            pt = ball_trajectory[i]

            if going_near:
                # Ball heading toward near baseline (y increasing)
                if pt['ry'] >= extreme_pt['ry']:
                    extreme_pt = pt  # new furthest point
                elif extreme_pt['ry'] - pt['ry'] > REVERSAL_THRESHOLD:
                    # Ball reversed — record the furthest point as landing
                    if len(shot_landings) == 0 or abs(extreme_pt['ry'] - shot_landings[-1]['y_coord']) >= MIN_Y_TRAVEL:
                        zone = court_zones.classify_real(extreme_pt['rx'], extreme_pt['ry'])
                        shot_landings.append({
                            'frame': extreme_pt['frame'],
                            'x_coord': extreme_pt['rx'],
                            'y_coord': extreme_pt['ry'],
                            'zone': zone,
                        })
                    extreme_pt = pt
                    going_near = False
            else:
                # Ball heading toward far baseline (y decreasing)
                if pt['ry'] <= extreme_pt['ry']:
                    extreme_pt = pt  # new furthest point
                elif pt['ry'] - extreme_pt['ry'] > REVERSAL_THRESHOLD:
                    # Ball reversed — record the furthest point as landing
                    if len(shot_landings) == 0 or abs(extreme_pt['ry'] - shot_landings[-1]['y_coord']) >= MIN_Y_TRAVEL:
                        zone = court_zones.classify_real(extreme_pt['rx'], extreme_pt['ry'])
                        shot_landings.append({
                            'frame': extreme_pt['frame'],
                            'x_coord': extreme_pt['rx'],
                            'y_coord': extreme_pt['ry'],
                            'zone': zone,
                        })
                    extreme_pt = pt
                    going_near = True

    print(f"\nShots detected: {len(shot_landings)}")
    for idx, b in enumerate(shot_landings):
        print(f"  Shot {idx+1} - Frame {b['frame']}: ({b['x_coord']:.2f}, {b['y_coord']:.2f}) -> {b['zone']}")

    # Visualize ball trajectory on 2D court
    visualizer = CourtVisualizer()
    visualizer.plot_trajectory(ball_trajectory)
    visualizer.plot_shots(ball_trajectory, shot_landings)


if __name__ == "__main__":
    main()
