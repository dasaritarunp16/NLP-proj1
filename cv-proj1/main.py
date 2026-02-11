
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
    vid_frames = read(input, start_time=1034)  # start at 17:14

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
    STATIC_THRESHOLD = 5     # pixels - if ball moves less than this, it's "static"
    STATIC_MAX_COUNT = 8     # consecutive static frames before we start filtering
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

    # No sampling needed — TrackNet provides reliable detections
    print(f"Trajectory points: {len(ball_trajectory)}")

    if len(ball_trajectory) > 0:
        print(f"First detection: frame {ball_trajectory[0]['frame']} ({ball_trajectory[0]['rx']:.2f}, {ball_trajectory[0]['ry']:.2f})")
        print(f"Last detection: frame {ball_trajectory[-1]['frame']} ({ball_trajectory[-1]['rx']:.2f}, {ball_trajectory[-1]['ry']:.2f})")
    print(f"--- END DEBUG ---\n")

    # --- Shot detection: 3-point trajectory per shot ---
    # For each shot, capture: start (after last reversal), mid (a few frames in),
    # and end (furthest y-point before next reversal).
    # Only confirm a shot once the next direction change validates it.
    MIN_Y_TRAVEL = 3.0        # minimum y-distance between consecutive shots
    REVERSAL_THRESHOLD = 2.0  # ball must move this far back before we confirm reversal
    MIN_PASS_POINTS = 8       # minimum trajectory points for a pass to count as a real shot
    MID_SKIP = 2              # frames to skip from start to capture mid-flight position

    shot_landings = []
    pending_shot = None       # held until next reversal confirms it
    pass_start_idx = 0        # index where the current pass began

    if len(ball_trajectory) >= 2:
        extreme_pt = ball_trajectory[0]
        extreme_idx = 0
        going_near = ball_trajectory[1]['ry'] > ball_trajectory[0]['ry']

        for i in range(1, len(ball_trajectory)):
            pt = ball_trajectory[i]

            if going_near:
                if pt['ry'] >= extreme_pt['ry']:
                    extreme_pt = pt
                    extreme_idx = i
                elif extreme_pt['ry'] - pt['ry'] > REVERSAL_THRESHOLD:
                    pass_length = extreme_idx - pass_start_idx + 1
                    if pass_length < MIN_PASS_POINTS:
                        # Too short — not a real shot, just crosscourt wobble
                        # Keep tracking in the same direction, don't reset
                        continue

                    # Reversal detected — confirm the PREVIOUS pending shot
                    if pending_shot is not None:
                        if len(shot_landings) == 0 or abs(pending_shot['end']['ry'] - shot_landings[-1]['end']['ry']) >= MIN_Y_TRAVEL:
                            shot_landings.append(pending_shot)

                    # Build 3-point shot: start, mid, end
                    start_pt = ball_trajectory[pass_start_idx]
                    mid_idx = min(pass_start_idx + MID_SKIP, extreme_idx)
                    mid_pt = ball_trajectory[mid_idx]
                    end_pt = extreme_pt

                    end_zone = court_zones.classify_real(end_pt['rx'], end_pt['ry'])
                    pending_shot = {
                        'start': start_pt,
                        'mid': mid_pt,
                        'end': end_pt,
                        'zone': end_zone,
                    }
                    pass_start_idx = i
                    extreme_pt = pt
                    extreme_idx = i
                    going_near = False
            else:
                if pt['ry'] <= extreme_pt['ry']:
                    extreme_pt = pt
                    extreme_idx = i
                elif pt['ry'] - extreme_pt['ry'] > REVERSAL_THRESHOLD:
                    pass_length = extreme_idx - pass_start_idx + 1
                    if pass_length < MIN_PASS_POINTS:
                        # Too short — not a real shot, just crosscourt wobble
                        # Keep tracking in the same direction, don't reset
                        continue

                    # Reversal detected — confirm the PREVIOUS pending shot
                    if pending_shot is not None:
                        if len(shot_landings) == 0 or abs(pending_shot['end']['ry'] - shot_landings[-1]['end']['ry']) >= MIN_Y_TRAVEL:
                            shot_landings.append(pending_shot)

                    # Build 3-point shot: start, mid, end
                    start_pt = ball_trajectory[pass_start_idx]
                    mid_idx = min(pass_start_idx + MID_SKIP, extreme_idx)
                    mid_pt = ball_trajectory[mid_idx]
                    end_pt = extreme_pt

                    end_zone = court_zones.classify_real(end_pt['rx'], end_pt['ry'])
                    pending_shot = {
                        'start': start_pt,
                        'mid': mid_pt,
                        'end': end_pt,
                        'zone': end_zone,
                    }
                    pass_start_idx = i
                    extreme_pt = pt
                    extreme_idx = i
                    going_near = True

    # Capture the last pending shot (confirmed by end of data)
    if pending_shot is not None:
        if len(shot_landings) == 0 or abs(pending_shot['end']['ry'] - shot_landings[-1]['end']['ry']) >= MIN_Y_TRAVEL:
            shot_landings.append(pending_shot)

    # Also capture the final in-progress pass (from last reversal to end of trajectory)
    if len(ball_trajectory) >= 2:
        start_pt = ball_trajectory[pass_start_idx]
        mid_idx = min(pass_start_idx + MID_SKIP, len(ball_trajectory) - 1)
        mid_pt = ball_trajectory[mid_idx]
        end_pt = extreme_pt
        end_zone = court_zones.classify_real(end_pt['rx'], end_pt['ry'])
        final_shot = {
            'start': start_pt,
            'mid': mid_pt,
            'end': end_pt,
            'zone': end_zone,
        }
        if len(shot_landings) == 0 or abs(final_shot['end']['ry'] - shot_landings[-1]['end']['ry']) >= MIN_Y_TRAVEL:
            shot_landings.append(final_shot)

    # --- Merge consecutive same-direction shots ---
    # In tennis, every real shot changes direction. If two consecutive shots
    # both go near→far (or both far→near), it's the same shot with a bounce
    # in between. Merge them: keep first shot's start, last shot's end.
    if len(shot_landings) >= 2:
        merged = [shot_landings[0]]
        for s in shot_landings[1:]:
            prev = merged[-1]
            prev_dir = prev['end']['ry'] - prev['start']['ry']  # positive=going near, negative=going far
            curr_dir = s['end']['ry'] - s['start']['ry']

            if (prev_dir > 0 and curr_dir > 0) or (prev_dir < 0 and curr_dir < 0):
                # Same direction — merge: keep prev start, use curr end (more extreme)
                # Recalculate mid as midpoint between start and new end
                start_frame = prev['start']['frame']
                end_frame = s['end']['frame']
                # Find the trajectory point closest to the temporal midpoint
                target_frame = (start_frame + end_frame) // 2
                mid_pt = min(ball_trajectory, key=lambda p: abs(p['frame'] - target_frame))
                end_zone = court_zones.classify_real(s['end']['rx'], s['end']['ry'])
                merged[-1] = {
                    'start': prev['start'],
                    'mid': mid_pt,
                    'end': s['end'],
                    'zone': end_zone,
                }
            else:
                merged.append(s)
        shot_landings = merged

    print(f"\nShots detected: {len(shot_landings)}")
    for idx, s in enumerate(shot_landings):
        start = s['start']
        mid = s['mid']
        end = s['end']
        print(f"  Shot {idx+1}:")
        print(f"    Start  - Frame {start['frame']}: ({start['rx']:.2f}, {start['ry']:.2f}) -> {court_zones.classify_real(start['rx'], start['ry'])}")
        print(f"    Mid    - Frame {mid['frame']}: ({mid['rx']:.2f}, {mid['ry']:.2f}) -> {court_zones.classify_real(mid['rx'], mid['ry'])}")
        print(f"    End    - Frame {end['frame']}: ({end['rx']:.2f}, {end['ry']:.2f}) -> {s['zone']}")

    # Build shot_landings in the format plot_shots expects (using end point)
    shot_landings_for_viz = []
    for s in shot_landings:
        shot_landings_for_viz.append({
            'frame': s['end']['frame'],
            'x_coord': s['end']['rx'],
            'y_coord': s['end']['ry'],
            'zone': s['zone'],
        })

    # Visualize ball trajectory on 2D court
    visualizer = CourtVisualizer()
    visualizer.plot_trajectory(ball_trajectory)
    visualizer.plot_shots(ball_trajectory, shot_landings_for_viz)


if __name__ == "__main__":
    main()
