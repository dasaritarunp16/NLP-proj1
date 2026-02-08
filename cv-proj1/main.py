
from utils.video import read, save
from utils.player_utils import PT
from utils.ball_utils import BT
from utils.court_line_detector import CLD
from utils.homography import homography
from utils.court_zones import CourtZones
from utils.court_visualizer import CourtVisualizer


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

    # Build list of ball positions in real-world coords per frame
    ball_trajectory = []
    frames_with_ball = 0
    frames_out_of_court = 0
    for frame_count, balls in enumerate(b_detect):
        if 1 in balls:
            frames_with_ball += 1
            box = balls[1]
            x, y = ball_tracker.ball_center(box)
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
    print(f"Frames out of court (filtered): {frames_out_of_court}")
    print(f"Frames in court (ball_trajectory): {len(ball_trajectory)}")
    if len(ball_trajectory) > 0:
        print(f"First detection: frame {ball_trajectory[0]['frame']} ({ball_trajectory[0]['rx']:.2f}, {ball_trajectory[0]['ry']:.2f})")
        print(f"Last detection: frame {ball_trajectory[-1]['frame']} ({ball_trajectory[-1]['rx']:.2f}, {ball_trajectory[-1]['ry']:.2f})")
    print(f"--- END DEBUG ---\n")

    # --- Shot detection via direction reversals + path classification ---
    MIN_FRAME_GAP = 25       # minimum frames between reversal points
    MIN_Y_DISTANCE = 3.0     # minimum y-distance between reversal points
    BACKWARD_THRESHOLD = 0.3 # path efficiency below this = backward loop (noise)
    CLEAN_THRESHOLD = 0.7    # path efficiency above this = clean one-direction shot

    # Step 1: Find reversal indices (local peaks/valleys in ry)
    reversal_indices = []
    for i in range(1, len(ball_trajectory) - 1):
        prev_ry = ball_trajectory[i - 1]['ry']
        curr_ry = ball_trajectory[i]['ry']
        next_ry = ball_trajectory[i + 1]['ry']

        is_peak = curr_ry > prev_ry and curr_ry > next_ry
        is_valley = curr_ry < prev_ry and curr_ry < next_ry

        if is_peak or is_valley:
            if len(reversal_indices) > 0:
                last_rev = reversal_indices[-1]
                if ball_trajectory[i]['frame'] - ball_trajectory[last_rev]['frame'] < MIN_FRAME_GAP:
                    continue
                if abs(curr_ry - ball_trajectory[last_rev]['ry']) < MIN_Y_DISTANCE:
                    continue
            reversal_indices.append(i)

    # Step 2: Build shot segments between consecutive reversals
    boundaries = [0] + reversal_indices + [len(ball_trajectory) - 1]
    shot_landings = []

    for seg_idx in range(len(boundaries) - 1):
        start_idx = boundaries[seg_idx]
        end_idx = boundaries[seg_idx + 1]
        segment = ball_trajectory[start_idx:end_idx + 1]

        if len(segment) < 3:
            continue

        # Step 3: Compute path efficiency = net_y / total_y
        start_ry = segment[0]['ry']
        end_ry = segment[-1]['ry']
        net_y = abs(end_ry - start_ry)
        total_y = sum(abs(segment[j+1]['ry'] - segment[j]['ry']) for j in range(len(segment) - 1))

        if total_y == 0:
            continue

        ratio = net_y / total_y

        # Backward path (loops back on itself) -> discard as noise
        if ratio < BACKWARD_THRESHOLD:
            print(f"  Segment {seg_idx+1} (frames {segment[0]['frame']}-{segment[-1]['frame']}): DISCARDED (backward, ratio={ratio:.2f})")
            continue

        # Step 4: Classify landing zone based on path shape
        if ratio >= CLEAN_THRESHOLD:
            # Clean one-direction path -> zone of the final point
            last_pt = segment[-1]
            zone = court_zones.classify(last_pt['px'], last_pt['py'])
        else:
            # Zig-zaggy path -> zone with most frame points
            zone_counts = {}
            zone_points = {}
            for pt in segment:
                z = court_zones.classify(pt['px'], pt['py'])
                if z not in zone_counts:
                    zone_counts[z] = 0
                    zone_points[z] = []
                zone_counts[z] += 1
                zone_points[z].append((pt['rx'], pt['ry']))

            max_count = max(zone_counts.values())
            # Zones within 80% of the max count are "tied"
            top_zones = [z for z, c in zone_counts.items() if c >= max_count * 0.8]

            if len(top_zones) == 1:
                zone = top_zones[0]
            else:
                # Tiebreak: zone where points are closest to each other (most clustered)
                best_zone = top_zones[0]
                best_density = float('inf')
                for z in top_zones:
                    pts = zone_points[z]
                    if len(pts) < 2:
                        avg_dist = 0
                    else:
                        total_dist = 0
                        count = 0
                        for a in range(len(pts)):
                            for b in range(a + 1, len(pts)):
                                total_dist += ((pts[a][0] - pts[b][0])**2 + (pts[a][1] - pts[b][1])**2)**0.5
                                count += 1
                        avg_dist = total_dist / count
                    if avg_dist < best_density:
                        best_density = avg_dist
                        best_zone = z
                zone = best_zone

            last_pt = segment[-1]

        shot_landings.append({
            'frame': last_pt['frame'],
            'x_coord': last_pt['rx'],
            'y_coord': last_pt['ry'],
            'zone': zone,
            'path_ratio': ratio,
        })

    print(f"\nShots detected: {len(shot_landings)}")
    for idx, b in enumerate(shot_landings):
        print(f"  Shot {idx+1} - Frame {b['frame']}: ({b['x_coord']:.2f}, {b['y_coord']:.2f}) -> {b['zone']} (ratio={b['path_ratio']:.2f})")

    # Visualize ball trajectory on 2D court
    visualizer = CourtVisualizer()
    visualizer.plot_trajectory(ball_trajectory)
    visualizer.plot_shots(ball_trajectory, shot_landings)


if __name__ == "__main__":
    main()
