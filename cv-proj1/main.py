
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.video import read, save
from utils.player_utils import PT
from utils.ball_utils import BT
from utils.ball_tracker_tracknet import BallTrackerTN
from utils.court_detector_robust import CourtDetectorRobust
from utils.homography import homography
from utils.court_zones import CourtZones
from utils.court_visualizer import CourtVisualizer


def main():
    input = "test_video.mp4"
    vid_frames = read(input, start_time=25)  # start at 0:25

    if len(vid_frames) == 0:
        print(f"ERROR: No frames read from '{input}'.")
        return

    Player_tracker = PT(model = "yolo12n.pt")
    p_detect = Player_tracker.detect_frames(vid_frames)

    ball_tracker = BallTrackerTN(model_path="tracknet_weights.pt")

    court_line_detector = CourtDetectorRobust("court_detection_weights.pth")

    # predict court keypoints periodically to handle camera movement
    KEYPOINT_INTERVAL = 30
    keypoint_frames = {}
    homography_frames = {}
    boundary_frames = {}

    print(f"Running keypoint detection (every {KEYPOINT_INTERVAL} frames)...")
    for f_idx in range(0, len(vid_frames), KEYPOINT_INTERVAL):
        kp = court_line_detector.predict(vid_frames[f_idx])
        kp_r = kp.reshape(-1, 2)
        H = homography(kp_r)[0]
        boundary = np.array([
            kp_r[0], kp_r[1], kp_r[3], kp_r[2],
        ], dtype=np.float32).reshape(-1, 1, 2)
        keypoint_frames[f_idx] = kp_r
        homography_frames[f_idx] = H
        boundary_frames[f_idx] = boundary
    print(f"  Keypoints predicted on {len(keypoint_frames)} frames")

    keypoint_indices = sorted(keypoint_frames.keys())
    def get_nearest_kp_idx(frame_idx):
        best = keypoint_indices[0]
        for ki in keypoint_indices:
            if abs(ki - frame_idx) < abs(best - frame_idx):
                best = ki
        return best

    court_keypoints = court_line_detector.predict(vid_frames[0])
    court_keypoints_r = court_keypoints.reshape(-1,2)

    b_detect = ball_tracker.detect_frames(vid_frames)

    o_vid_frames = Player_tracker.draw_boxes(vid_frames, p_detect)
    o_vid_frames = ball_tracker.draw_boxes(o_vid_frames, b_detect)
    for f_idx in range(len(o_vid_frames)):
        nearest_kp = get_nearest_kp_idx(f_idx)
        kp_flat = keypoint_frames[nearest_kp].flatten()
        o_vid_frames[f_idx] = court_line_detector.draw_keypoints(o_vid_frames[f_idx], kp_flat)

    save(o_vid_frames, "output_video/output.mp4")

    print(f"Type: {type(court_keypoints_r[0])}")
    print(f"Shape : {court_keypoints_r.shape}")
    print(f"Content: \n{court_keypoints_r}")

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

    court_zones = CourtZones(court_keypoints_reshaped)

    # build ball trajectory in real-world coords
    ball_trajectory = []
    frames_with_ball = 0
    frames_out_of_court = 0
    frames_outside_pixel_court = 0
    frames_static = 0
    STATIC_THRESHOLD = 5
    STATIC_MAX_COUNT = 8
    prev_px, prev_py = None, None
    static_count = 0

    for frame_count, balls in enumerate(b_detect):
        if 1 in balls:
            frames_with_ball += 1
            box = balls[1]
            x, y = ball_tracker.ball_center(box)

            # filter static false positives
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

            nearest_kp = get_nearest_kp_idx(frame_count)
            court_boundary = boundary_frames[nearest_kp]
            H_points = homography_frames[nearest_kp]

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
    print(f"Frames with ball: {frames_with_ball}")
    print(f"Static: {frames_static}")
    print(f"Outside pixel court: {frames_outside_pixel_court}")
    print(f"Out of real court: {frames_out_of_court}")
    print(f"Trajectory points: {len(ball_trajectory)}")
    if len(ball_trajectory) > 0:
        print(f"First: frame {ball_trajectory[0]['frame']} ({ball_trajectory[0]['rx']:.2f}, {ball_trajectory[0]['ry']:.2f})")
        print(f"Last: frame {ball_trajectory[-1]['frame']} ({ball_trajectory[-1]['rx']:.2f}, {ball_trajectory[-1]['ry']:.2f})")
    print(f"--- END DEBUG ---\n")

    # build player positions in real-world coords (foot position)
    player_positions = {}
    for frame_idx, players in enumerate(p_detect):
        player_positions[frame_idx] = {}
        nearest_kp = get_nearest_kp_idx(frame_idx)
        H_points = homography_frames[nearest_kp]
        court_boundary = boundary_frames[nearest_kp]
        for track_id, box in players.items():
            x1, y1, x2, y2 = box
            foot_x = (x1 + x2) / 2
            foot_y = y2
            if cv2.pointPolygonTest(court_boundary, (float(foot_x), float(foot_y)), False) < 0:
                continue
            foot_frame = np.array([[[foot_x, foot_y]]], dtype=np.float32)
            foot_real = cv2.perspectiveTransform(foot_frame, H_points)
            rx, ry = float(foot_real[0][0][0]), float(foot_real[0][0][1])
            if ball_tracker.balls_in_court(rx, ry):
                player_positions[frame_idx][track_id] = (rx, ry)

    # identify near/far players by avg y position
    player_avg_y = {}
    player_counts = {}
    for frame_idx, players in player_positions.items():
        for track_id, (rx, ry) in players.items():
            if track_id not in player_avg_y:
                player_avg_y[track_id] = 0.0
                player_counts[track_id] = 0
            player_avg_y[track_id] += ry
            player_counts[track_id] += 1

    sorted_players = sorted(player_counts.keys(), key=lambda tid: player_counts[tid], reverse=True)
    near_player_id = None
    far_player_id = None
    if len(sorted_players) >= 2:
        p1, p2 = sorted_players[0], sorted_players[1]
        avg_y1 = player_avg_y[p1] / player_counts[p1]
        avg_y2 = player_avg_y[p2] / player_counts[p2]
        if avg_y1 > avg_y2:
            near_player_id, far_player_id = p1, p2
        else:
            near_player_id, far_player_id = p2, p1
        print(f"Near player: ID {near_player_id} (avg y={player_avg_y[near_player_id]/player_counts[near_player_id]:.2f})")
        print(f"Far player: ID {far_player_id} (avg y={player_avg_y[far_player_id]/player_counts[far_player_id]:.2f})")

    # shot detection - track ball y extremes and detect reversals
    MIN_Y_TRAVEL = 3.0
    REVERSAL_THRESHOLD = 2.0
    MIN_PASS_POINTS = 8
    MID_SKIP = 2
    MAX_FRAME_GAP = 12

    shot_landings = []
    pending_shot = None
    pass_start_idx = 0

    if len(ball_trajectory) >= 2:
        extreme_pt = ball_trajectory[0]
        extreme_idx = 0
        going_near = ball_trajectory[1]['ry'] > ball_trajectory[0]['ry']

        for i in range(1, len(ball_trajectory)):
            pt = ball_trajectory[i]
            prev_pt = ball_trajectory[i - 1]

            # force break on large frame gap (tracker lost the ball)
            if pt['frame'] - prev_pt['frame'] > MAX_FRAME_GAP:
                pass_length = extreme_idx - pass_start_idx + 1
                if pass_length >= MIN_PASS_POINTS:
                    if pending_shot is not None:
                        if len(shot_landings) == 0 or abs(pending_shot['end']['ry'] - shot_landings[-1]['end']['ry']) >= MIN_Y_TRAVEL:
                            shot_landings.append(pending_shot)
                    start_pt = ball_trajectory[pass_start_idx]
                    mid_idx = min(pass_start_idx + MID_SKIP, extreme_idx)
                    mid_pt = ball_trajectory[mid_idx]
                    end_pt = extreme_pt
                    end_zone = court_zones.classify_real(end_pt['rx'], end_pt['ry'])
                    pending_shot = {
                        'start': start_pt, 'mid': mid_pt,
                        'end': end_pt, 'zone': end_zone,
                    }
                pass_start_idx = i
                extreme_pt = pt
                extreme_idx = i
                if i + 1 < len(ball_trajectory):
                    going_near = ball_trajectory[i + 1]['ry'] > pt['ry']
                continue

            if going_near:
                if pt['ry'] >= extreme_pt['ry']:
                    extreme_pt = pt
                    extreme_idx = i
                elif extreme_pt['ry'] - pt['ry'] > REVERSAL_THRESHOLD:
                    pass_length = i - pass_start_idx + 1
                    if pass_length < MIN_PASS_POINTS:
                        continue

                    if pending_shot is not None:
                        if len(shot_landings) == 0 or abs(pending_shot['end']['ry'] - shot_landings[-1]['end']['ry']) >= MIN_Y_TRAVEL:
                            shot_landings.append(pending_shot)

                    start_pt = ball_trajectory[pass_start_idx]
                    mid_idx = min(pass_start_idx + MID_SKIP, extreme_idx)
                    mid_pt = ball_trajectory[mid_idx]
                    end_pt = extreme_pt
                    end_zone = court_zones.classify_real(end_pt['rx'], end_pt['ry'])
                    pending_shot = {
                        'start': start_pt, 'mid': mid_pt,
                        'end': end_pt, 'zone': end_zone,
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
                    pass_length = i - pass_start_idx + 1
                    if pass_length < MIN_PASS_POINTS:
                        continue

                    if pending_shot is not None:
                        if len(shot_landings) == 0 or abs(pending_shot['end']['ry'] - shot_landings[-1]['end']['ry']) >= MIN_Y_TRAVEL:
                            shot_landings.append(pending_shot)

                    start_pt = ball_trajectory[pass_start_idx]
                    mid_idx = min(pass_start_idx + MID_SKIP, extreme_idx)
                    mid_pt = ball_trajectory[mid_idx]
                    end_pt = extreme_pt
                    end_zone = court_zones.classify_real(end_pt['rx'], end_pt['ry'])
                    pending_shot = {
                        'start': start_pt, 'mid': mid_pt,
                        'end': end_pt, 'zone': end_zone,
                    }
                    pass_start_idx = i
                    extreme_pt = pt
                    extreme_idx = i
                    going_near = True

    # capture last pending + final in-progress pass
    if pending_shot is not None:
        if len(shot_landings) == 0 or abs(pending_shot['end']['ry'] - shot_landings[-1]['end']['ry']) >= MIN_Y_TRAVEL:
            shot_landings.append(pending_shot)

    if len(ball_trajectory) >= 2:
        start_pt = ball_trajectory[pass_start_idx]
        mid_idx = min(pass_start_idx + MID_SKIP, len(ball_trajectory) - 1)
        mid_pt = ball_trajectory[mid_idx]
        end_pt = extreme_pt
        end_zone = court_zones.classify_real(end_pt['rx'], end_pt['ry'])
        final_shot = {
            'start': start_pt, 'mid': mid_pt,
            'end': end_pt, 'zone': end_zone,
        }
        if len(shot_landings) == 0 or abs(final_shot['end']['ry'] - shot_landings[-1]['end']['ry']) >= MIN_Y_TRAVEL:
            shot_landings.append(final_shot)

    # merge consecutive same-direction shots
    def merge_same_dir(shots):
        if len(shots) < 2:
            return shots
        merged = [shots[0]]
        for s in shots[1:]:
            prev = merged[-1]
            prev_dir = prev['end']['ry'] - prev['start']['ry']
            curr_dir = s['end']['ry'] - s['start']['ry']
            if (prev_dir > 0 and curr_dir > 0) or (prev_dir < 0 and curr_dir < 0):
                start_frame = prev['start']['frame']
                end_frame = s['end']['frame']
                target_frame = (start_frame + end_frame) // 2
                mid_pt = min(ball_trajectory, key=lambda p: abs(p['frame'] - target_frame))
                end_zone = court_zones.classify_real(s['end']['rx'], s['end']['ry'])
                merged[-1] = {
                    'start': prev['start'], 'mid': mid_pt,
                    'end': s['end'], 'zone': end_zone,
                }
            else:
                merged.append(s)
        return merged

    shot_landings = merge_same_dir(shot_landings)

    # filter short shots
    MIN_SHOT_FRAMES = 5
    shot_landings = [s for s in shot_landings
                     if s['end']['frame'] - s['start']['frame'] >= MIN_SHOT_FRAMES]

    # second merge pass after filtering
    shot_landings = merge_same_dir(shot_landings)

    # player-ball fusion to validate landing zones
    PLAYER_SEARCH_WINDOW = 5
    BALL_WEIGHT = 0.6
    PLAYER_WEIGHT = 0.4
    NET_Y = 11.885

    for s in shot_landings:
        end_frame = s['end']['frame']
        ball_going_far = s['end']['ry'] < s['start']['ry']

        receiver_id = far_player_id if ball_going_far else near_player_id
        if receiver_id is None:
            s['player_zone'] = None
            s['fused_zone'] = s['zone']
            continue

        receiver_pos = None
        for offset in range(0, PLAYER_SEARCH_WINDOW + 1):
            for f in [end_frame + offset, end_frame - offset]:
                if f in player_positions and receiver_id in player_positions[f]:
                    receiver_pos = player_positions[f][receiver_id]
                    break
            if receiver_pos is not None:
                break

        if receiver_pos is None:
            s['player_zone'] = None
            s['fused_zone'] = s['zone']
            continue

        p_rx, p_ry = receiver_pos
        player_zone = court_zones.classify_real(p_rx, p_ry)
        s['player_zone'] = player_zone
        s['player_pos'] = receiver_pos

        ball_zone = s['zone']
        player_closer_to_net = abs(p_ry - NET_Y) < abs(s['end']['ry'] - NET_Y)

        if ball_zone == player_zone:
            s['fused_zone'] = ball_zone
        elif player_closer_to_net:
            s['fused_zone'] = player_zone
        else:
            fused_rx = BALL_WEIGHT * s['end']['rx'] + PLAYER_WEIGHT * p_rx
            fused_ry = s['end']['ry']
            s['fused_zone'] = court_zones.classify_real(fused_rx, fused_ry)

    print(f"\nShots detected: {len(shot_landings)}")
    for idx, s in enumerate(shot_landings):
        start = s['start']
        mid = s['mid']
        end = s['end']
        print(f"  Shot {idx+1}:")
        print(f"    Start  - Frame {start['frame']}: ({start['rx']:.2f}, {start['ry']:.2f}) -> {court_zones.classify_real(start['rx'], start['ry'])}")
        print(f"    Mid    - Frame {mid['frame']}: ({mid['rx']:.2f}, {mid['ry']:.2f}) -> {court_zones.classify_real(mid['rx'], mid['ry'])}")
        print(f"    End    - Frame {end['frame']}: ({end['rx']:.2f}, {end['ry']:.2f}) -> Ball: {s['zone']}")
        if s.get('player_zone'):
            print(f"    Player - ({s['player_pos'][0]:.2f}, {s['player_pos'][1]:.2f}) -> {s['player_zone']}")
        print(f"    FINAL  -> {s['fused_zone']}")

    shot_landings_for_viz = []
    for s in shot_landings:
        shot_landings_for_viz.append({
            'frame': s['end']['frame'],
            'x_coord': s['end']['rx'],
            'y_coord': s['end']['ry'],
            'zone': s['fused_zone'],
        })

    visualizer = CourtVisualizer()
    visualizer.plot_trajectory(ball_trajectory)
    visualizer.plot_shots(ball_trajectory, shot_landings_for_viz)


if __name__ == "__main__":
    main()
