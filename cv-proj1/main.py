
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
    vid_frames = read(input, start_time=25)  # start at 0:25

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

    # --- Build player trajectories in real-world coords ---
    # Use foot position (bottom-center of bbox) + homography
    # player_positions[frame_idx] = {track_id: (rx, ry), ...}
    player_positions = {}
    for frame_idx, players in enumerate(p_detect):
        player_positions[frame_idx] = {}
        for track_id, box in players.items():
            x1, y1, x2, y2 = box
            foot_x = (x1 + x2) / 2
            foot_y = y2  # bottom of bounding box = feet
            # Only use players inside the court polygon
            if cv2.pointPolygonTest(court_boundary, (float(foot_x), float(foot_y)), False) < 0:
                continue
            foot_frame = np.array([[[foot_x, foot_y]]], dtype=np.float32)
            foot_real = cv2.perspectiveTransform(foot_frame, H_points)
            rx, ry = float(foot_real[0][0][0]), float(foot_real[0][0][1])
            if ball_tracker.balls_in_court(rx, ry):
                player_positions[frame_idx][track_id] = (rx, ry)

    # Identify near player and far player by average y-position
    # Near player: avg ry > 11.885 (near half), Far player: avg ry < 11.885
    player_avg_y = {}
    player_counts = {}
    for frame_idx, players in player_positions.items():
        for track_id, (rx, ry) in players.items():
            if track_id not in player_avg_y:
                player_avg_y[track_id] = 0.0
                player_counts[track_id] = 0
            player_avg_y[track_id] += ry
            player_counts[track_id] += 1

    # Pick the two most-seen players
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

    # --- Shot detection: 3-point trajectory per shot ---
    # For each shot, capture: start (after last reversal), mid (a few frames in),
    # and end (furthest y-point before next reversal).
    # Only confirm a shot once the next direction change validates it.
    MIN_Y_TRAVEL = 3.0        # minimum y-distance between consecutive shots
    REVERSAL_THRESHOLD = 2.0  # ball must move this far back before we confirm reversal
    MIN_PASS_POINTS = 8       # minimum trajectory points for a pass to count as a real shot
    MID_SKIP = 2              # frames to skip from start to capture mid-flight position
    MAX_FRAME_GAP = 12        # if gap between consecutive detections exceeds this, force a pass break

    shot_landings = []
    pending_shot = None       # held until next reversal confirms it
    pass_start_idx = 0        # index where the current pass began

    if len(ball_trajectory) >= 2:
        extreme_pt = ball_trajectory[0]
        extreme_idx = 0
        going_near = ball_trajectory[1]['ry'] > ball_trajectory[0]['ry']

        for i in range(1, len(ball_trajectory)):
            pt = ball_trajectory[i]
            prev_pt = ball_trajectory[i - 1]

            # Force a pass break if large frame gap (tracker lost the ball)
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
                        'start': start_pt,
                        'mid': mid_pt,
                        'end': end_pt,
                        'zone': end_zone,
                    }
                # Reset pass — re-determine direction from this point
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
                    pass_length = i - pass_start_idx + 1
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

    # --- Filter out shots that are too short (noise) ---
    MIN_SHOT_FRAMES = 5  # shot must span at least this many video frames
    shot_landings = [s for s in shot_landings
                     if s['end']['frame'] - s['start']['frame'] >= MIN_SHOT_FRAMES]

    # --- Second merge pass: filtering may have exposed new same-direction pairs ---
    if len(shot_landings) >= 2:
        merged = [shot_landings[0]]
        for s in shot_landings[1:]:
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
                    'start': prev['start'],
                    'mid': mid_pt,
                    'end': s['end'],
                    'zone': end_zone,
                }
            else:
                merged.append(s)
        shot_landings = merged

    # --- Player-ball fusion: validate landing zones with receiving player position ---
    # The receiving player runs to where the ball lands, so their foot position
    # at the shot's end frame provides a second opinion on the landing zone.
    PLAYER_SEARCH_WINDOW = 5  # search +/- frames around end frame for player position
    BALL_WEIGHT = 0.6         # weight for ball position in fusion
    PLAYER_WEIGHT = 0.4       # weight for player position in fusion

    for s in shot_landings:
        end_frame = s['end']['frame']
        ball_going_far = s['end']['ry'] < s['start']['ry']

        # Determine receiving player: far player if ball going far, near player if going near
        receiver_id = far_player_id if ball_going_far else near_player_id
        if receiver_id is None:
            s['player_zone'] = None
            s['fused_zone'] = s['zone']
            continue

        # Find receiver's position near the end frame (search a window)
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

        # Fuse: mandatory rule + fallback
        # Rule 1: If the player is closer to the net than the ball, the player
        #         intercepted the ball there — use the player's position entirely.
        # Rule 2: Otherwise, player informs x (deuce/ad), ball informs y (depth).
        NET_Y = 11.885
        ball_zone = s['zone']
        player_closer_to_net = abs(p_ry - NET_Y) < abs(s['end']['ry'] - NET_Y)

        if ball_zone == player_zone:
            s['fused_zone'] = ball_zone
        elif player_closer_to_net:
            # Player is closer to net than ball — they got to the ball here
            s['fused_zone'] = player_zone
        else:
            # Player is deeper — use player's x (deuce/ad) + ball's y (depth)
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

    # Build shot_landings in the format plot_shots expects (using end point + fused zone)
    shot_landings_for_viz = []
    for s in shot_landings:
        shot_landings_for_viz.append({
            'frame': s['end']['frame'],
            'x_coord': s['end']['rx'],
            'y_coord': s['end']['ry'],
            'zone': s['fused_zone'],
        })

    # Visualize ball trajectory on 2D court
    visualizer = CourtVisualizer()
    visualizer.plot_trajectory(ball_trajectory)
    visualizer.plot_shots(ball_trajectory, shot_landings_for_viz)


if __name__ == "__main__":
    main()
