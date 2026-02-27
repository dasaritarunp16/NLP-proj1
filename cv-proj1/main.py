
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils.video import read, save
from utils.player_utils import PT
from utils.ball_utils import BT
from utils.ball_tracker_tracknet import BallTrackerTN
from utils.court_detector_robust import CourtDetector
from utils.homography import homography
from utils.court_zones import CourtZones
from utils.court_visualizer import CourtVisualizer


def main():
    input = "test_video.mp4"
    vid_frames = read(input, start_time=25)

    if len(vid_frames) == 0:
        print(f"ERROR: No frames read from '{input}'.")
        return

    Player_tracker = PT(model = "yolo12n.pt")
    p_detect = Player_tracker.detect_frames(vid_frames)

    ball_tracker = BallTrackerTN(model_path="tracknet_weights.pt")
    court_det = CourtDetector("court_detection_weights.pth")

    kp_interval = 30
    kp_cache = {}
    h_cache = {}
    bnd_cache = {}

    print(f"Running keypoint detection (every {kp_interval} frames)...")
    for fi in range(0, len(vid_frames), kp_interval):
        kp = court_det.predict(vid_frames[fi])
        kp_r = kp.reshape(-1, 2)
        H = homography(kp_r)[0]
        bnd = np.array([kp_r[0], kp_r[1], kp_r[3], kp_r[2]], dtype=np.float32).reshape(-1, 1, 2)
        kp_cache[fi] = kp_r
        h_cache[fi] = H
        bnd_cache[fi] = bnd
    print(f"  Done on {len(kp_cache)} frames")

    kp_idxs = sorted(kp_cache.keys())
    def nearest_kp(fi):
        best = kp_idxs[0]
        for ki in kp_idxs:
            if abs(ki - fi) < abs(best - fi):
                best = ki
        return best

    court_kp = court_det.predict(vid_frames[0])
    court_kp_r = court_kp.reshape(-1, 2)

    b_detect = ball_tracker.detect_frames(vid_frames)

    o_frames = Player_tracker.draw_boxes(vid_frames, p_detect)
    o_frames = ball_tracker.draw_boxes(o_frames, b_detect)
    for fi in range(len(o_frames)):
        nk = nearest_kp(fi)
        o_frames[fi] = court_det.draw_kp(o_frames[fi], kp_cache[nk].flatten())

    save(o_frames, "output_video/output.mp4")

    print(f"Type: {type(court_kp_r[0])}")
    print(f"Shape : {court_kp_r.shape}")
    print(f"Content: \n{court_kp_r}")

    kp_reshaped = court_kp.reshape(-1, 2)
    frame_kp = vid_frames[0].copy()

    for i in range(len(kp_reshaped)):
        x = int(float(kp_reshaped[i, 0]))
        y = int(float(kp_reshaped[i, 1]))

        cv2.circle(frame_kp, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(frame_kp, str(i), (x+10, y-10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(frame_kp, cv2.COLOR_BGR2RGB))
        plt.title("Court Keypoints - Numbered")
        plt.show()

    zones = CourtZones(kp_reshaped)

    traj = []
    ball_count = 0
    out_count = 0
    outside_count = 0
    static_count = 0
    static_thresh = 5
    max_static = 8
    prev_x, prev_y = None, None
    static_run = 0

    for fi, balls in enumerate(b_detect):
        if 1 not in balls:
            continue
        ball_count += 1
        box = balls[1]
        x, y = ball_tracker.ball_center(box)

        if prev_x is not None:
            dist = ((x - prev_x)**2 + (y - prev_y)**2)**0.5
            if dist < static_thresh:
                static_run += 1
            else:
                static_run = 0
        prev_x, prev_y = x, y
        if static_run >= max_static:
            static_count += 1
            continue

        nk = nearest_kp(fi)
        bnd = bnd_cache[nk]
        H = h_cache[nk]

        if cv2.pointPolygonTest(bnd, (float(x), float(y)), False) < 0:
            outside_count += 1
            continue

        pt = np.array([[[x, y]]], dtype=np.float32)
        real = cv2.perspectiveTransform(pt, H)
        rx, ry = real[0][0][0], real[0][0][1]

        if ball_tracker.balls_in_court(rx, ry):
            traj.append({'frame': fi, 'px': x, 'py': y, 'rx': rx, 'ry': ry})
        else:
            out_count += 1
            if out_count <= 10:
                print(f"  OUT: frame {fi} pixel=({x:.0f},{y:.0f}) -> real=({rx:.2f},{ry:.2f})")

    print(f"\nTotal frames: {len(b_detect)}")
    print(f"Ball detected: {ball_count}, Static: {static_count}")
    print(f"Outside court: {outside_count}, Out of bounds: {out_count}")
    print(f"Trajectory pts: {len(traj)}")
    if traj:
        print(f"First: frame {traj[0]['frame']} ({traj[0]['rx']:.2f}, {traj[0]['ry']:.2f})")
        print(f"Last: frame {traj[-1]['frame']} ({traj[-1]['rx']:.2f}, {traj[-1]['ry']:.2f})")

    player_pos = {}
    for fi, players in enumerate(p_detect):
        player_pos[fi] = {}
        nk = nearest_kp(fi)
        H = h_cache[nk]
        bnd = bnd_cache[nk]
        for tid, box in players.items():
            x1, y1, x2, y2 = box
            foot_x = (x1 + x2) / 2
            foot_y = y2
            if cv2.pointPolygonTest(bnd, (float(foot_x), float(foot_y)), False) < 0:
                continue
            pt = np.array([[[foot_x, foot_y]]], dtype=np.float32)
            real = cv2.perspectiveTransform(pt, H)
            rx, ry = float(real[0][0][0]), float(real[0][0][1])
            if ball_tracker.balls_in_court(rx, ry):
                player_pos[fi][tid] = (rx, ry)

    avg_y = {}
    counts = {}
    for fi, players in player_pos.items():
        for tid, (rx, ry) in players.items():
            if tid not in avg_y:
                avg_y[tid] = 0.0
                counts[tid] = 0
            avg_y[tid] += ry
            counts[tid] += 1

    top_players = sorted(counts.keys(), key=lambda t: counts[t], reverse=True)
    near_id, far_id = None, None
    if len(top_players) >= 2:
        p1, p2 = top_players[0], top_players[1]
        y1 = avg_y[p1] / counts[p1]
        y2 = avg_y[p2] / counts[p2]
        if y1 > y2:
            near_id, far_id = p1, p2
        else:
            near_id, far_id = p2, p1
        print(f"Near player: ID {near_id} (avg y={avg_y[near_id]/counts[near_id]:.2f})")
        print(f"Far player: ID {far_id} (avg y={avg_y[far_id]/counts[far_id]:.2f})")

    min_travel = 3.0
    rev_thresh = 2.0
    min_pts = 8
    mid_skip = 2
    max_gap = 12

    shots = []
    pending = None
    start_i = 0

    if len(traj) >= 2:
        extreme = traj[0]
        extreme_i = 0
        to_near = traj[1]['ry'] > traj[0]['ry']

        for i in range(1, len(traj)):
            pt = traj[i]
            prev = traj[i - 1]

            if pt['frame'] - prev['frame'] > max_gap:
                run_len = extreme_i - start_i + 1
                if run_len >= min_pts:
                    if pending is not None:
                        if not shots or abs(pending['end']['ry'] - shots[-1]['end']['ry']) >= min_travel:
                            shots.append(pending)
                    s0 = traj[start_i]
                    mi = min(start_i + mid_skip, extreme_i)
                    pending = {
                        'start': s0, 'mid': traj[mi],
                        'end': extreme, 'zone': zones.get_zone(extreme['rx'], extreme['ry']),
                    }
                start_i = i
                extreme = pt
                extreme_i = i
                if i + 1 < len(traj):
                    to_near = traj[i + 1]['ry'] > pt['ry']
                continue

            if to_near:
                if pt['ry'] >= extreme['ry']:
                    extreme = pt
                    extreme_i = i
                elif extreme['ry'] - pt['ry'] > rev_thresh:
                    run_len = i - start_i + 1
                    if run_len < min_pts:
                        continue
                    if pending is not None:
                        if not shots or abs(pending['end']['ry'] - shots[-1]['end']['ry']) >= min_travel:
                            shots.append(pending)
                    s0 = traj[start_i]
                    mi = min(start_i + mid_skip, extreme_i)
                    pending = {
                        'start': s0, 'mid': traj[mi],
                        'end': extreme, 'zone': zones.get_zone(extreme['rx'], extreme['ry']),
                    }
                    start_i = i
                    extreme = pt
                    extreme_i = i
                    to_near = False
            else:
                if pt['ry'] <= extreme['ry']:
                    extreme = pt
                    extreme_i = i
                elif pt['ry'] - extreme['ry'] > rev_thresh:
                    run_len = i - start_i + 1
                    if run_len < min_pts:
                        continue
                    if pending is not None:
                        if not shots or abs(pending['end']['ry'] - shots[-1]['end']['ry']) >= min_travel:
                            shots.append(pending)
                    s0 = traj[start_i]
                    mi = min(start_i + mid_skip, extreme_i)
                    pending = {
                        'start': s0, 'mid': traj[mi],
                        'end': extreme, 'zone': zones.get_zone(extreme['rx'], extreme['ry']),
                    }
                    start_i = i
                    extreme = pt
                    extreme_i = i
                    to_near = True

    if pending is not None:
        if not shots or abs(pending['end']['ry'] - shots[-1]['end']['ry']) >= min_travel:
            shots.append(pending)

    if len(traj) >= 2:
        s0 = traj[start_i]
        mi = min(start_i + mid_skip, len(traj) - 1)
        last = {
            'start': s0, 'mid': traj[mi],
            'end': extreme, 'zone': zones.get_zone(extreme['rx'], extreme['ry']),
        }
        if not shots or abs(last['end']['ry'] - shots[-1]['end']['ry']) >= min_travel:
            shots.append(last)

    def merge_same_direction(shot_list):
        if len(shot_list) < 2:
            return shot_list
        merged = [shot_list[0]]
        for s in shot_list[1:]:
            prev = merged[-1]
            prev_dir = prev['end']['ry'] - prev['start']['ry']
            curr_dir = s['end']['ry'] - s['start']['ry']
            if (prev_dir > 0 and curr_dir > 0) or (prev_dir < 0 and curr_dir < 0):
                mid_frame = (prev['start']['frame'] + s['end']['frame']) // 2
                mp = min(traj, key=lambda t: abs(t['frame'] - mid_frame))
                merged[-1] = {
                    'start': prev['start'], 'mid': mp,
                    'end': s['end'], 'zone': zones.get_zone(s['end']['rx'], s['end']['ry']),
                }
            else:
                merged.append(s)
        return merged

    shots = merge_same_direction(shots)
    shots = [s for s in shots if s['end']['frame'] - s['start']['frame'] >= 5]
    shots = merge_same_direction(shots)

    search_window = 5
    net_y = 11.885

    for s in shots:
        end_frame = s['end']['frame']
        going_far = s['end']['ry'] < s['start']['ry']

        recv_id = far_id if going_far else near_id
        if recv_id is None:
            s['p_zone'] = None
            s['final_zone'] = s['zone']
            continue

        recv = None
        for off in range(search_window + 1):
            for f in [end_frame + off, end_frame - off]:
                if f in player_pos and recv_id in player_pos[f]:
                    recv = player_pos[f][recv_id]
                    break
            if recv:
                break

        if recv is None:
            s['p_zone'] = None
            s['final_zone'] = s['zone']
            continue

        prx, pry = recv
        pz = zones.get_zone(prx, pry)
        s['p_zone'] = pz
        s['p_pos'] = recv

        ball_zone = s['zone']
        player_closer = abs(pry - net_y) < abs(s['end']['ry'] - net_y)

        if ball_zone == pz:
            s['final_zone'] = ball_zone
        elif player_closer:
            s['final_zone'] = pz
        else:
            fused_x = 0.6 * s['end']['rx'] + 0.4 * prx
            s['final_zone'] = zones.get_zone(fused_x, s['end']['ry'])

    print(f"\nShots detected: {len(shots)}")
    for idx, s in enumerate(shots):
        st, md, en = s['start'], s['mid'], s['end']
        print(f"  Shot {idx+1}:")
        print(f"    Start  - Frame {st['frame']}: ({st['rx']:.2f}, {st['ry']:.2f}) -> {zones.get_zone(st['rx'], st['ry'])}")
        print(f"    Mid    - Frame {md['frame']}: ({md['rx']:.2f}, {md['ry']:.2f}) -> {zones.get_zone(md['rx'], md['ry'])}")
        print(f"    End    - Frame {en['frame']}: ({en['rx']:.2f}, {en['ry']:.2f}) -> Ball: {s['zone']}")
        if s.get('p_zone'):
            print(f"    Player - ({s['p_pos'][0]:.2f}, {s['p_pos'][1]:.2f}) -> {s['p_zone']}")
        print(f"    FINAL  -> {s['final_zone']}")

    viz_shots = []
    for s in shots:
        viz_shots.append({
            'frame': s['end']['frame'],
            'x_coord': s['end']['rx'],
            'y_coord': s['end']['ry'],
            'zone': s['final_zone'],
        })

    vis = CourtVisualizer()
    vis.plot_trajectory(traj)
    vis.plot_shots(traj, viz_shots)


if __name__ == "__main__":
    main()
