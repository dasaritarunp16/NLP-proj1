
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

    # predict keypoints periodically for camera movement
    KP_INTERVAL = 30
    kp_cache = {}
    h_cache = {}
    bnd_cache = {}

    print(f"Running keypoint detection (every {KP_INTERVAL} frames)...")
    for fi in range(0, len(vid_frames), KP_INTERVAL):
        kp = court_det.predict(vid_frames[fi])
        kp_r = kp.reshape(-1, 2)
        H = homography(kp_r)[0]
        bnd = np.array([
            kp_r[0], kp_r[1], kp_r[3], kp_r[2],
        ], dtype=np.float32).reshape(-1, 1, 2)
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

    # build ball trajectory
    traj = []
    n_ball = 0
    n_out = 0
    n_outside = 0
    n_static = 0
    STATIC_THRESH = 5
    MAX_STATIC = 8
    prev_px, prev_py = None, None
    static_cnt = 0

    for fi, balls in enumerate(b_detect):
        if 1 in balls:
            n_ball += 1
            box = balls[1]
            x, y = ball_tracker.ball_center(box)

            if prev_px is not None:
                dist = ((x - prev_px)**2 + (y - prev_py)**2)**0.5
                if dist < STATIC_THRESH:
                    static_cnt += 1
                else:
                    static_cnt = 0
            prev_px, prev_py = x, y
            if static_cnt >= MAX_STATIC:
                n_static += 1
                continue

            nk = nearest_kp(fi)
            bnd = bnd_cache[nk]
            H = h_cache[nk]

            if cv2.pointPolygonTest(bnd, (float(x), float(y)), False) < 0:
                n_outside += 1
                continue

            pt = np.array([[[x, y]]], dtype=np.float32)
            real = cv2.perspectiveTransform(pt, H)
            rx, ry = real[0][0][0], real[0][0][1]

            if ball_tracker.balls_in_court(rx, ry):
                traj.append({'frame': fi, 'px': x, 'py': y, 'rx': rx, 'ry': ry})
            else:
                n_out += 1
                if n_out <= 10:
                    print(f"  OUT: frame {fi} pixel=({x:.0f},{y:.0f}) -> real=({rx:.2f},{ry:.2f})")

    print(f"\n--- DEBUG ---")
    print(f"Total frames: {len(b_detect)}")
    print(f"Ball detected: {n_ball}, Static: {n_static}")
    print(f"Outside court: {n_outside}, Out of bounds: {n_out}")
    print(f"Trajectory pts: {len(traj)}")
    if len(traj) > 0:
        print(f"First: frame {traj[0]['frame']} ({traj[0]['rx']:.2f}, {traj[0]['ry']:.2f})")
        print(f"Last: frame {traj[-1]['frame']} ({traj[-1]['rx']:.2f}, {traj[-1]['ry']:.2f})")
    print(f"--- END DEBUG ---\n")

    # build player positions (foot position -> real coords)
    p_pos = {}
    for fi, players in enumerate(p_detect):
        p_pos[fi] = {}
        nk = nearest_kp(fi)
        H = h_cache[nk]
        bnd = bnd_cache[nk]
        for tid, box in players.items():
            x1, y1, x2, y2 = box
            fx = (x1 + x2) / 2
            fy = y2
            if cv2.pointPolygonTest(bnd, (float(fx), float(fy)), False) < 0:
                continue
            pt = np.array([[[fx, fy]]], dtype=np.float32)
            real = cv2.perspectiveTransform(pt, H)
            rx, ry = float(real[0][0][0]), float(real[0][0][1])
            if ball_tracker.balls_in_court(rx, ry):
                p_pos[fi][tid] = (rx, ry)

    # figure out who's near vs far
    avg_y = {}
    counts = {}
    for fi, players in p_pos.items():
        for tid, (rx, ry) in players.items():
            if tid not in avg_y:
                avg_y[tid] = 0.0
                counts[tid] = 0
            avg_y[tid] += ry
            counts[tid] += 1

    top_players = sorted(counts.keys(), key=lambda t: counts[t], reverse=True)
    near_id = None
    far_id = None
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

    # shot detection
    MIN_TRAVEL = 3.0
    REV_THRESH = 2.0
    MIN_PTS = 8
    MID_SKIP = 2
    MAX_GAP = 12

    shots = []
    pending = None
    start_i = 0

    if len(traj) >= 2:
        ext_pt = traj[0]
        ext_i = 0
        to_near = traj[1]['ry'] > traj[0]['ry']

        for i in range(1, len(traj)):
            pt = traj[i]
            prev = traj[i - 1]

            if pt['frame'] - prev['frame'] > MAX_GAP:
                plen = ext_i - start_i + 1
                if plen >= MIN_PTS:
                    if pending is not None:
                        if len(shots) == 0 or abs(pending['end']['ry'] - shots[-1]['end']['ry']) >= MIN_TRAVEL:
                            shots.append(pending)
                    s0 = traj[start_i]
                    mi = min(start_i + MID_SKIP, ext_i)
                    pending = {
                        'start': s0, 'mid': traj[mi],
                        'end': ext_pt, 'zone': zones.get_zone(ext_pt['rx'], ext_pt['ry']),
                    }
                start_i = i
                ext_pt = pt
                ext_i = i
                if i + 1 < len(traj):
                    to_near = traj[i + 1]['ry'] > pt['ry']
                continue

            if to_near:
                if pt['ry'] >= ext_pt['ry']:
                    ext_pt = pt
                    ext_i = i
                elif ext_pt['ry'] - pt['ry'] > REV_THRESH:
                    plen = i - start_i + 1
                    if plen < MIN_PTS:
                        continue

                    if pending is not None:
                        if len(shots) == 0 or abs(pending['end']['ry'] - shots[-1]['end']['ry']) >= MIN_TRAVEL:
                            shots.append(pending)

                    s0 = traj[start_i]
                    mi = min(start_i + MID_SKIP, ext_i)
                    pending = {
                        'start': s0, 'mid': traj[mi],
                        'end': ext_pt, 'zone': zones.get_zone(ext_pt['rx'], ext_pt['ry']),
                    }
                    start_i = i
                    ext_pt = pt
                    ext_i = i
                    to_near = False
            else:
                if pt['ry'] <= ext_pt['ry']:
                    ext_pt = pt
                    ext_i = i
                elif pt['ry'] - ext_pt['ry'] > REV_THRESH:
                    plen = i - start_i + 1
                    if plen < MIN_PTS:
                        continue

                    if pending is not None:
                        if len(shots) == 0 or abs(pending['end']['ry'] - shots[-1]['end']['ry']) >= MIN_TRAVEL:
                            shots.append(pending)

                    s0 = traj[start_i]
                    mi = min(start_i + MID_SKIP, ext_i)
                    pending = {
                        'start': s0, 'mid': traj[mi],
                        'end': ext_pt, 'zone': zones.get_zone(ext_pt['rx'], ext_pt['ry']),
                    }
                    start_i = i
                    ext_pt = pt
                    ext_i = i
                    to_near = True

    if pending is not None:
        if len(shots) == 0 or abs(pending['end']['ry'] - shots[-1]['end']['ry']) >= MIN_TRAVEL:
            shots.append(pending)

    if len(traj) >= 2:
        s0 = traj[start_i]
        mi = min(start_i + MID_SKIP, len(traj) - 1)
        last = {
            'start': s0, 'mid': traj[mi],
            'end': ext_pt, 'zone': zones.get_zone(ext_pt['rx'], ext_pt['ry']),
        }
        if len(shots) == 0 or abs(last['end']['ry'] - shots[-1]['end']['ry']) >= MIN_TRAVEL:
            shots.append(last)

    def merge_dir(shots_list):
        if len(shots_list) < 2:
            return shots_list
        merged = [shots_list[0]]
        for s in shots_list[1:]:
            p = merged[-1]
            pd = p['end']['ry'] - p['start']['ry']
            cd = s['end']['ry'] - s['start']['ry']
            if (pd > 0 and cd > 0) or (pd < 0 and cd < 0):
                tf = (p['start']['frame'] + s['end']['frame']) // 2
                mp = min(traj, key=lambda t: abs(t['frame'] - tf))
                merged[-1] = {
                    'start': p['start'], 'mid': mp,
                    'end': s['end'], 'zone': zones.get_zone(s['end']['rx'], s['end']['ry']),
                }
            else:
                merged.append(s)
        return merged

    shots = merge_dir(shots)

    MIN_FRAMES = 5
    shots = [s for s in shots if s['end']['frame'] - s['start']['frame'] >= MIN_FRAMES]

    shots = merge_dir(shots)

    # fuse ball + player positions for zone validation
    SEARCH_WIN = 5
    B_W = 0.6
    P_W = 0.4
    NET_Y = 11.885

    for s in shots:
        ef = s['end']['frame']
        going_far = s['end']['ry'] < s['start']['ry']

        recv_id = far_id if going_far else near_id
        if recv_id is None:
            s['p_zone'] = None
            s['final_zone'] = s['zone']
            continue

        recv = None
        for off in range(0, SEARCH_WIN + 1):
            for f in [ef + off, ef - off]:
                if f in p_pos and recv_id in p_pos[f]:
                    recv = p_pos[f][recv_id]
                    break
            if recv is not None:
                break

        if recv is None:
            s['p_zone'] = None
            s['final_zone'] = s['zone']
            continue

        prx, pry = recv
        pz = zones.get_zone(prx, pry)
        s['p_zone'] = pz
        s['p_pos'] = recv

        bz = s['zone']
        closer = abs(pry - NET_Y) < abs(s['end']['ry'] - NET_Y)

        if bz == pz:
            s['final_zone'] = bz
        elif closer:
            s['final_zone'] = pz
        else:
            fused_rx = B_W * s['end']['rx'] + P_W * prx
            s['final_zone'] = zones.get_zone(fused_rx, s['end']['ry'])

    print(f"\nShots detected: {len(shots)}")
    for idx, s in enumerate(shots):
        st = s['start']
        md = s['mid']
        en = s['end']
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
