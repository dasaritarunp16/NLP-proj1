import numpy as np
import cv2


def homography(court_keypoints_r):
    # All 14 keypoints mapped to real-world tennis court coordinates (meters)
    # x-axis: court width (0 = left doubles sideline, 10.97 = right doubles sideline)
    # y-axis: court length (0 = far baseline, 23.77 = near baseline)
    #
    # Doubles alley width: 1.37m each side
    # Singles sideline: 1.37m from doubles sideline
    # Service line: 5.485m from each baseline (6.40m from net)
    # Center of court width: 5.485m

    points = np.array([
        court_keypoints_r[0],   # far baseline, left doubles
        court_keypoints_r[1],   # far baseline, right doubles
        court_keypoints_r[2],   # near baseline, left doubles
        court_keypoints_r[3],   # near baseline, right doubles
        court_keypoints_r[4],   # far baseline, left singles
        court_keypoints_r[5],   # near baseline, left singles
        court_keypoints_r[6],   # far baseline, right singles
        court_keypoints_r[7],   # near baseline, right singles
        court_keypoints_r[8],   # far service line, left singles
        court_keypoints_r[9],   # far service line, right singles
        court_keypoints_r[10],  # near service line, left singles
        court_keypoints_r[11],  # near service line, right singles
        court_keypoints_r[12],  # far service line, center
        court_keypoints_r[13],  # near service line, center
    ], dtype=np.float32)

    real_coords = np.array([
        [0, 0],            # 0:  far baseline, left doubles
        [10.97, 0],        # 1:  far baseline, right doubles
        [0, 23.77],        # 2:  near baseline, left doubles
        [10.97, 23.77],    # 3:  near baseline, right doubles
        [1.37, 0],         # 4:  far baseline, left singles
        [1.37, 23.77],     # 5:  near baseline, left singles
        [9.60, 0],         # 6:  far baseline, right singles
        [9.60, 23.77],     # 7:  near baseline, right singles
        [1.37, 5.485],     # 8:  far service line, left singles
        [9.60, 5.485],     # 9:  far service line, right singles
        [1.37, 18.285],    # 10: near service line, left singles
        [9.60, 18.285],    # 11: near service line, right singles
        [5.485, 5.485],    # 12: far service line, center
        [5.485, 18.285],   # 13: near service line, center
    ], dtype=np.float32)

    H_points = cv2.findHomography(points, real_coords)

    return H_points
