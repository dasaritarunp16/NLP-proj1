import numpy as np
import cv2


def homography(court_keypoints_r):
    
    points = np.array([
        court_keypoints_r[4],
        court_keypoints_r[6],
        court_keypoints_r[5],
        court_keypoints_r[7],
    ], dtype= np.float32)
    
    real_coords = np.array([
        [0,0],
        [23.77,0],
        [0,8.23],
        [23.77,8.23],
        
    ], dtype = np.float32)
    
    H_points = cv2.findHomography(points,real_coords)
    
    return H_points