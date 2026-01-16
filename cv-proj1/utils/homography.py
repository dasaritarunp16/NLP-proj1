import numpy as np
import cv2


def homography(court_keypoints_r):
    points = np.array([
        court_keypoints_r[0],
        court_keypoints_r[1],
        court_keypoints_r[2],
        court_keypoints_r[3],
    ], dtype= np.float32)
    
    real_coords = np.array([
        [0,0],
        [23.77,0],
        [0,10.97],
        [23.77,10.97],
        
    ], dtype = np.float32)
    
    H_points = cv2.findHomography(points,real_coords)
    
    return H_points