import cv2
import numpy as np


class CourtZones:
    def __init__(self, court_keypoints_r):
        # court_keypoints_r: (14, 2) array of pixel coordinates
        #
        # Keypoint layout:
        #   0 -------- 4 ----------- 6 -------- 1   <- Far baseline
        #   |  alley   |             |  alley    |
        #   |          8 ---- 12 --- 9           |   <- Far service line
        #   |          |      |      |           |
        #   |          NL --- NC --- NR          |   <- Net (interpolated)
        #   |          |      |      |           |
        #   |          10 --- 13 --- 11          |   <- Near service line
        #   |  alley   |             |  alley    |
        #   2 -------- 5 ----------- 7 -------- 3   <- Near baseline

        kp = court_keypoints_r

        # Interpolate net points (midpoint between far and near service lines)
        net_left = (kp[8] + kp[10]) / 2
        net_right = (kp[9] + kp[11]) / 2
        net_center = (kp[12] + kp[13]) / 2

        # Interpolate baseline center points (for deuce/ad split in backcourt)
        far_baseline_center = (kp[4] + kp[6]) / 2
        near_baseline_center = (kp[5] + kp[7]) / 2

        # Define zones as 4 corners each (clockwise order)
        self.zones = {
            # Service boxes (between service line and net)
            "far_deuce_service_box":  np.array([kp[8], kp[12], net_center, net_left]),
            "far_ad_service_box":     np.array([kp[12], kp[9], net_right, net_center]),
            "near_ad_service_box":    np.array([net_left, net_center, kp[13], kp[10]]),
            "near_deuce_service_box": np.array([net_center, net_right, kp[11], kp[13]]),

            # Backcourt (between baseline and service line)
            "far_deuce_backcourt":    np.array([kp[4], far_baseline_center, kp[12], kp[8]]),
            "far_ad_backcourt":       np.array([far_baseline_center, kp[6], kp[9], kp[12]]),
            "near_ad_backcourt":      np.array([kp[10], kp[13], near_baseline_center, kp[5]]),
            "near_deuce_backcourt":   np.array([kp[13], kp[11], kp[7], near_baseline_center]),

            # Doubles alleys (full length)
            "left_alley":  np.array([kp[0], kp[4], kp[5], kp[2]]),
            "right_alley": np.array([kp[6], kp[1], kp[3], kp[7]]),
        }

    def classify(self, pixel_x, pixel_y):
        point = (float(pixel_x), float(pixel_y))
        for zone_name, corners in self.zones.items():
            contour = corners.reshape(-1, 1, 2).astype(np.float32)
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return zone_name
        return "out"

    @staticmethod
    def classify_real(rx, ry):
        # Classify using real-world coordinates (meters)
        # Court: x=0-10.97 (width), y=0-23.77 (length)
        # Net at y=11.885, service lines at y=5.485 and y=18.285
        # Singles sidelines at x=1.37 and x=9.60, center at x=5.485

        # Clamp to court bounds for classification
        rx_c = max(0, min(rx, 10.97))
        ry_c = max(0, min(ry, 23.77))

        # Alleys
        if rx_c < 1.37:
            return "left_alley"
        if rx_c > 9.60:
            return "right_alley"

        # Deuce (x < 5.485) vs Ad (x >= 5.485)
        side = "deuce" if rx_c < 5.485 else "ad"

        # Far vs Near and backcourt vs service box
        if ry_c < 5.485:
            return f"far_{side}_backcourt"
        elif ry_c < 11.885:
            return f"far_{side}_service_box"
        elif ry_c < 18.285:
            return f"near_{side}_service_box"
        else:
            return f"near_{side}_backcourt"

    def draw_zones(self, frame):
        colors = {
            "far_deuce_service_box":  (255, 200, 200),
            "far_ad_service_box":     (200, 200, 255),
            "near_ad_service_box":    (200, 255, 200),
            "near_deuce_service_box": (255, 255, 200),
            "far_deuce_backcourt":    (255, 150, 150),
            "far_ad_backcourt":       (150, 150, 255),
            "near_ad_backcourt":      (150, 255, 150),
            "near_deuce_backcourt":   (255, 255, 150),
            "left_alley":            (200, 200, 200),
            "right_alley":           (200, 200, 200),
        }
        overlay = frame.copy()
        for zone_name, corners in self.zones.items():
            pts = corners.astype(np.int32)
            color = colors.get(zone_name, (128, 128, 128))
            cv2.fillPoly(overlay, [pts], color)
        # Blend overlay with original frame
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw zone labels
        for zone_name, corners in self.zones.items():
            center_x = int(np.mean(corners[:, 0]))
            center_y = int(np.mean(corners[:, 1]))
            label = zone_name.replace("_", " ")
            cv2.putText(frame, label, (center_x - 60, center_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        return frame
