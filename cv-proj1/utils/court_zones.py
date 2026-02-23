import cv2
import numpy as np


class CourtZones:
    def __init__(self, kp):
        #   0 -------- 4 ----------- 6 -------- 1   far baseline
        #   |          8 ---- 12 --- 9           |   far service line
        #   |          |      |      |           |
        #   |          10 --- 13 --- 11          |   near service line
        #   2 -------- 5 ----------- 7 -------- 3   near baseline

        nl = (kp[8] + kp[10]) / 2
        nr = (kp[9] + kp[11]) / 2
        nc = (kp[12] + kp[13]) / 2
        far_bc = (kp[4] + kp[6]) / 2
        near_bc = (kp[5] + kp[7]) / 2

        self.zones = {
            "far_deuce_service_box":  np.array([kp[8], kp[12], nc, nl]),
            "far_ad_service_box":     np.array([kp[12], kp[9], nr, nc]),
            "near_ad_service_box":    np.array([nl, nc, kp[13], kp[10]]),
            "near_deuce_service_box": np.array([nc, nr, kp[11], kp[13]]),
            "far_deuce_backcourt":    np.array([kp[4], far_bc, kp[12], kp[8]]),
            "far_ad_backcourt":       np.array([far_bc, kp[6], kp[9], kp[12]]),
            "near_ad_backcourt":      np.array([kp[10], kp[13], near_bc, kp[5]]),
            "near_deuce_backcourt":   np.array([kp[13], kp[11], kp[7], near_bc]),
            "left_alley":  np.array([kp[0], kp[4], kp[5], kp[2]]),
            "right_alley": np.array([kp[6], kp[1], kp[3], kp[7]]),
        }

    def classify(self, px, py):
        point = (float(px), float(py))
        for name, corners in self.zones.items():
            contour = corners.reshape(-1, 1, 2).astype(np.float32)
            if cv2.pointPolygonTest(contour, point, False) >= 0:
                return name
        return "out"

    @staticmethod
    def get_zone(rx, ry):
        rx = max(0, min(rx, 10.97))
        ry = max(0, min(ry, 23.77))

        if rx < 1.37:
            return "left_alley"
        if rx > 9.60:
            return "right_alley"

        cx = 5.485
        mid = 1.0

        if ry < 11.885:
            # far court: deuce = low x
            if cx - mid <= rx <= cx + mid:
                side = "middle"
            elif rx < cx:
                side = "deuce"
            else:
                side = "ad"
            return f"far_{side}_backcourt" if ry < 5.485 else f"far_{side}_service_box"
        else:
            # near court: deuce = high x
            if cx - mid <= rx <= cx + mid:
                side = "middle"
            elif rx >= cx:
                side = "deuce"
            else:
                side = "ad"
            return f"near_{side}_service_box" if ry < 18.285 else f"near_{side}_backcourt"

    def draw(self, frame):
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
        for name, corners in self.zones.items():
            pts = corners.astype(np.int32)
            cv2.fillPoly(overlay, [pts], colors.get(name, (128, 128, 128)))
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        for name, corners in self.zones.items():
            cx = int(np.mean(corners[:, 0]))
            cy = int(np.mean(corners[:, 1]))
            cv2.putText(frame, name.replace("_", " "), (cx - 60, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        return frame
