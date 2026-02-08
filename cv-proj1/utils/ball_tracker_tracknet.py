import cv2
import numpy as np
import torch
import pickle
from utils.tracknet import BallTrackerNet


class BallTrackerTN:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = BallTrackerNet()
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.to(self.device)
        self.model.eval()
        self.width = 640
        self.height = 360

    def _preprocess(self, frame):
        resized = cv2.resize(frame, (self.width, self.height))
        return resized.astype(np.float32) / 255.0

    def detect_frame(self, frames_triplet):
        # frames_triplet: list of 3 consecutive BGR frames
        processed = [self._preprocess(f) for f in frames_triplet]
        stacked = np.concatenate(processed, axis=2)  # (360, 640, 9)
        inp = torch.from_numpy(stacked).permute(2, 0, 1).unsqueeze(0)
        inp = inp.to(self.device)

        with torch.no_grad():
            output = self.model(inp)

        heatmap = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        _, binary = cv2.threshold(heatmap, 128, 255, cv2.THRESH_BINARY)

        circles = cv2.HoughCircles(
            binary, cv2.HOUGH_GRADIENT, dp=1, minDist=1,
            param1=50, param2=2, minRadius=2, maxRadius=7
        )

        if circles is not None and len(circles[0]) > 0:
            x, y, r = circles[0][0]
            orig_h, orig_w = frames_triplet[0].shape[:2]
            x_orig = x * (orig_w / self.width)
            y_orig = y * (orig_h / self.height)
            r_scaled = max(r * (orig_w / self.width), 5)
            x1 = x_orig - r_scaled
            y1 = y_orig - r_scaled
            x2 = x_orig + r_scaled
            y2 = y_orig + r_scaled
            return {1: [x1, y1, x2, y2]}

        return {}

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        ball_detections = []
        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
                return ball_detections

        for i in range(len(frames)):
            if i < 2:
                # Not enough context frames yet
                ball_detections.append({})
            else:
                triplet = [frames[i - 2], frames[i - 1], frames[i]]
                det = self.detect_frame(triplet)
                ball_detections.append(det)

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def draw_boxes(self, v_frames, b_detect):
        o_frames = []
        for frame, balls in zip(v_frames, b_detect):
            for t_id, box in balls.items():
                x1, y1, x2, y2 = box
                cv2.putText(frame, f"Ball ID: {t_id}", (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0, 255, 255), 2)
            o_frames.append(frame)
        return o_frames

    def ball_center(self, box):
        x1, y1, x2, y2 = box
        cX = (x1 + x2) / 2
        cY = (y1 + y2) / 2
        return cX, cY

    def balls_in_court(self, x, y, margin=3.0):
        return (-margin <= x <= 10.97 + margin) and (-margin <= y <= 23.77 + margin)
