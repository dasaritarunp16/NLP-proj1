import cv2
import pickle
import numpy as np
from ultralytics import YOLO

class BT:
    def __init__ (self, model):
        self.model = YOLO(model)
        
    def detect_frame(self, frame):
        results = self.model.track(frame, conf = 0.15,persist= True)[0]
    
        ball_list = {}
        for i in results.boxes:
        
            result = i.xyxy.tolist()[0]
            
            ball_list[1] = result
            
        return ball_list
    
    def detect_frames(self, frames, read_from_stub = False, stub_path=None ):
            ball_detections = []
            if read_from_stub and stub_path is not None:
                with open(stub_path, 'rb') as f:
                    ball_detections = pickle.load(f)
                    return ball_detections
            for frame in frames:
                list = self.detect_frame(frame)
                ball_detections.append(list)
            if stub_path is not None:
                with open(stub_path, 'rb') as f:
                    pickle.dump(ball_detections, f)
            return ball_detections
    def draw_boxes(self, v_frames, b_detect):
        o_frames = []
        for frame, balls in zip(v_frames, b_detect):
            for t_id, box in balls.items():
                x1, y1, x2, y2 = box
                cv2.putText(frame, f"Ball ID: {t_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)
            o_frames.append(frame)
        return o_frames
    def ball_center(self, box):
        x1, y1, x2, y2 = box
        cX = (x1 + x2) / 2
        cY = (y1 + y2) / 2
        return cX, cY 
    
    def balls_in_court(self, x, y):
        return (0 <= x <= 10.97) and (0 <= y <= 23.77)

    def classify_shot(self, x, y):
        # Court dimensions (meters)
        # x: 0 = left doubles sideline, 10.97 = right doubles sideline
        # y: 0 = far baseline, 23.77 = near baseline
        #
        # Key lines:
        #   Left doubles sideline:  x = 0
        #   Left singles sideline:  x = 1.37
        #   Center line:            x = 5.485
        #   Right singles sideline: x = 9.60
        #   Right doubles sideline: x = 10.97
        #
        #   Far baseline:           y = 0
        #   Far service line:       y = 5.485
        #   Net:                    y = 11.885
        #   Near service line:      y = 18.285
        #   Near baseline:          y = 23.77

        if not self.balls_in_court(x, y):
            return "out"

        # Which side of net
        if y <= 11.885:
            side = "far"
        else:
            side = "near"

        # Doubles alley check
        if x < 1.37 or x > 9.60:
            return f"{side}_alley"

        # Service box (between service line and net)
        if 5.485 <= y <= 11.885:
            # Far side service boxes
            if x < 5.485:
                return "far_deuce_service_box"
            else:
                return "far_ad_service_box"
        elif 11.885 < y <= 18.285:
            # Near side service boxes
            if x < 5.485:
                return "near_ad_service_box"
            else:
                return "near_deuce_service_box"

        # Backcourt (between baseline and service line)
        if y < 5.485:
            # Far backcourt
            if x < 5.485:
                return "far_deuce_backcourt"
            else:
                return "far_ad_backcourt"
        else:
            # Near backcourt (y > 18.285)
            if x < 5.485:
                return "near_ad_backcourt"
            else:
                return "near_deuce_backcourt"

    def detect_bounces(self, b_detect, min_frames_between=5):
        # Extract ball pixel y-positions (bottom of bounding box = closest to ground)
        # When ball bounces, its pixel y reaches a local maximum (lowest point in image)
        positions = []
        for frame_idx, balls in enumerate(b_detect):
            if 1 in balls:
                box = balls[1]
                _, bottom_y = self.ball_center(box)
                positions.append((frame_idx, bottom_y))

        if len(positions) < 3:
            return []

        bounces = []
        for i in range(1, len(positions) - 1):
            prev_frame, prev_y = positions[i - 1]
            curr_frame, curr_y = positions[i]
            next_frame, next_y = positions[i + 1]

            # Skip if frames are too far apart (ball lost tracking)
            if curr_frame - prev_frame > 5 or next_frame - curr_frame > 5:
                continue

            # Local maximum in pixel y = ball at lowest point = bounce
            if curr_y > prev_y and curr_y > next_y:
                # Check minimum distance from last bounce
                if len(bounces) == 0 or curr_frame - bounces[-1]['frame'] >= min_frames_between:
                    bounces.append({
                        'frame': curr_frame,
                        'pixel_y': curr_y,
                    })

        return bounces

    def detect_hits(self, b_detect, speed_threshold=2.0, min_frames_between=10):
        # Detect when the ball is hit by a player
        # At moment of contact, the ball's velocity changes drastically:
        # - sudden speed spike (ball accelerates off the racket)
        # - sharp direction change (ball reverses or changes angle)
        #
        # We track pixel-space positions and compute speed between frames.
        # A hit is detected when speed jumps significantly compared to
        # the average speed in surrounding frames.

        positions = []
        for frame_idx, balls in enumerate(b_detect):
            if 1 in balls:
                box = balls[1]
                cx, cy = self.ball_center(box)
                positions.append((frame_idx, cx, cy))

        if len(positions) < 5:
            return []

        # Compute speeds between consecutive detected positions
        speeds = []
        for i in range(1, len(positions)):
            prev_frame, prev_x, prev_y = positions[i - 1]
            curr_frame, curr_x, curr_y = positions[i]

            frame_gap = curr_frame - prev_frame
            if frame_gap == 0 or frame_gap > 5:
                speeds.append(None)
                continue

            dist = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            speed = dist / frame_gap
            speeds.append(speed)

        # Detect sudden speed spikes
        hits = []
        window = 3  # frames to average for baseline speed
        for i in range(window, len(speeds) - 1):
            if speeds[i] is None:
                continue

            # Compute average speed in the window before this frame
            prev_speeds = [s for s in speeds[i-window:i] if s is not None]
            if len(prev_speeds) == 0:
                continue

            avg_speed = np.mean(prev_speeds)

            # Hit detected if current speed is significantly higher than recent average
            if avg_speed > 0 and speeds[i] / avg_speed >= speed_threshold:
                frame_idx = positions[i + 1][0]  # +1 because speeds is offset by 1
                # Check minimum distance from last hit
                if len(hits) == 0 or frame_idx - hits[-1]['frame'] >= min_frames_between:
                    hits.append({
                        'frame': frame_idx,
                        'speed': speeds[i],
                        'avg_speed_before': avg_speed,
                        'speed_ratio': speeds[i] / avg_speed,
                    })

        return hits