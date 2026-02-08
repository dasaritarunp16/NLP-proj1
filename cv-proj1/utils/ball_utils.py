import cv2
import pickle
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