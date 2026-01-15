import cv2
import pickle
from ultralytics import YOLO

class PT:
    def __init__ (self, model):
        self.model = YOLO(model)
        
    def detect_frame(self, frame):
        results = self.model.track(frame, persist = True)[0]
        cm = results.names
        player_list = {}
        for i in results.boxes:
            if i.id is None:  # Add this check
                continue
            t_id = int(i.id.tolist()[0])
            result = i.xyxy.tolist()[0]
            c_ids = i.cls.tolist()[0]
            detect_names = cm[c_ids]
            if detect_names == "person":
                player_list[t_id] = result
        return player_list
    
    def detect_frames(self, frames, read_from_stub = False, stub_path=None ):
            player_detections = []
            if read_from_stub and stub_path is not None:
                with open(stub_path, 'rb') as f:
                    player_detections = pickle.load(f)
                    return player_detections
            for frame in frames:
                list = self.detect_frame(frame)
                player_detections.append(list)
            if stub_path is not None:
                with open(stub_path, 'rb') as f:
                    pickle.dump(player_detections, f)
            return player_detections
    def draw_boxes(self, v_frames, p_detect):
        o_frames = []
        for frame, players in zip(v_frames, p_detect):
            for t_id, box in players.items():
                x1, y1, x2, y2 = box
                cv2.putText(frame, f"Player ID: {t_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
            o_frames.append(frame)
        return o_frames
            