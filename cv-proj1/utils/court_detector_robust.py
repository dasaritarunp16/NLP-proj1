import torch
import cv2
import numpy as np
from utils.tracknet import BallTrackerNet


class CourtDetectorRobust:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = BallTrackerNet(in_channels=3, out_channels=15)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.width = 640
        self.height = 360

    def predict(self, image):
        orig_h, orig_w = image.shape[:2]

        resized = cv2.resize(image, (self.width, self.height))
        inp = resized.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)
        inp = inp.to(self.device)

        with torch.no_grad():
            output = self.model(inp)

        keypoint_map = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        raw_kp = np.zeros((14, 2))
        for k in range(14):
            mask = np.zeros_like(keypoint_map)
            mask[keypoint_map == k] = 255

            circles = cv2.HoughCircles(
                mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                param1=50, param2=2, minRadius=10, maxRadius=25
            )

            if circles is not None and len(circles[0]) > 0:
                x, y, _ = circles[0][0]
                raw_kp[k] = [x * (orig_w / self.width), y * (orig_h / self.height)]
            else:
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    y_mean = np.mean(coords[0])
                    x_mean = np.mean(coords[1])
                    raw_kp[k] = [x_mean * (orig_w / self.width), y_mean * (orig_h / self.height)]

        # remap: model uses clockwise corners (TL,TR,BR,BL), we need (TL,TR,BL,BR)
        kp = np.zeros((14, 2))
        kp[0] = raw_kp[0]
        kp[1] = raw_kp[1]
        kp[2] = raw_kp[3]   # swap 2<->3
        kp[3] = raw_kp[2]
        kp[4:] = raw_kp[4:]

        # fix missing near-court doubles corners
        center_x = orig_w / 2.0

        if kp[2][0] == 0 or kp[2][0] > center_x:
            if kp[5][0] > 0 and kp[4][0] > 0 and kp[0][0] > 0:
                alley_offset = kp[4][0] - kp[0][0]
                kp[2] = [kp[5][0] - alley_offset, kp[5][1]]
            elif kp[5][0] > 0:
                kp[2] = [kp[5][0] * 0.85, kp[5][1]]

        if kp[3][0] == 0 or kp[3][0] < center_x:
            if kp[7][0] > 0 and kp[6][0] > 0 and kp[1][0] > 0:
                alley_offset = kp[1][0] - kp[6][0]
                kp[3] = [kp[7][0] + alley_offset, kp[7][1]]
            elif kp[7][0] > 0:
                kp[3] = [kp[7][0] * 1.15, kp[7][1]]

        return kp.flatten()

    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(image, str(i // 2), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
