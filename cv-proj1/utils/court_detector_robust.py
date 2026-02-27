import torch
import cv2
import numpy as np
from utils.tracknet import BallTrackerNet


class CourtDetector:
    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = BallTrackerNet(in_channels=3, out_channels=15)

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                weights = ckpt['model_state_dict']
            elif 'state_dict' in ckpt:
                weights = ckpt['state_dict']
            elif 'model' in ckpt:
                weights = ckpt['model']
            else:
                weights = ckpt
        else:
            weights = ckpt
        self.model.load_state_dict(weights)
        self.model.to(self.device)
        self.model.eval()
        self.w = 640
        self.h = 360

    def predict(self, image):
        oh, ow = image.shape[:2]
        resized = cv2.resize(image, (self.w, self.h))
        inp = resized.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(inp)

        kp_map = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        raw = np.zeros((14, 2))
        for k in range(14):
            mask = np.zeros_like(kp_map)
            mask[kp_map == k] = 255
            circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                       param1=50, param2=2, minRadius=10, maxRadius=25)
            if circles is not None and len(circles[0]) > 0:
                x, y, _ = circles[0][0]
                raw[k] = [x * (ow / self.w), y * (oh / self.h)]
            else:
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    raw[k] = [np.mean(coords[1]) * (ow / self.w),
                              np.mean(coords[0]) * (oh / self.h)]

        # swap corners 2 and 3 (model outputs clockwise, we need TL TR BL BR)
        kp = np.zeros((14, 2))
        kp[0] = raw[0]
        kp[1] = raw[1]
        kp[2] = raw[3]
        kp[3] = raw[2]
        kp[4:] = raw[4:]

        # if near-court doubles corners are missing or on the wrong side, estimate them
        cx = ow / 2.0
        if kp[2][0] == 0 or kp[2][0] > cx:
            if kp[5][0] > 0 and kp[4][0] > 0 and kp[0][0] > 0:
                off = kp[4][0] - kp[0][0]
                kp[2] = [kp[5][0] - off, kp[5][1]]
            elif kp[5][0] > 0:
                kp[2] = [kp[5][0] * 0.85, kp[5][1]]

        if kp[3][0] == 0 or kp[3][0] < cx:
            if kp[7][0] > 0 and kp[6][0] > 0 and kp[1][0] > 0:
                off = kp[1][0] - kp[6][0]
                kp[3] = [kp[7][0] + off, kp[7][1]]
            elif kp[7][0] > 0:
                kp[3] = [kp[7][0] * 1.15, kp[7][1]]

        return kp.flatten()

    def draw_kp(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            cv2.putText(image, str(i // 2), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_kp_video(self, frames, keypoints):
        out = []
        for frame in frames:
            frame = self.draw_kp(frame, keypoints)
            out.append(frame)
        return out
