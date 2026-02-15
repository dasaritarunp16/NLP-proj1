import torch
import cv2
import numpy as np
from utils.tracknet import BallTrackerNet


class CourtDetectorRobust:
    """Court keypoint detector using heatmap-based TennisCourtDetector model.

    Uses a BallTrackerNet (VGG-16 encoder-decoder) with in_channels=3, out_channels=15
    trained on 8,841 tennis court images across multiple surfaces and camera angles.
    Outputs 14 court keypoints (same ordering as CLD) via heatmap peak extraction.
    """

    def __init__(self, model_path, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = BallTrackerNet(in_channels=3, out_channels=15)

        # Handle both raw state_dict and full checkpoint formats
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict):
            # Full training checkpoint — extract just the model weights
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                # Assume the dict itself is the state_dict
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        self.width = 640
        self.height = 360

    def predict(self, image):
        """Predict 14 court keypoints from a single BGR frame.

        Returns a flat numpy array of 28 values: [x0, y0, x1, y1, ..., x13, y13]
        in the original image resolution. Same interface as CLD.predict().
        """
        orig_h, orig_w = image.shape[:2]

        # Preprocess: resize, normalize, to tensor
        resized = cv2.resize(image, (self.width, self.height))
        inp = resized.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0)
        inp = inp.to(self.device)

        with torch.no_grad():
            output = self.model(inp)

        # output shape: (1, 15, 360, 640)
        # argmax across channels to get per-pixel keypoint label
        keypoint_map = output.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        keypoints = np.zeros(28)  # 14 keypoints × 2 coords

        for k in range(14):  # skip channel 14 (center point)
            # Create binary mask for this keypoint
            mask = np.zeros_like(keypoint_map)
            mask[keypoint_map == k] = 255

            circles = cv2.HoughCircles(
                mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                param1=50, param2=2, minRadius=10, maxRadius=25
            )

            if circles is not None and len(circles[0]) > 0:
                x, y, _ = circles[0][0]
                # Scale from 640×360 back to original resolution
                keypoints[k * 2] = x * (orig_w / self.width)
                keypoints[k * 2 + 1] = y * (orig_h / self.height)
            else:
                # Fallback: use center of mass of the mask
                coords = np.where(mask > 0)
                if len(coords[0]) > 0:
                    y_mean = np.mean(coords[0])
                    x_mean = np.mean(coords[1])
                    keypoints[k * 2] = x_mean * (orig_w / self.width)
                    keypoints[k * 2 + 1] = y_mean * (orig_h / self.height)

        return keypoints

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
