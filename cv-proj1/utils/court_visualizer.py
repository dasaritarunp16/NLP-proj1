import cv2
import numpy as np


class CourtVisualizer:
    # Draws a top-down 2D tennis court and plots ball positions on it
    #
    # Real-world court dimensions (meters):
    #   Width (doubles): 10.97m
    #   Length: 23.77m
    #   Singles width: 8.23m
    #   Doubles alley: 1.37m each side
    #   Service line from baseline: 5.485m
    #   Net at: 11.885m

    def __init__(self, scale=40, margin=40):
        self.scale = scale  # pixels per meter
        self.margin = margin
        self.court_width = 10.97
        self.court_length = 23.77
        self.img_w = int(self.court_width * scale) + 2 * margin
        self.img_h = int(self.court_length * scale) + 2 * margin

    def _to_px(self, x, y):
        # Convert real-world coords (meters) to pixel coords on the image
        px = int(x * self.scale + self.margin)
        py = int(y * self.scale + self.margin)
        return px, py

    def draw_court(self):
        img = np.ones((self.img_h, self.img_w, 3), dtype=np.uint8) * 34  # dark background

        # Court surface (green)
        tl = self._to_px(0, 0)
        br = self._to_px(self.court_width, self.court_length)
        cv2.rectangle(img, tl, br, (0, 120, 0), -1)

        white = (255, 255, 255)
        thickness = 2

        # Doubles sidelines
        cv2.line(img, self._to_px(0, 0), self._to_px(0, self.court_length), white, thickness)
        cv2.line(img, self._to_px(10.97, 0), self._to_px(10.97, self.court_length), white, thickness)

        # Singles sidelines
        cv2.line(img, self._to_px(1.37, 0), self._to_px(1.37, self.court_length), white, thickness)
        cv2.line(img, self._to_px(9.60, 0), self._to_px(9.60, self.court_length), white, thickness)

        # Baselines
        cv2.line(img, self._to_px(0, 0), self._to_px(10.97, 0), white, thickness)
        cv2.line(img, self._to_px(0, 23.77), self._to_px(10.97, 23.77), white, thickness)

        # Service lines
        cv2.line(img, self._to_px(1.37, 5.485), self._to_px(9.60, 5.485), white, thickness)
        cv2.line(img, self._to_px(1.37, 18.285), self._to_px(9.60, 18.285), white, thickness)

        # Center service line
        cv2.line(img, self._to_px(5.485, 5.485), self._to_px(5.485, 18.285), white, thickness)

        # Net
        cv2.line(img, self._to_px(0, 11.885), self._to_px(10.97, 11.885), (200, 200, 200), 3)

        # Center marks on baselines
        cv2.line(img, self._to_px(5.485, 0), self._to_px(5.485, 0.3), white, thickness)
        cv2.line(img, self._to_px(5.485, 23.47), self._to_px(5.485, 23.77), white, thickness)

        return img

    def plot_trajectory(self, ball_trajectory, output_path="output_video/court_trajectory.png"):
        img = self.draw_court()

        if len(ball_trajectory) == 0:
            cv2.imwrite(output_path, img)
            return img

        # Draw trajectory line connecting consecutive positions
        for i in range(1, len(ball_trajectory)):
            pt1 = self._to_px(ball_trajectory[i-1]['rx'], ball_trajectory[i-1]['ry'])
            pt2 = self._to_px(ball_trajectory[i]['rx'], ball_trajectory[i]['ry'])
            # Color fades from blue (start) to red (end)
            t = i / len(ball_trajectory)
            color = (int(255 * (1 - t)), 50, int(255 * t))
            cv2.line(img, pt1, pt2, color, 1)

        # Draw ball positions as dots
        for i, pos in enumerate(ball_trajectory):
            px, py = self._to_px(pos['rx'], pos['ry'])
            t = i / len(ball_trajectory)
            color = (int(255 * (1 - t)), 50, int(255 * t))
            cv2.circle(img, (px, py), 3, color, -1)

        # Mark start and end
        start = self._to_px(ball_trajectory[0]['rx'], ball_trajectory[0]['ry'])
        end = self._to_px(ball_trajectory[-1]['rx'], ball_trajectory[-1]['ry'])
        cv2.circle(img, start, 8, (255, 0, 0), -1)  # blue = start
        cv2.putText(img, "START", (start[0]+10, start[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.circle(img, end, 8, (0, 0, 255), -1)  # red = end
        cv2.putText(img, "END", (end[0]+10, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        cv2.imwrite(output_path, img)
        print(f"Court trajectory saved to {output_path}")
        return img
