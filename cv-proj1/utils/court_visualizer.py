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

        # Sample every 5th point to reduce clutter
        sampled = ball_trajectory[::5]

        # Draw trajectory lines connecting sampled positions
        for i in range(1, len(sampled)):
            pt1 = self._to_px(sampled[i-1]['rx'], sampled[i-1]['ry'])
            pt2 = self._to_px(sampled[i]['rx'], sampled[i]['ry'])
            t = i / len(sampled)
            color = (int(255 * (1 - t)), 50, int(255 * t))
            cv2.line(img, pt1, pt2, color, 2)

        # Draw sampled positions as dots with frame numbers
        for i, pos in enumerate(sampled):
            px, py = self._to_px(pos['rx'], pos['ry'])
            t = i / len(sampled)
            color = (int(255 * (1 - t)), 50, int(255 * t))
            cv2.circle(img, (px, py), 5, color, -1)
            cv2.putText(img, str(pos['frame']), (px+8, py-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Mark start and end
        start = self._to_px(sampled[0]['rx'], sampled[0]['ry'])
        end = self._to_px(sampled[-1]['rx'], sampled[-1]['ry'])
        cv2.circle(img, start, 10, (255, 0, 0), -1)
        cv2.putText(img, "START", (start[0]+12, start[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(img, end, 10, (0, 0, 255), -1)
        cv2.putText(img, "END", (end[0]+12, end[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imwrite(output_path, img)
        print(f"Court trajectory saved to {output_path}")
        return img

    def plot_shots(self, ball_trajectory, shot_landings, output_dir="output_video"):
        # Generate one court image per shot
        # Split trajectory at each direction reversal (shot landing)
        import os
        os.makedirs(output_dir, exist_ok=True)

        if len(shot_landings) == 0:
            return

        # Build frame ranges for each shot
        # Shot i goes from shot_landings[i-1] to shot_landings[i]
        landing_frames = [s['frame'] for s in shot_landings]

        shot_segments = []
        for i in range(len(landing_frames)):
            if i == 0:
                start_frame = ball_trajectory[0]['frame'] if ball_trajectory else 0
            else:
                start_frame = landing_frames[i - 1]
            end_frame = landing_frames[i]
            segment = [p for p in ball_trajectory if start_frame <= p['frame'] <= end_frame]
            if len(segment) > 0:
                shot_segments.append(segment)

        shot_color = (0, 200, 255)  # yellow

        for shot_num, segment in enumerate(shot_segments):
            img = self.draw_court()

            # Draw trajectory for this shot
            for i in range(1, len(segment)):
                pt1 = self._to_px(segment[i-1]['rx'], segment[i-1]['ry'])
                pt2 = self._to_px(segment[i]['rx'], segment[i]['ry'])
                cv2.line(img, pt1, pt2, shot_color, 2)

            # Draw dots with frame numbers
            for pos in segment:
                px, py = self._to_px(pos['rx'], pos['ry'])
                cv2.circle(img, (px, py), 4, shot_color, -1)
                cv2.putText(img, str(pos['frame']), (px+6, py-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

            # Mark start of shot
            start_pt = self._to_px(segment[0]['rx'], segment[0]['ry'])
            cv2.circle(img, start_pt, 8, (255, 0, 0), -1)

            # Mark landing (end of shot)
            end_pt = self._to_px(segment[-1]['rx'], segment[-1]['ry'])
            cv2.circle(img, end_pt, 8, (0, 0, 255), -1)

            # Arrow from start to end
            cv2.arrowedLine(img, start_pt, end_pt, (255, 255, 255), 1, tipLength=0.05)

            # Label
            cv2.putText(img, f"Shot {shot_num + 1}", (self.margin, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(img, f"Frames {segment[0]['frame']}-{segment[-1]['frame']}", (self.margin, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            path = f"{output_dir}/shot_{shot_num + 1}.png"
            cv2.imwrite(path, img)
            print(f"  Shot {shot_num + 1} saved to {path}")
