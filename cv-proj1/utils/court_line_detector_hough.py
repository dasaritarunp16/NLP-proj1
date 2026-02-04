import cv2
import numpy as np

class CLDH:
    """Court Line Detector using Hough Transform"""

    def __init__(self, c_low=50, c_high=150, h_thresh=100, min_len=100, max_gap=10, w_thresh=200):
        self.low = c_low
        self.high = c_high
        self.h_thresh = h_thresh
        self.min_len = min_len
        self.max_gap = max_gap
        self.w_thresh = w_thresh

    def prepare(self, image):
        """Convert to grayscale, blur, and detect edges."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.low, self.high)
        return edges

    def HLT(self, edges):
        """Hough Line Transform - detect lines."""
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=self.h_thresh,
            minLineLength=self.min_len,
            maxLineGap=self.max_gap
        )
        return lines

    def filter(self, image):
        """Create mask for white court lines."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, self.w_thresh])
        upper = np.array([180, 30, 255])
        mask = cv2.inRange(hsv, lower, upper)
        return mask

    def get_lines(self, image):
        """Get edges filtered for white lines."""
        edges = self.prepare(image)
        white_mask = self.filter(image)
        court_edges = cv2.bitwise_and(edges, edges, mask=white_mask)
        return court_edges

    def label_lines(self, lines):
        """Separate lines into horizontal and vertical groups."""
        h_lines = []
        v_lines = []

        if lines is None:
            return h_lines, v_lines

        for line in lines:
            x1, y1, x2, y2 = line[0]
            theta = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

            if abs(theta) < 30:
                h_lines.append(line[0])
            elif abs(theta) > 60:
                v_lines.append(line[0])

        return h_lines, v_lines

    def intersections(self, line_one, line_two):
        """Find intersection point of two lines."""
        x1, y1, x2, y2 = line_one
        x3, y3, x4, y4 = line_two

        dx1 = x2 - x1
        dx2 = x4 - x3

        # Handle vertical lines
        if dx1 == 0 and dx2 == 0:
            return None

        if dx1 == 0:
            slope_one = float('inf')
        else:
            slope_one = (y2 - y1) / dx1

        if dx2 == 0:
            slope_two = float('inf')
        else:
            slope_two = (y4 - y3) / dx2

        # Check if parallel
        if slope_one != float('inf') and slope_two != float('inf'):
            if abs(slope_one - slope_two) < 0.1:
                return None

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        if abs(denom) < 1e-10:
            return None

        px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom

        return (int(px), int(py))

    def find_intersections(self, h_lines, v_lines, width, height):
        """Find all intersections between horizontal and vertical lines."""
        keypoints = []

        for h_line in h_lines:
            for v_line in v_lines:
                point = self.intersections(h_line, v_line)

                if point is not None:
                    px, py = point
                    if 0 <= px < width and 0 <= py < height:
                        keypoints.append(point)

        return keypoints

    def predict(self, image):
        """Main method - detect court keypoints."""
        height, width = image.shape[:2]
        court_edges = self.get_lines(image)
        lines = self.HLT(court_edges)

        if lines is None:
            return np.array([])

        h_lines, v_lines = self.label_lines(lines)
        keypoints = self.find_intersections(h_lines, v_lines, width, height)

        return np.array(keypoints)

    # Visualization methods
    def draw_lines(self, image, h_lines, v_lines):
        """Draw horizontal (blue) and vertical (green) lines."""
        output = image.copy()

        for line in h_lines:
            x1, y1, x2, y2 = line
            cv2.line(output, (x1, y1), (x2, y2), (255, 0, 0), 2)

        for line in v_lines:
            x1, y1, x2, y2 = line
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return output

    def draw_keypoints(self, image, keypoints):
        """Draw keypoints on image."""
        output = image.copy()

        for i, (x, y) in enumerate(keypoints):
            cv2.circle(output, (x, y), 8, (0, 0, 255), -1)
            cv2.putText(output, str(i), (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return output

    def draw_all(self, image, h_lines, v_lines, keypoints):
        """Draw everything for debugging."""
        output = self.draw_lines(image, h_lines, v_lines)
        output = self.draw_keypoints(output, keypoints)
        return output

    def draw_keypoints_on_video(self, video_frames, keypoints):
        """Draw keypoints on all video frames."""
        output_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_frames.append(frame)
        return output_frames
