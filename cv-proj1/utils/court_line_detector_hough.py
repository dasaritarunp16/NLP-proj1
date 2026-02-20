import cv2
import numpy as np

class CLDH:
    def __init__(self, c_low, c_high, h_thresh, min, max, w_thresh):
        self.low = c_low
        self.high = c_high
        self.h_thresh = h_thresh
        self.min = min
        self.max = max
        self.w_thresh = w_thresh
        
    
    def prepare(self, image):
        scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_blur = cv2.GaussianBlur(scale, (5,5),0)
        edges = cv2.Canny(noise_blur, self.low, self.high)
        return edges
    
    def HLT(self, edges):
        Hough_lines = cv2.HoughLinesP(
            edges,
            rho = 1,
            theta=np.pi/180,
            threshold = self.h_thresh,
            minLineLength=  self.min,
            maxLineGap = self.max
        )
        return Hough_lines
    
    def filter(self, image):
        color = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        less = np.array([0, 0, 240])    # Changed from 200 to 240
        more = np.array([180, 20, 255]) # Changed from 30 to 20
        mask = cv2.inRange(color, less, more)
        return mask

    def apply_roi(self, image):
   
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
    
        # Define court rectangle (adjust for your video)
        top = int(h * 0.05)      # Start below ads
        bottom = int(h * 0.85)   # End above scoreboard
        left = int(w * 0.15)     # Left margin
        right = int(w * 1.6)    # Right margin
    
        mask[top:bottom, left:right] = 255
        return mask



    def get_lines(self, image):
    # Get white line mask first
        white_mask = self.filter(image)
    
    # Get blue court ROI
        roi_mask = self.apply_roi(image)
    
    # Combine: white lines within blue court only
        combined = cv2.bitwise_and(white_mask, roi_mask)
    
    # Detect edges on the white mask (not grayscale)
        edges = cv2.Canny(combined, self.low, self.high)
    
        return edges


    def label_lines(self, lines):
        h_lines = []
        v_lines = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            theta = np.arctan2(y2-y1, x2 -x1) * 180 / np.pi
            
            if abs(theta) < 30:
                h_lines.append(line[0])
            if abs(theta) > 60:
                v_lines.append(line[0])
                
        return h_lines, v_lines
    
    def intersections(self, line_one, line_two):
        x1, y1, x2, y2 = line_one
        x3,y3,x4,y4 = line_two
         
        slope_one = (y2-y1)/(x2-x1) 
        slope_two = (y4-y3)/(x4-x3)
        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if abs(slope_one - slope_two) < 0.1:
            return None
        
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
            
        return (int(px), int(py))   
            
    def find_intersections(self, h_lines, v_lines, width, height):
        
        keyPs = []
        for h_line in h_lines:
            for v_line in v_lines:
                keypoint = self.intersections(h_line, v_line)
                if keypoint is not None:
                    px = keypoint[0]
                    py = keypoint[1]
                    if not 0 <= px < width or not 0 <= py < height:
                        continue
                    keyPs.append(keypoint)
                    
        return keyPs    
    
    def predict(self, image):
        height, width = image.shape[:2]
        ct_edges = self.get_lines(image)
        ct_hough_edges = self.HLT(ct_edges)
        if ct_hough_edges is None:
            return np.array([])
        h, v = self.label_lines(ct_hough_edges)
        keypoints = self.find_intersections(h, v, width, height)
        keypoints = self.filter_keypoints(keypoints, min_dist=30)  # Add this line
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
    
    def filter_keypoints(self, keypoints, min_dist=30):
   
        if len(keypoints) == 0:
            return keypoints
    
        filtered = [keypoints[0]]
        for kp in keypoints[1:]:
            too_close = False
            for f in filtered:
                dist = np.sqrt((kp[0]-f[0])**2 + (kp[1]-f[1])**2)
                if dist < min_dist:
                    too_close = True
                    break
            if not too_close:
                filtered.append(kp)
        return filtered
    