import numpy as np
import cv2
import apriltag
from PIL import Image
import time

class AprilTagDetector(object):
    def __init__(self, tag_dict, detection_hz, proximity_thresholds):
        self.tag_dict = tag_dict
        self.detection_hz = detection_hz
        self.detector = apriltag.Detector()
        self.last_detection_time = 0
        self.proximity_thresholds = proximity_thresholds

    def detect_apriltags(self, img_arr):
        gray_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray_img)
        return detections

    def is_tag_close(self, tag):
        tag_id = tag.tag_id
        if tag_id in self.proximity_thresholds:
            tag_width = tag.corners[2][0] - tag.corners[0][0]
            tag_height = tag.corners[2][1] - tag.corners[0][1]
            img_height, img_width = tag.corners.shape[:2]
            proximity_threshold = self.proximity_thresholds[tag_id]
            if (tag_width / img_width > proximity_threshold) and (tag_height / img_height > proximity_threshold):
                return True
        return False

class ZebraCrosswalkDetector(object):
    def __init__(self, detection_hz, top_crop_ratio):
        self.detection_hz = detection_hz
        self.last_detection_time = 0
        self.top_crop_ratio = top_crop_ratio

    def detect_crosswalk(self, img_arr):
        current_time = time.time()
        img_height, img_width, _ = img_arr.shape
        cropped_img = img_arr[int(img_height * self.top_crop_ratio):, :]
        if current_time - self.last_detection_time >= 1.0 / self.detection_hz:
            self.last_detection_time = current_time
            gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_img, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
            if lines is not None:
                return lines, cropped_img
        return [], cropped_img

class TurnManager(object):
    def __init__(self, turn_duration, max_throttle, max_angle):
        self.turn_duration = turn_duration
        self.max_throttle = max_throttle
        self.max_angle = max_angle
        self.turn_start_time = 0

    def start_turn(self):
        self.turn_start_time = time.time()

    def is_turning(self):
        return time.time() - self.turn_start_time < self.turn_duration

    def get_turn_values(self):
        elapsed_time = time.time() - self.turn_start_time
        throttle = min(self.max_throttle * (elapsed_time / self.turn_duration), self.max_throttle)
        angle = min(self.max_angle * (elapsed_time / self.turn_duration), self.max_angle)
        return throttle, angle

class FIRAEngine(object):
    def __init__(self, tag_dict, proximity_thresholds, apriltag_hz, zebra_hz, top_crop_ratio, stop_duration=5, turn_duration=2, proceed_duration=3, max_throttle=1.0, max_angle=30.0, debug_visuals=True, debug=False):
        self.tag_dict = tag_dict
        self.proximity_thresholds = proximity_thresholds
        self.apriltag_detector = AprilTagDetector(tag_dict, apriltag_hz, proximity_thresholds)
        self.zebra_crosswalk_detector = ZebraCrosswalkDetector(zebra_hz, top_crop_ratio)
        self.turn_manager = TurnManager(turn_duration, max_throttle, max_angle)
        self.debug_visuals = debug_visuals
        self.debug = debug

        self.state = 'idle'
        self.stop_start_time = 0
        self.stop_duration = stop_duration
        self.proceed_start_time = 0
        self.proceed_duration = proceed_duration
        self.last_apriltag_detection_time = 0
        self.apriltag_hz = apriltag_hz

        if self.debug:
            print("FIRA engine running...")

    def draw_bounding_box(self, tag, img_arr):
        img_arr = np.ascontiguousarray(img_arr)
        for corner in tag.corners:
            cv2.circle(img_arr, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
        cv2.polylines(img_arr, [tag.corners.astype(int)], True, (0, 255, 0), 2)

    def draw_crosswalk_lines(self, lines, img_arr):
        img_arr = np.ascontiguousarray(img_arr)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_arr, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def detect_apriltags_and_update_state(self, img_arr, current_time, throttle, angle):
        apriltag_detections = self.apriltag_detector.detect_apriltags(img_arr)
        for tag in apriltag_detections:
            if self.apriltag_detector.is_tag_close(tag): #TODO: Measure distances first and select closest Tag for processing
                tag_name = self.apriltag_detector.tag_dict.get(tag.tag_id, 'UNKNOWN')
                if tag_name in ['STOP', 'DEAD_END']:
                    self.state = 'stop'
                    self.stop_start_time = current_time
                    if self.debug_visuals:
                        self.draw_bounding_box(tag, img_arr)
                    if self.debug:
                        print(f"AprilTag detected: {tag_name} - Stop")
                    break
                elif tag_name == 'TURN_LEFT':
                    self.state = 'turn_left'
                    self.turn_manager.start_turn()
                    if self.debug_visuals:
                        self.draw_bounding_box(tag, img_arr)
                    if self.debug:
                        print(f"AprilTag detected: {tag_name} - Turn Left")
                    break
                elif tag_name == 'TURN_RIGHT':
                    self.state = 'turn_right'
                    self.turn_manager.start_turn()
                    if self.debug_visuals:
                        self.draw_bounding_box(tag, img_arr)
                    if self.debug:
                        print(f"AprilTag detected: {tag_name} - Turn Right")
                    break
                elif tag_name == 'FORWARD':
                    self.state = 'proceeding'
                    self.proceed_start_time = current_time
                    if self.debug_visuals:
                        self.draw_bounding_box(tag, img_arr)
                    if self.debug:
                        print(f"AprilTag detected: {tag_name} - Proceeding")
                    break
        return throttle, angle, img_arr


    def run(self, angle, throttle, img_arr):
        current_time = time.time()

        if img_arr is None:
            return throttle, angle, img_arr

        if self.state == 'stop':
            if current_time - self.stop_start_time >= self.stop_duration:
                self.state = 'idle'
            return 0, 0, img_arr

        if self.state in ['turn_left', 'turn_right']:
            if self.turn_manager.is_turning():
                turn_throttle, turn_angle = self.turn_manager.get_turn_values()
                if self.state == 'turn_right':
                    turn_angle = -turn_angle
                return turn_throttle, turn_angle, img_arr
            else:
                self.state = 'idle'
                return throttle, angle, img_arr

        if self.state == 'proceeding':
            if current_time - self.proceed_start_time >= self.proceed_duration:
                self.state = 'idle'
            return throttle, angle, img_arr

        if self.state == 'idle':

            # Detect Zebra Crosswalks
            crosswalk_lines, cropped_img = self.zebra_crosswalk_detector.detect_crosswalk(img_arr)
            if crosswalk_lines:
                self.state = 'stop'
                self.stop_start_time = current_time
                if self.debug_visuals:
                    self.draw_crosswalk_lines(crosswalk_lines, cropped_img)
                if self.debug:
                    print("Zebra crosswalk detected")
                return throttle, angle, img_arr

            # Determine if it's time to detect AprilTags
            if current_time - self.last_apriltag_detection_time >= 1.0 / self.apriltag_hz:
                self.last_apriltag_detection_time = current_time
                if self.debug:
                    print("Searching for AprilTag...")
                # Detect AprilTags
                result = self.detect_apriltags_and_update_state(img_arr, current_time, throttle, angle)
                if result:
                    return result

            # Show current detection results if debug_visuals is enabled
            if self.debug_visuals:
                cv2.imshow('AprilTag Detection', img_arr)
                cv2.imshow('Zebra Crosswalk Detection', cropped_img)
                cv2.waitKey(1)

        return throttle, angle, img_arr
