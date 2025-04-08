import os
import time
import math
import numpy as np
import cv2
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

# Parámetros de calibración para estimar la distancia
KNOWN_DISTANCE = 10  # cm (Distancia de referencia)
KNOWN_WIDTH = 5  # cm (Ancho real del objeto de referencia)
FOCAL_LENGTH = 300  # Ajustar según calibración

cv2.setUseOptimized(True)
cv2.setNumThreads(4)

class TensorflowDetect(object):
    def __init__(self, model_folder, model_name):
        # Load TensorFlow model
        model_path = os.path.join(model_folder, model_name)
        self.model = tf.saved_model.load(model_path)
        self.detect_fn = self.model.signatures['serving_default']
        self.classes = {
            1: 'Stop',
            2: 'No_entry',
            3: 'End',
            4: 'Left',
            5: 'Right',
            6: 'Forward',
        }  # Define your classes here

    def show_fps(self, prev_frame_time, img_arr):
        img = img_arr.copy()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        # Mostrar FPS en la imagen
        logger.info(f'FPS: {fps:.2f}')
        cv2.putText(img,f"FPS: {int(fps)}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 255, 0),2,cv2.LINE_AA,)
        return img
    
    def estimate_distance(self, known_width, focal_length, object_width_px):
        """ Calcula la distancia estimada de un objeto en cm. """
        if object_width_px == 0:
            return None
        return (known_width * focal_length) / object_width_px

    def show_distance(self, distance, x, y, img_arr):
        img_arr = img_arr.copy()
        # Mostrar distancia estimada en cm
        cv2.putText(img_arr, f"{distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,)
        return img_arr

    def run_prediction(self, img_arr):
        # Convert the image to the format TensorFlow expects
        input_tensor = tf.convert_to_tensor(img_arr)
        input_tensor = input_tensor[tf.newaxis,...]  # Add batch dimension

        # Run inference
        output_dict = self.detect_fn(input_tensor)

        # All outputs are batches tensors
        # Convert to numpy arrays, and take index [0] to get rid of batch dimension
        boxes = output_dict['detection_boxes'][0].numpy()
        class_ids = output_dict['detection_classes'][0].numpy().astype(np.int32)
        scores = output_dict['detection_scores'][0].numpy()

        return boxes, class_ids, scores

    def run(self, img_arr):
        return self.run_prediction(img_arr)

class ZebraCrosswalkDetector(object):
    def __init__(self, detection_hz):
        self.detection_hz = detection_hz
        self.last_detection_time = 0

    def detect_crosswalk(self, img_arr, debug_visuals):
        img = img_arr.copy()
        current_time = time.time()
        if current_time - self.last_detection_time >= 1.0 / self.detection_hz:
            self.last_detection_time = current_time
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_img, 175, 225, apertureSize=3)
            
            # Use Hough Line Transform to detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=21, maxLineGap=5)
            # if debug_visuals:
                # cv2.imshow('Crosswalk', edges)
                # cv2.waitKey(1)
            if lines is not None:
                # Filter lines to detect zebra crosswalk pattern
                vertical_lines = [line for line in lines if abs(line[0][0] - line[0][2]) < 10]
                return vertical_lines
        return []

    def draw_crosswalk_lines(self, lines, img_arr):
        img_arr = img_arr.copy()
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return img_arr

# Other classes remain the same
class TurnManager(object):
    def __init__(self, turn_duration, initial_wait_time):
        self.turn_duration = turn_duration
        self.turn_start_time = 0
        self.initial_wait_time = initial_wait_time  # Wait time before executing the turn
        self.wait_start_time = 0

    def start_turn(self):
        self.turn_start_time = time.time()
        self.wait_start_time = time.time()

    def is_waiting(self):
        return time.time() - self.wait_start_time < self.initial_wait_time

    def is_turning(self):
        return time.time() - self.turn_start_time < self.turn_duration

class ProceedManager(object):
    def __init__(self, correction_duration, straight_duration):
        self.correction_duration = correction_duration
        self.straight_duration = straight_duration
        self.proceed_start_time = 0

    def start_proceed(self):
        self.proceed_start_time = time.time()

    def is_correcting(self):
        return time.time() - self.proceed_start_time < self.correction_duration

    def is_going_straight(self):
        elapsed_time = time.time() - self.proceed_start_time
        return self.correction_duration <= elapsed_time < (self.correction_duration + self.straight_duration)

class FIRAEngineTensorFlow(object):
    def __init__(self, model_folder, tf_model_name, apriltag_hz, zebra_hz, top_crop_ratio, stop_duration=5, turn_duration=2, wait_duration=3.0, turn_initial_wait_duration=1.0, proceed_correction_duration=1.0, proceed_straight_duration=1.0, debug_visuals=True, debug=False):
        self.tf_detector = TensorflowDetect(model_folder, tf_model_name)
        self.zebra_crosswalk_detector = ZebraCrosswalkDetector(zebra_hz)
        self.turn_manager = TurnManager(turn_duration, turn_initial_wait_duration)
        self.proceed_manager = ProceedManager(proceed_correction_duration, proceed_straight_duration)
        self.debug_visuals = debug_visuals
        self.debug = debug
        self.wait_duration = wait_duration

        # State management
        self.state = "idle"
        self.stop_start_time = 0
        self.stop_duration = stop_duration
        self.apriltag_hz = apriltag_hz
        self.top_crop_ratio = top_crop_ratio
        self.last_tf_detection_time = 0
        self.detected_object = None

        if self.debug:
            logger.info("FIRA engine running...")

    def crop_image(self, img, crop_ratio=None):
        if crop_ratio is None:
            crop_ratio = self.top_crop_ratio
        img_height, img_width, _ = img.shape
        cropped_img = img[int(img_height * crop_ratio):, :]
        return cropped_img
    
    def resize_image(self, image):
        img = image.copy()  # Copy the input image to avoid modifying the original image
        img = cv2.resize(img, (640, 480))  # Assign the resized image back to img
        return img

    def detect_tf_signals(self, img_arr, current_time, throttle, angle):
        img = img_arr.copy()

        # Ensure img is a valid OpenCV image (uint8, 3 channels)
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("❌ Error: img_arr is not a valid numpy array")

        if img.dtype != np.uint8:
            img = img.astype(np.uint8)

        if len(img.shape) == 2:  # Convert grayscale to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.shape[-1] == 4:  # Convert RGBA to BGR
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        # Ensure it's a valid image shape
        if len(img.shape) != 3 or img.shape[-1] != 3:
            raise ValueError(f"❌ Error: img has incorrect shape {img.shape}")

        # Run detection with TensorFlow model
        boxes, class_ids, scores = self.tf_detector.run(img)
        if self.debug:
            logger.info(f"TensorFlow results: {boxes}, {class_ids}, {scores}")

        if len(boxes) == 0:
            return angle, throttle, img_arr

        # Assuming you have a list of class names and confidence threshold
        for i, score in enumerate(scores):
            if score > 0.7:
                class_id = class_ids[i]
                box = boxes[i]
                y1, x1, y2, x2 = box

                class_name = "Unknown"  # Define your classes mapping here
                distance = self.tf_detector.estimate_distance(KNOWN_WIDTH, FOCAL_LENGTH, x2 - x1)

                if class_name in ['Stop', 'No_entry', 'End']:
                    self.state = 'stop'
                    self.stop_start_time = current_time
                    break
                elif class_name in ['Left', 'Right', 'Forward']:
                    self.state = 'wait-for-crosswalk'
                    self.saved_angle = angle
                    self.saved_throttle = throttle
                    break

        return angle, throttle, img_arr

    def run(self, angle, throttle, input_img_arr):
        current_time = time.time()        

        if self.state == 'stop':
            if current_time - self.stop_start_time >= self.stop_duration:
                self.state = 'idle'
            return 0, 0, input_img_arr

        if self.state == 'wait-for-crosswalk':
            crosswalk_lines = self.zebra_crosswalk_detector.detect_crosswalk(input_img_arr, self.debug_visuals)                
            if len(crosswalk_lines) >= 5:
                self.state = 'wait-at-crosswalk'
                self.stop_start_time = current_time
                if self.debug_visuals:
                    input_img_arr = self.zebra_crosswalk_detector.draw_crosswalk_lines(crosswalk_lines, input_img_arr)
                if self.debug:
                    logger.info("Zebra crosswalk detected")

                return 0, 0, input_img_arr
            return angle, throttle, input_img_arr

        if self.state == 'wait-at-crosswalk':
            if current_time - self.stop_start_time >= self.wait_duration:
                if self.detected_object == 'Left':
                    self.turn_manager.start_turn()
                    self.state = 'turn_left'
                elif self.detected_object == 'Right':
                    self.turn_manager.start_turn()                    
                    self.state = 'turn_right'
                elif self.detected_object == 'Forward':
                    self.proceed_manager.start_proceed()
                    self.state = 'proceeding'
                self.detected_object = None
                return 0, 0, input_img_arr
            return 0, 0, input_img_arr

        if self.state == 'idle':
            if current_time - self.last_tf_detection_time < 1.0 / self.apriltag_hz:
                return angle, throttle, input_img_arr
            self.last_tf_detection_time = current_time
            if(self.debug):
                logger.info("Searching with Tensorflow...")
                logger.info(f"YOLO response : angle: {angle}, throttle: {throttle}")
            return self.detect_tf_signals(input_img_arr, current_time, throttle, angle)
        
        return angle, throttle, input_img_arr
