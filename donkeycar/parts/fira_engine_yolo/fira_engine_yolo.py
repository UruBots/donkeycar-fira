import os
import numpy as np
import cv2
from ultralytics import YOLO
import time
import math

# Parámetros de calibración para estimar la distancia
KNOWN_DISTANCE = 10  # cm (Distancia de referencia)
KNOWN_WIDTH = 5  # cm (Ancho real del objeto de referencia)
FOCAL_LENGTH = 300  # Ajustar según calibración

model_path = os.path.abspath("./model/yolov8-trained.pt")

class YoloDetect(object):
    def __init__(self, model_path, classes_dict):
        self.model = YOLO(model_path)
        self.classes = classes_dict
        
    def show_fps(self, prev_frame_time, img_arr):
        img = np.copy(img_arr)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        cv2.putText(img,f"FPS: {int(fps)}",(10, 30),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2,cv2.LINE_AA,)
    
    def estimate_distance(self, known_width, focal_length, object_width_px):
        """ Calcula la distancia estimada de un objeto en cm. """
        if object_width_px == 0:
            return None
        return (known_width * focal_length) / object_width_px

    def show_distance(self, distance, x, y, img_arr):
        # Mostrar distancia estimada en cm
        cv2.putText(img_arr, f"{distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,)

    def run(self, img_arr):
        results = self.model(img_arr)
        return results

class ZebraCrosswalkDetector(object):
    def __init__(self, detection_hz):
        self.detection_hz = detection_hz
        self.last_detection_time = 0

    def detect_crosswalk(self, img_arr, debug_visuals):
        # img = np.copy(img_arr)
        current_time = time.time()
        if current_time - self.last_detection_time >= 1.0 / self.detection_hz:
            self.last_detection_time = current_time
            gray_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
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
        img_arr = np.ascontiguousarray(img_arr)
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img_arr, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

class FIRAEngineYolo(object):
    def __init__(self, yolo_classes, apriltag_hz, zebra_hz, top_crop_ratio, stop_duration=5, turn_duration=2, wait_duration=3.0, turn_initial_wait_duration=1.0, proceed_correction_duration=1.0, proceed_straight_duration=1.0, debug_visuals=True, debug=False):
        self.yolo_detector = YoloDetect(model_path, yolo_classes)
        self.yolo_classes = yolo_classes
        self.zebra_crosswalk_detector = ZebraCrosswalkDetector(zebra_hz)
        self.turn_manager = TurnManager(turn_duration,turn_initial_wait_duration)
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
        self.last_yolo_detection_time = 0
        self.detected_object = None

        if self.debug:
            print("FIRA engine running...")

    def crop_image(self, img, crop_ratio=None):
        if crop_ratio is None:
            crop_ratio = self.top_crop_ratio
        img_height, img_width, _ = img.shape
        cropped_img = img[int(img_height * crop_ratio):, :]
        return cropped_img

    def detect_lane_and_correction(self, img):
        img = np.copy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 200, 300, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

        if self.debug_visuals:
            img = edges.copy()
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if self.debug_visuals:
                    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            right_lane = max(lines, key=lambda line: line[0][0])
            x1, y1, x2, y2 = right_lane[0]

            correction_angle = (x1 + x2) / 2 - img.shape[1] / 2

            if self.debug_visuals:
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.arrowedLine(img, (img.shape[1] // 2, img.shape[0]), (int((x1 + x2) / 2), int((y1 + y2) / 2)), (255, 0, 0), 2)

            return correction_angle / img.shape[1], img
        return 0, img

    def detect_dashed_center_line(self, img):
        img = np.copy(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 150, 250, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=30)

        if lines is not None:
            img_height, img_width = img.shape[:2]
            center_x = img_width // 2
            best_line = None
            min_distance = float('inf')
            correction_angle = 0  # Default angle is 0 (no correction needed)

            for line in lines:
                x1, y1, x2, y2 = line[0]
                line_center_x = (x1 + x2) // 2
                distance_to_center = abs(line_center_x - center_x)

                # Filtrar líneas casi verticales y cercanas al centro
                if abs(x2 - x1) < 30 and distance_to_center < min_distance:
                    best_line = (x1, y1, x2, y2)
                    min_distance = distance_to_center

                    # Calcular el ángulo de inclinación
                    delta_y = y2 - y1
                    delta_x = x2 - x1
                    correction_angle = math.degrees(math.atan2(delta_y, delta_x))  # Convert to degrees

            if best_line:
                x1, y1, x2, y2 = best_line
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Dibujar en verde
                if self.debug_visuals:
                    # Mostrar el ángulo de corrección en la imagen
                    text_position = (50, 50)  # Coordenadas donde se mostrará el texto
                    cv2.putText(img, f"Angle: {correction_angle:.2f} deg", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return correction_angle, img
    
    def detect_yolo_signals(self, img_arr, current_time, throttle, angle):
        img = np.copy(img_arr)

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

        results = self.yolo_detector.run(img_arr)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                object_width_px = x2 - x1  # Ancho del objeto en píxeles

                if conf > 0.5:
                    class_name = f"{self.yolo_classes[cls]}: {conf:.2f}"
                    self.detected_object = class_name
                    color = (0, 255, 0)  # Color del cuadro (verde)

                    if(self.debug_visuals):
                        cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        #cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                    distance = self.estimate_distance(KNOWN_WIDTH, FOCAL_LENGTH, object_width_px)

                    if self.debug_visuals:
                        self.yolo_detector.show_distance(distance, x1, y1, img)

                    if distance <= 10:
                        if class_name in ['Stop', 'No_entry', 'End']:
                            self.state = 'stop'
                            self.stop_start_time = current_time
                            if self.debug:
                                print(f"YOLO class detected: {class_name} - Stop || No_entry")
                            break
                        elif class_name in ['Left', 'Right', 'Forward']:
                            self.state = 'wait-for-crosswalk'
                            self.saved_angle = angle
                            self.saved_throttle = throttle
                            if self.debug:
                                print(f"YOLO class detected: {class_name} - Left || Right || Forward")
                            break
        # Mostrar la imagen con anotaciones
        # if(self.debug_visuals):
            # annotated_frame = results[0].plot()
            # cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        return angle, throttle, img_arr

    def run(self, angle, throttle, input_img_arr):
        current_time = time.time()        
        # cropped_input_img = self.crop_image(input_img_arr, 0.65)

        if self.state == 'stop':
            if current_time - self.stop_start_time >= self.stop_duration:
                self.state = 'idle'
            return 0, 0, input_img_arr

        if self.state == 'wait-for-crosswalk':
            crosswalk_lines = self.zebra_crosswalk_detector.detect_crosswalk(input_img_arr, self.debug_visuals)
            if self.debug_visuals:
                for line in crosswalk_lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(input_img_arr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if len(crosswalk_lines) >= 5:
                self.state = 'wait-at-crosswalk'
                self.stop_start_time = current_time
                if self.debug_visuals:
                    self.zebra_crosswalk_detector.draw_crosswalk_lines(crosswalk_lines, input_img_arr)
                if self.debug:
                    print("Zebra crosswalk detected")
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

        if self.state in ['turn_left', 'turn_right']:
            if self.turn_manager.is_waiting():
                return 0, 1, input_img_arr
            if self.turn_manager.is_turning():
                turn_angle = -1 if self.state == 'turn_left' else 1
                turn_throttle = 1
                return turn_angle, turn_throttle, input_img_arr
            else:
                self.state = 'idle'
                self.detected_object = None
                return angle, throttle, input_img_arr

        if self.state == 'proceeding':
            if self.proceed_manager.is_correcting():
                # correction_angle, show_img = self.detect_lane_and_correction(input_img_arr)
                # testear
                correction_angle, show_img = self.detect_dashed_center_line(input_img_arr)
                # if self.debug_visuals:
                #     # cv2.imshow('Street', show_img)
                #     # cv2.waitKey(1)
                if self.debug:
                    print("proceeding - correction_angle", correction_angle)
                return correction_angle, 1, input_img_arr
            elif self.proceed_manager.is_going_straight():
                return 0, 1, input_img_arr
            else:
                self.state = 'idle'
                self.detected_object = None
                return 0, 1, input_img_arr

        if self.state == 'idle':
            if current_time - self.last_yolo_detection_time < 1.0 / self.apriltag_hz:
                if self.debug:
                    print("IDLE - EARLY EXIT...")
                    return angle, throttle, input_img_arr  # Early exit
            
            if current_time - self.last_yolo_detection_time >= 1.0 / self.apriltag_hz:  # Detect every 1 second    
                self.last_yolo_detection_time = current_time
                angle, throttle, input_img_arr = self.detect_yolo_signals(input_img_arr, current_time, throttle, angle)
                if(self.debug):
                    print("Searching with Yolo...")
                    print(f"YOLO response : angle: {angle}, throttle: {throttle} ")
                if(self.debug_visuals):
                    self.yolo_detector.show_fps(current_time, input_img_arr)
                if angle and throttle:
                    return angle, throttle, input_img_arr
            
        return angle, throttle, input_img_arr
