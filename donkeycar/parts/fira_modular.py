# fira_modular.py

import time
import numpy as np
import cv2
import apriltag

class FiraModular:
    def __init__(self, tag_dict, camera_to_front, vehicle_width, vehicle_length, use_route_plan=False, route_plan=None, proximity_thresholds=None, debug=True, require_zebra=True):
        self.tag_dict = tag_dict
        self.camera_to_front = camera_to_front
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.use_route_plan = use_route_plan
        self.route_plan = route_plan or []
        self.proximity_thresholds = proximity_thresholds or {}
        self.debug = debug

        self.current_step = 0
        self.state = 'conduciendo'
        self.last_state_printed = ''
        self.last_tag_detected = None
        self.tag_detector = apriltag.Detector()
        self.wait_end_time = 0
        self.last_zebra_detect_time = 0
        self.zebra_detection_interval = 0.2

        self.crosswalk_threshold_lines = 4
        self.zebra_alignment_margin = 0.05

        self.require_zebra = require_zebra
        self.tag_disappear_timeout = 0.3
        self.last_tag_seen_time = 0


        self.alignment_end_time = 0

        self.sequence = []
        self.sequence_index = 0
        self.sequence_start_time = 0
        self.last_require_zebra_state = None

        self.giro_config = {
            'TURN_RIGHT': 'corregido',
            'TURN_LEFT': 'simple'
        }

        self.sequences = {
            'simple_right': [
                {'duration': 1.5, 'angle': 1.0, 'throttle': 0.14},
            ],
            'corregido_right': [
                {'duration': 0.8, 'angle': 0, 'throttle': 0.16},
                {'duration': 1.2, 'angle': 1.0, 'throttle': 0.16},
                {'duration': 1, 'angle': -1.0, 'throttle': -0.16},
                {'duration': 0.5, 'angle': -0.5, 'throttle': 0.14},
            ],
            'simple_left': [
                {'duration': 5, 'angle': -1.0, 'throttle': 0.14},
            ],
            'corregido_left': [
                {'duration': 0.7, 'angle': -1.0, 'throttle': 0.15},
                {'duration': 0.5, 'angle': 1.0, 'throttle': -0.14},
                {'duration': 0.8, 'angle': -0.5, 'throttle': -0.12},
                {'duration': 0.5, 'angle': 0.0, 'throttle': 0.15},
            ]
        }

    def run(self, angle_model, throttle_model, img_arr):
        current_time = time.time()

        original_img = img_arr.copy()  # imagen del modelo (160x120)
        rescaled_img = cv2.resize(img_arr, (640, 480), interpolation=cv2.INTER_LINEAR)  # para análisis

        if self.debug and self.state != self.last_state_printed:
            print(f"[STATE] Estado actual: {self.state} | Paso: {self.current_step} | Tag: {self.last_tag_detected}")
            self.last_state_printed = self.state

        if self.require_zebra != self.last_require_zebra_state:
            if self.require_zebra:
                print("[MODO] Requiere zebra para ejecutar acción.")
            else:
                print("[MODO] Ejecuta acción tras perder tag (sin zebra).")
            self.last_require_zebra_state = self.require_zebra

        if self.state == 'conduciendo':
            tag = self.detect_apriltag(rescaled_img)
            if tag:
                self.last_tag_detected = self.tag_dict.get(tag.tag_id, 'UNKNOWN')
                self.last_tag_seen_time = time.time()
                if self.debug:
                    print(f"[TAG] Detectado: ID {tag.tag_id} → {self.last_tag_detected}")
                if self.require_zebra:
                    self.state = 'buscando_zebra'
                else:
                    self.state = 'esperando_sin_tag'

        elif self.state == 'buscando_zebra':
            if current_time - self.last_zebra_detect_time > self.zebra_detection_interval:
                self.last_zebra_detect_time = current_time
                zebra_detected = self.detect_zebra(rescaled_img)
                if zebra_detected:
                    if self.debug:
                        print(f"[ZEBRA] Detectada → esperando 3s antes de ejecutar")
                    self.wait_end_time = time.time() + 3.0
                    self.state = 'esperando'

        elif self.state == 'esperando_sin_tag':
            tag = self.detect_apriltag(rescaled_img)
            if tag:
                self.last_tag_seen_time = time.time()
            elif time.time() - self.last_tag_seen_time > self.tag_disappear_timeout:
                if self.debug:
                    print(f"[TAG] Ya no se ve el tag, esperando 3s antes de ejecutar")
                self.wait_end_time = time.time() + 3.0
                self.state = 'esperando'

        elif self.state == 'esperando':
            if time.time() >= self.wait_end_time:
                if self.debug:
                    print(f"[ESPERA] Finalizada, ejecutando acción {self.last_tag_detected}")
                self.state = 'ejecutando'
            return 0.0, 0.0, original_img

        elif self.state == 'ejecutando':
            self.execute_action(self.last_tag_detected)
            if self.sequence:
                self.state = 'ejecutando_secuencia'
                self.sequence_index = 0
                self.sequence_start_time = current_time
            else:
                self.advance_step_or_reset()

        elif self.state == 'ejecutando_secuencia':
            if self.sequence_index < len(self.sequence):
                step = self.sequence[self.sequence_index]
                elapsed = current_time - self.sequence_start_time
                if elapsed < step['duration']:
                    return step['angle'], step['throttle'], original_img
                else:
                    self.sequence_index += 1
                    self.sequence_start_time = current_time
                    return 0.0, 0.0, original_img
            else:
                self.sequence = []
                self.advance_step_or_reset()

        return angle_model, throttle_model, original_img


    def advance_step_or_reset(self):
        if self.use_route_plan:
            self.current_step += 1
            if self.current_step >= len(self.route_plan):
                self.state = 'finalizado'
                if self.debug:
                    print("[RUTA] Secuencia finalizada.")
            else:
                self.state = 'conduciendo'
        else:
            self.state = 'conduciendo'

    def detect_apriltag(self, img_arr):
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        detections = self.tag_detector.detect(gray)
        for tag in detections:
            if tag.tag_id in self.tag_dict and self.is_tag_close(tag, img_arr.shape):
                if self.debug:
                    print(f"[TAG] Detectado cerca: ID {tag.tag_id}")
                return tag
        return None

    def is_tag_close(self, tag, img_shape):
        if not tag or not hasattr(tag, 'corners'):
            return False

        tag_width = tag.corners[2][0] - tag.corners[0][0]
        img_height, img_width = img_shape[:2]

        tag_id = tag.tag_id
        proximity_threshold = self.proximity_thresholds.get(tag_id, 1.0)

        if tag_width == 0:
            return False

        ratio = img_width / tag_width
        if self.debug:
            print(f"[PROXIMIDAD] Ratio: {ratio:.2f} (umbral: {proximity_threshold})")

        return ratio < proximity_threshold

    def detect_zebra(self, img_arr):
        frame = cv2.resize(img_arr, (160, 120))
        h = frame.shape[0]
        mitad_inferior = frame[h//2:, :]

        gray = cv2.cvtColor(mitad_inferior, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

        _, mask_white = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(mask_white, kernel, iterations=1)

        edges = cv2.Canny(dilated, 30, 100)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                minLineLength=10, maxLineGap=10)

        vertical_count = 0
        horizontal_count = 0

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                length = np.hypot(x2 - x1, y2 - y1)

                if 30 <= angle <= 150:
                    vertical_count += 1
                if length >= 10 and (angle < 20 or angle > 160):
                    horizontal_count += 1

        if self.debug:
            print(f"[Zebra] Verticales: {vertical_count}, Horizontales: {horizontal_count}")

        zebra_detected = vertical_count >= 6 and horizontal_count >= 1
        return zebra_detected


    def execute_action(self, tag_name):
        if self.debug:
            print(f"[ACCION] Ejecutando: {tag_name}")
        if tag_name == 'TURN_RIGHT':
            tipo = self.giro_config.get(tag_name, 'simple')
            self.sequence = self.sequences.get(f'{tipo}_right', [])
        elif tag_name == 'TURN_LEFT':
            tipo = self.giro_config.get(tag_name, 'simple')
            self.sequence = self.sequences.get(f'{tipo}_left', [])
        elif tag_name == 'FORWARD':
            self.sequence = [
                {'duration': 1.0, 'angle': 0.0, 'throttle': 0.15}
            ]
        elif tag_name == 'STOP':
            self.sequence = [
                {'duration': 2.0, 'angle': 0.0, 'throttle': 0.0}
            ]
