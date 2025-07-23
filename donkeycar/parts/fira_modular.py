# fira_modular.py (versión limpia y corregida con threading y resize 320x240)

import time
import numpy as np
import cv2
import apriltag
import threading

def detectar_zebra(frame, min_stripes=4):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (15, 15), 6)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripe_count = 0
    horizontal_found = False
    horizontal_y = None

    for c in contours:
        area = cv2.contourArea(c)
        if area < 2000:
            continue

        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        width, height = rect[1]

        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)
        angle = rect[2]
        center_y = rect[0][1]

        # Detectar la línea horizontal en la parte baja
        if aspect_ratio > 5.0 and abs(angle) < 10 and center_y > frame.shape[0] * 0.6:
            horizontal_found = True
            horizontal_y = center_y
            continue

        # Detectar líneas verticales arriba de la línea horizontal
        if 1.5 < aspect_ratio < 8.0 and center_y < frame.shape[0] * 0.9:
            if horizontal_y is None or center_y < horizontal_y - 20:
                stripe_count += 1

    return stripe_count >= min_stripes



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
        self.require_zebra = require_zebra

        self.current_step = 0
        self.state = 'conduciendo'
        self.last_state_printed = ''
        self.last_tag_detected = None
        self.current_visible_tag_id = None

        self.tag_detector = apriltag.Detector()
        #self.cam2 = cv2.VideoCapture(2)

        #self.latest_img = None
        #self.tag_detection_lock = threading.Lock()
        #self.last_tag_detected_threaded = None
        #self.tag_detection_interval = 0.2

        self.zebra_detection_interval = 0.2
        self.tag_disappear_timeout = 0.3
        self.tag_cooldown_time = 5.0
        self.last_tag_seen_time = 0.0
        self.last_action_time = 0.0
        self.wait_end_time = 0.0
        self.last_zebra_detect_time = 0.0

        self.default_tag_ratio_threshold = 6.0

        self.sequence = []
        self.sequence_index = 0
        self.sequence_start_time = 0.0

        self.last_require_zebra_state = None

        self.giro_config = {
            'TURN_RIGHT': 'corregido',
            'TURN_LEFT': 'simple'
        }

        self.sequences = {
            'corregido_right': [
                {'duration': 1.2, 'angle': 0, 'throttle': 0.10},
                {'duration': 1.4, 'angle': -1.0, 'throttle': -0.10},
                {'duration': 0.5, 'angle': 0, 'throttle': 0.10},
                {'duration': 1.0, 'angle': 1.0, 'throttle': 0.10},
            ],
            'simple_left': [
                {'duration': 1.2, 'angle': 0, 'throttle': 0.10},
                {'duration': 2.5, 'angle': -1.0, 'throttle': 0.10},
            ],
            'FORWARD': [
                {'duration': 1.0, 'angle': 0.0, 'throttle': 0.15},
            ],
            'STOP': [
                {'duration': 2.0, 'angle': 0.0, 'throttle': 0.0},
            ]
        }

        #threading.Thread(target=self.tag_detection_loop, daemon=True).start()

    def run(self, angle_model, throttle_model, img_arr, image_array_1): 
        current_time = time.time()
        original_img = img_arr.copy()
        rescaled_img = cv2.resize(img_arr, (320, 240), interpolation=cv2.INTER_LINEAR)
        self.latest_img = img_arr.copy()

        if self.debug and self.state != self.last_state_printed:
            print(f"[STATE] Estado actual: {self.state} | Paso: {self.current_step} | Tag: {self.last_tag_detected}")
            self.last_state_printed = self.state

        if self.require_zebra != self.last_require_zebra_state:
            print("[MODO] Requiere zebra para ejecutar acción." if self.require_zebra else "[MODO] Ejecuta acción tras perder tag (sin zebra).")
            self.last_require_zebra_state = self.require_zebra

        if self.state == 'conduciendo':
            #with self.tag_detection_lock:
                #tag = self.last_tag_detected_threaded
            tag = self.detect_apriltag(rescaled_img)

            if (
                tag and
                time.time() - self.last_action_time > self.tag_cooldown_time
            ):



                tag_name = self.tag_dict.get(tag.tag_id, 'UNKNOWN')
                self.last_tag_detected = tag_name
                self.last_tag_seen_time = current_time
                self.current_visible_tag_id = tag.tag_id

                tag_width = tag.corners[2][0] - tag.corners[0][0]
                if tag_width > 0:
                    ratio = rescaled_img.shape[1] / tag_width
                    threshold = self.proximity_thresholds.get(tag.tag_id, self.default_tag_ratio_threshold)
                    dist_est = ratio / threshold * self.camera_to_front
                    print(f"[PROXIMIDAD] Distancia estimada al tag: {dist_est:.2f} m (ratio: {ratio:.2f}, umbral: {threshold})")

                self.wait_end_time = current_time + 3.0
                self.state = 'esperando_sin_tag'

        elif self.state == 'esperando_sin_tag':
            if current_time - self.last_tag_seen_time > self.tag_disappear_timeout:
                print(f"[TAG] Ya no se ve el tag, esperando 3s antes de ejecutar")
                self.current_visible_tag_id = None  # Esto evita que se quede enganchado
                self.wait_end_time = current_time + 3.0
                self.state = 'esperando'


        elif self.state == 'esperando':
            if time.time() >= self.wait_end_time:
                ret2, zebra_frame = self.cam2.read()
                if ret2 and detectar_zebra(zebra_frame):
                    print(f"[ZEBRA] Zebra detectada, ejecutando acción {self.last_tag_detected}")
                    self.state = 'ejecutando'
                else:
                    print("[ZEBRA] Aún no se detecta zebra, esperando...")

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
                if current_time - self.sequence_start_time < step['duration']:
                    return step['angle'], step['throttle'], original_img
                else:
                    self.sequence_index += 1
                    self.sequence_start_time = current_time
                    return 0.0, 0.0, original_img
            else:
                self.sequence = []
                self.current_visible_tag_id = None  # ← Liberamos el tag para no repetir
                self.last_tag_detected = None       # ← Limpiamos el nombre del tag mostrado en consola
                self.advance_step_or_reset()

            pass
        #with self.tag_detection_lock:
            #if self.state != 'conduciendo':
                #self.last_tag_detected_threaded = None

        return angle_model, throttle_model, original_img

    #def tag_detection_loop(self):
        #while True:
            #time.sleep(self.tag_detection_interval)
            #if self.state != 'conduciendo' or self.latest_img is None:
            #    continue
            #try:
            #    img_copy = cv2.resize(self.latest_img.copy(), (320, 240))
            #    tag = self.detect_apriltag(img_copy)
            #    with self.tag_detection_lock:
            #        if tag:
            #            if self.debug:
            #                tag_name = self.tag_dict.get(tag.tag_id, 'UNKNOWN')
            #                print(f"[DETECCIÓN] Visto tag ID {tag.tag_id} ({tag_name})")
            #           self.last_tag_detected_threaded = tag
             #       else:
            #            if self.debug:
            #                print("[DETECCIÓN] No se detecta tag")
            #            self.last_tag_detected_threaded = None


            #except Exception as e:
             #   if self.debug:
              #      print(f"[THREAD ERROR] {e}")

    def detect_apriltag(self, img_arr):
        gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        detections = self.tag_detector.detect(gray)
        for tag in detections:
            if tag.tag_id in self.tag_dict and self.is_tag_close(tag, img_arr.shape):
                return tag
        return None

    def is_tag_close(self, tag, img_shape):
        tag_width = tag.corners[2][0] - tag.corners[0][0]
        if tag_width == 0:
            return False
        ratio = img_shape[1] / tag_width
        threshold = self.proximity_thresholds.get(tag.tag_id, self.default_tag_ratio_threshold)
        return ratio < threshold

    def advance_step_or_reset(self):
        if self.use_route_plan:
            self.current_step += 1
            if self.current_step >= len(self.route_plan):
                self.state = 'finalizado'
                print("[RUTA] Secuencia finalizada.")
            else:
                self.state = 'conduciendo'
        else:
            self.state = 'conduciendo'

    def execute_action(self, tag_name):
        print(f"[ACCION] Ejecutando: {tag_name}")
        if tag_name in ['TURN_RIGHT', 'TURN_LEFT']:
            tipo = self.giro_config.get(tag_name, 'simple')
            self.sequence = self.sequences.get(f"{tipo}_{tag_name.lower().split('_')[1]}", [])
        else:
            self.sequence = self.sequences.get(tag_name, [])