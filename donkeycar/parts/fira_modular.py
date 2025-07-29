import time
import numpy as np
import cv2
import apriltag
import threading

def detect_crosswalk(img):
    resized = cv2.resize(img, (160, 120))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 5)
    _, thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stripe_count = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue

        rect = cv2.minAreaRect(c)
        width, height = rect[1]
        if width == 0 or height == 0:
            continue

        aspect_ratio = max(width, height) / min(width, height)
        center_x, center_y = rect[0]

        if aspect_ratio >= 2 and center_y < 80:
            stripe_count += 1

    zebra_detected = stripe_count >= 3
    return _, zebra_detected, stripe_count

class FiraModular:
    def __init__(self, tag_dict, camera_to_front, vehicle_width, vehicle_length,
                 use_route_plan=False, route_plan=None, proximity_thresholds=None,
                 debug=True, require_zebra=False, tag_detection_hz=0.5,
                 ejecutar_accion_al_dejar_de_ver_tag=False,
                 usar_una_sola_camara=False, zebra_detection_hz=2.0):

        self.tag_dict = tag_dict
        self.camera_to_front = camera_to_front
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.use_route_plan = use_route_plan
        self.route_plan = route_plan or []
        self.proximity_thresholds = proximity_thresholds or {}
        self.debug = debug
        self.require_zebra = require_zebra
        self.ejecutar_accion_al_dejar_de_ver_tag = ejecutar_accion_al_dejar_de_ver_tag
        self.usar_una_sola_camara = usar_una_sola_camara

        self.current_step = 0
        self.state = 'conduciendo'
        self.last_state_printed = ''
        self.last_tag_detected = None
        self.current_visible_tag_id = None

        self.tag_detector = apriltag.Detector()

        self.tag_disappear_timeout = 0.3
        self.tag_cooldown_time = 5.0
        self.last_tag_seen_time = 0.0
        self.last_action_time = 0.0
        self.wait_end_time = 0.0

        self.default_tag_ratio_threshold = 10

        self.sequence = []
        self.sequence_index = 0
        self.sequence_start_time = 0.0

        self.last_require_zebra_state = None
        self.latest_img = None
        self.latest_zebra_img = None

        self.tag_detection_lock = threading.Lock()
        self.last_tag_detected_threaded = None
        self.tag_detection_interval = tag_detection_hz
        self.last_debug_print = 0.0
        self.debug_interval = 1.0

        self.zebra_detection_interval = 1.0 / zebra_detection_hz
        self.last_zebra_detection_time = 0.0

        if not self.ejecutar_accion_al_dejar_de_ver_tag:
            threading.Thread(target=self.tag_detection_loop, daemon=True).start()

        self.giro_config = {
            'TURN_RIGHT': 'corregido',
            'TURN_LEFT': 'simple'
        }

        self.sequences = {
            'corregido_right': [
                {'duration': 1.0, 'angle': 0, 'throttle': 0.11},
                {'duration': 1.2, 'angle': 1.0, 'throttle': 0.11},
                {'duration': 1.4, 'angle': -1, 'throttle': -0.11},
                {'duration': 1, 'angle': 0.6, 'throttle': 0.11}
            ],
            'simple_left': [
                {'duration': 2, 'angle': 0, 'throttle': 0.10},
                {'duration': 2.4, 'angle': -1.0, 'throttle': 0.12},
            ],
            'FORWARD': [
                {'duration': 1.0, 'angle': 0.0, 'throttle': 0.15},
            ],
            'STOP': [
                {'duration': 2.0, 'angle': 0.0, 'throttle': 0.0},
            ]
        }

    def run(self, angle_model, throttle_model, img_arr, image_array_1=None):
        current_time = time.time()
        frame_main = img_arr
        frame_secondary = img_arr if self.usar_una_sola_camara else image_array_1
        original_img = frame_main.copy()
        self.latest_img = frame_main.copy()
        self.latest_zebra_img = frame_secondary.copy()

        if self.ejecutar_accion_al_dejar_de_ver_tag:
            tag = self.detect_apriltag(cv2.resize(self.latest_img.copy(), (160, 120)))
            if tag:
                self.last_tag_detected_threaded = tag

        if self.debug and self.state != self.last_state_printed:
            estado = f"[STATE] {self.state}"
            if self.last_tag_detected:
                estado += f" | Tag: {self.last_tag_detected}"
            if self.use_route_plan:
                estado += f" | Paso: {self.current_step}"
            print(estado)
            self.last_state_printed = self.state

        if self.require_zebra != self.last_require_zebra_state:
            print("[MODO] Requiere zebra para ejecutar acción." if self.require_zebra else "[MODO] Ejecuta acción tras perder tag (sin zebra).")
            self.last_require_zebra_state = self.require_zebra

        if self.state == 'conduciendo':
            with self.tag_detection_lock:
                tag = self.last_tag_detected_threaded

            if tag:
                tag_name = self.tag_dict.get(tag.tag_id, 'UNKNOWN')
                self.last_tag_detected = tag_name
                self.last_tag_seen_time = current_time
                self.current_visible_tag_id = tag.tag_id

                tag_width = tag.corners[2][0] - tag.corners[0][0]
                if tag_width > 0:
                    ratio = 320 / tag_width
                    threshold = self.proximity_thresholds.get(tag.tag_id, self.default_tag_ratio_threshold)
                    if current_time - self.last_debug_print > self.debug_interval:
                        print(f"[TAG] Visto {tag_name} | Ratio: {ratio:.2f} (umbral: {threshold})")
                        self.last_debug_print = current_time

                    if ratio < threshold and not self.ejecutar_accion_al_dejar_de_ver_tag:
                        if self.require_zebra:
                            self.state = 'buscando_zebra'
                            print(f"[TAG] Tag {tag_name} aceptado. Buscando zebra...")
                        else:
                            print(f"[TAG] Umbral alcanzado. Deteniendo por 3s...")
                            self.wait_end_time = current_time + 3.0
                            self.state = 'esperando'

            if self.ejecutar_accion_al_dejar_de_ver_tag and self.current_visible_tag_id is not None:
                if current_time - self.last_tag_seen_time > self.tag_disappear_timeout:
                    print(f"[TAG] Tag {self.last_tag_detected} desapareció. Ejecutando acción...")
                    self.wait_end_time = current_time + 3.0
                    self.state = 'esperando'

        elif self.state == 'buscando_zebra':
            if current_time - self.last_zebra_detection_time > self.zebra_detection_interval:
                self.last_zebra_detection_time = current_time
                _, zebra_detected, stripe_count = detect_crosswalk(self.latest_zebra_img)
                if zebra_detected:
                    print(f"[ZEBRA] Zebra detectada con {stripe_count} franjas. Deteniendo por 3s...")
                    self.wait_end_time = current_time + 3.0
                    self.state = 'esperando'
                else:
                    if current_time - self.last_debug_print > self.debug_interval:
                        print(f"[ZEBRA] Buscando zebra... Franjas: {stripe_count}")
                        self.last_debug_print = current_time

        elif self.state == 'esperando':
            if time.time() < self.wait_end_time:
                return 0.0, 0.0, original_img
            print(f"[ACCION] Ejecutando: {self.last_tag_detected}")
            self.execute_action_flow(current_time)

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
                self.last_tag_detected = None
                self.advance_step_or_reset()

        return angle_model, throttle_model, original_img

    def execute_action_flow(self, current_time):
        self.execute_action(self.last_tag_detected)
        if self.sequence:
            self.state = 'ejecutando_secuencia'
            self.sequence_index = 0
            self.sequence_start_time = current_time
        else:
            self.advance_step_or_reset()
        self.last_action_time = time.time()
        self.current_visible_tag_id = None
        with self.tag_detection_lock:
            self.last_tag_detected_threaded = None

    def tag_detection_loop(self):
        while True:
            time.sleep(self.tag_detection_interval)
            if self.state != 'conduciendo' or self.latest_img is None:
                continue
            try:
                img_copy = cv2.resize(self.latest_img.copy(), (160, 120))
                tag = self.detect_apriltag(img_copy)
                with self.tag_detection_lock:
                    self.last_tag_detected_threaded = tag
            except Exception as e:
                if self.debug:
                    print(f"[THREAD ERROR] {e}")

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
        if tag_name in ['TURN_RIGHT', 'TURN_LEFT']:
            tipo = self.giro_config.get(tag_name, 'simple')
            self.sequence = self.sequences.get(f"{tipo}_{tag_name.lower().split('_')[1]}", [])
        else:
            self.sequence = self.sequences.get(tag_name, [])
