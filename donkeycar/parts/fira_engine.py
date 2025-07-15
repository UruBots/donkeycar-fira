import math
import numpy as np
import cv2
import apriltag
import time

class AprilTagDetector(object):
    def __init__(self, tag_dict, proximity_thresholds):
        self.tag_dict = tag_dict
        self.detector = apriltag.Detector()
        self.proximity_thresholds = proximity_thresholds

    def detect_apriltags(self, img_arr):
        gray_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray_img)
        #print("DETECTOOOOOOOOOO",detections)
        return detections

    def is_tag_close(self, tag, img_shape):
        tag_id = tag.tag_id
        if tag_id in self.proximity_thresholds:
            tag_width = tag.corners[2][0] - tag.corners[0][0]
            img_height, img_width = img_shape[:2]
            proximity_threshold = self.proximity_thresholds[tag_id]
            if (img_width / tag_width > proximity_threshold):
                print("IDDDDDDDDDD",proximity_threshold,tag_id)
                return True
        return False

    def draw_bounding_box(self, tag, img_arr):
        img_arr = np.ascontiguousarray(img_arr)
        for corner in tag.corners:
            cv2.circle(img_arr, tuple(corner.astype(int)), 5, (0, 255, 0), -1)
        cv2.polylines(img_arr, [tag.corners.astype(int)], True, (0, 255, 0), 2)

class ZebraCrosswalkDetector(object):
    def __init__(self, detection_hz):
        self.detection_hz = detection_hz
        self.last_detection_time = 0

    def detect_crosswalk(self, img_arr, debug_visuals):
        current_time = time.time()
        if current_time - self.last_detection_time >= 1.0 / self.detection_hz:
            self.last_detection_time = current_time
            gray_img = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
            
            # === NUEVO: recortar la parte central hacia abajo ===
            img_height, img_width = gray_img.shape[:2]
            cropped_img = gray_img[int(img_height * 0.5):int(img_height * 0.9),
                                int(img_width * 0.2):int(img_width * 0.8)]

            edges = cv2.Canny(cropped_img, 175, 225, apertureSize=3)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, minLineLength=21, maxLineGap=5)

            if debug_visuals:
                print("[ZEBRA DEBUG] Zebra edge detection activa (cv2 deshabilitado)")
                #cv2.waitKey(1)

            if lines is not None:
                vertical_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if abs(x1 - x2) < 10:
                        vertical_lines.append(line)
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

class FIRAEngine(object):
    def __init__(self, tag_dict, proximity_thresholds, apriltag_hz, zebra_hz, top_crop_ratio, stop_duration=5, turn_duration=2, wait_duration=3.0, turn_initial_wait_duration=1.0, proceed_correction_duration=1.0, proceed_straight_duration=1.0, debug_visuals=False, debug=True):
        print("Initializing FIRA engine...")
        self.apriltag_detector = AprilTagDetector(tag_dict, proximity_thresholds)
        self.zebra_crosswalk_detector = ZebraCrosswalkDetector(zebra_hz)
        self.turn_manager = TurnManager(turn_duration,turn_initial_wait_duration)
        self.proceed_manager = ProceedManager(proceed_correction_duration, proceed_straight_duration)
        self.debug_visuals = debug_visuals
        self.debug = debug
        self.wait_duration = wait_duration

        self.wait_for_crosswalk_start_time = 0  #variables nuevas   
        self.last_correction_angle = 0.0

        self.last_state_print_time = 0

        self.forward_idle_duration = 0  # Duración del avance recto tras el FORWARD
        self.forward_throttle = 0.14      # Throttle en ese avance
        self.forward_idle_start_time = 0  # Marca de tiempo de inicio

        self.busqueda_tag_count = 0
        self.busqueda_zebra_count = 0
        self.zebra_detected_count = 0

        self.state = 'idle'
        self.stop_start_time = 0
        self.stop_duration = stop_duration
        self.last_apriltag_detection_time = 0
        self.apriltag_hz = apriltag_hz
        self.top_crop_ratio = top_crop_ratio

        self.recovery_idle_time = 0

        # Recorrido modular paso a paso
        self.route_sequence = [
            'FORWARD',
            'TURN_RIGHT_1',
            'TURN_RIGHT_2',
            'TURN_LEFT_1',
            'TURN_RIGHT_3',
            'STOP'
        ]

        self.current_step = 0

        # Parámetros por acción
        self.route_config = {
            'FORWARD': {
                'correction_duration': 1.5,
                'straight_duration': 1,     
                'forward_throttle': 0.16
            },
            'TURN_RIGHT_1': {
                'pre_turn_forward_duration': 0,
                'turn_duration': 2,
                'turn_angle': 1,
                'turn_throttle': 0.14,
                'after_turn_sequence': [
                    {'duration': 1.0, 'angle': -1.0, 'throttle': -0.15},   # corregir a la izquierda
                    {'duration': 1, 'angle': 0.4, 'throttle': -0.14},   # retroceso
                    {'duration': 0.5, 'angle': 1.0, 'throttle': 0.15},     # pausa
                ]
            },
                        'TURN_RIGHT_2': {
                'pre_turn_forward_duration': 0.5,
                'turn_duration': 3,
                'turn_angle': 0.80,
                'turn_throttle': 0.14
            },
            'TURN_LEFT_1': {
                'pre_turn_forward_duration': 0.4,
                'turn_duration': 1.7,
                'turn_angle': -0.90,
                'turn_throttle': 0.13
            },
            'TURN_RIGHT_3': {
                'pre_turn_forward_duration': 0.7,
                'turn_duration': 1.5,
                'turn_angle': 0.9,
                'turn_throttle': 0.15
            }
        }


        # Para avance previo al giro
        self.pre_turn_start_time = None


        # Variable to track the detected AprilTag type
        self.detected_apriltag = None

        if self.debug:
            print("FIRA engine running...")

    # scalate image to 640x480
    def scale_image(self, img, width=640, height=480):
        img_height, img_width, _ = img.shape
        if img_height != height or img_width != width:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            #print("reescaladooooo",img)
        return img

    def crop_image(self, img, crop_ratio=None):
        if crop_ratio is None:
            crop_ratio = self.top_crop_ratio
        img_height, img_width, _ = img.shape
        cropped_img = img[int(img_height * crop_ratio):, :]
        return cropped_img
    

    def reset_fira_state(self):
        if self.debug:
            print("[RESET] Reiniciando estados del FIRA Engine...")
        self.current_step = 0
        self.state = 'idle'
        self.detected_apriltag = None
        self.recovery_idle_time = time.time()
        self.forward_idle_start_time = 0
        self.pre_turn_start_time = None
        self.after_turn_sequence = []
        self.after_turn_step = 0
        self.busqueda_tag_count = 0
        self.busqueda_zebra_count = 0
        self.zebra_detected_count = 0


    def detect_two_reference_lines(self, img):
        img = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 150, 250, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=30)

        correction_angle = 0  # default
        img_height, img_width = img.shape[:2]
        center_x = img_width // 2

        left_line = None
        right_line = None
        min_left_dist = float('inf')
        min_right_dist = float('inf')

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 20:  # línea casi vertical
                    line_x = (x1 + x2) // 2
                    if line_x < center_x and abs(line_x - center_x) < min_left_dist:
                        left_line = (x1, y1, x2, y2)
                        min_left_dist = abs(line_x - center_x)
                    elif line_x > center_x and abs(line_x - center_x) < min_right_dist:
                        right_line = (x1, y1, x2, y2)
                        min_right_dist = abs(line_x - center_x)

        if left_line and right_line:
            left_x = (left_line[0] + left_line[2]) // 2
            right_x = (right_line[0] + right_line[2]) // 2
            middle_x = (left_x + right_x) // 2
            correction = middle_x - center_x
            correction_angle = correction / img_width  # normalizado

            if self.debug_visuals:
                cv2.line(img, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (255, 0, 0), 2)
                cv2.line(img, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 0, 255), 2)
                cv2.line(img, (center_x, 0), (center_x, img_height), (0, 255, 255), 1)
                cv2.putText(img, f"Corr: {correction_angle:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return correction_angle, img



    def detect_dashed_center_line(self, img):
        img = img.copy()
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

    def detect_lane_and_correction(self, img):
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

    def detect_apriltags_and_update_state(self, img_arr, current_time, throttle, angle):
        apriltag_detections = self.apriltag_detector.detect_apriltags(img_arr)
        #print("AAAAAAAAAAAAA",apriltag_detections)
        for tag in apriltag_detections:

            #print("MENSAJE",tag,img_arr.shape[:2].height)

            if self.apriltag_detector.is_tag_close(tag, img_arr.shape):


                if self.debug: 
                    print(f"[TAG] Evaluando cercanía del tag ID {tag.tag_id}")

                tag_name = self.apriltag_detector.tag_dict.get(tag.tag_id, 'UNKNOWN')
                print(f"[TAG] Tag detectado cerca: ID {tag.tag_id} → {tag_name}")

                if self.debug:
                    print("Detected tag: " + str(tag.tag_id))
                self.detected_apriltag = tag_name
                if tag_name in ['STOP', 'DEAD_END']:
                    self.state = 'stop'
                    self.stop_start_time = current_time
                    if self.debug_visuals:
                        self.apriltag_detector.draw_bounding_box(tag, img_arr)
                    if self.debug:
                        print(f"AprilTag detected: {tag_name} - Stop")
                    break

                elif tag_name in ['TURN_LEFT', 'TURN_RIGHT', 'FORWARD']:
                    if self.current_step < len(self.route_sequence):
                        current_action = self.route_sequence[self.current_step]
                    
                        if self.debug:
                            print(f"[TAG] Detected ID: {tag.tag_id} → Action: {current_action}")
                            print(f"[TAG] Confirmado: acción {tag_name} en paso {self.current_step}")
                        self.detected_apriltag = current_action
                        self.state = 'wait-for-crosswalk'
                        self.wait_for_crosswalk_start_time = current_time
                        self.saved_angle = angle
                        self.saved_throttle = throttle
                        if self.debug_visuals:
                            self.apriltag_detector.draw_bounding_box(tag, img_arr)
                        if self.debug:
                            print(f"AprilTag detected: {tag_name} - Step {self.current_step}: {current_action}")
                        break

        return angle, throttle, img_arr

    def run(self, angle, throttle, input_img_arr):
        try:
            
            current_time = time.time()


            if self.debug:
                if current_time - self.last_state_print_time > 1.0:
                    print(f"\n[STATE] Current state: {self.state} | Step: {self.current_step} | Tag: {self.detected_apriltag}")
                    self.last_state_print_time = current_time

            show_img = self.scale_image(input_img_arr)
            if show_img is None:
                return angle, throttle, input_img_arr

            if self.state == 'stop':
                if current_time - self.stop_start_time >= self.stop_duration:
                    self.state = 'idle'
                return 0, 0, input_img_arr

            if self.state == 'wait-for-crosswalk':
                if current_time - self.wait_for_crosswalk_start_time < 0.2:
                    return angle, throttle/2, input_img_arr

                crosswalk_lines = self.zebra_crosswalk_detector.detect_crosswalk(show_img, self.debug_visuals)

                if self.debug and self.busqueda_zebra_count < 5:
                    print(f"[ZEBRA] Buscando cebra... intento {self.busqueda_zebra_count+1}")
                    self.busqueda_zebra_count += 1

                if self.debug_visuals:
                    for line in crosswalk_lines:
                        for x1, y1, x2, y2 in line:
                            cv2.line(show_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if len(crosswalk_lines) >= 4:
                    self.state = 'wait-at-crosswalk'
                    self.stop_start_time = current_time
                    if self.debug_visuals:
                        self.zebra_crosswalk_detector.draw_crosswalk_lines(crosswalk_lines, show_img)
                   
                    if self.debug:
                        self.zebra_detected_count += 1
                        if self.zebra_detected_count <= 2:
                            print(f"[ZEBRA] Detectada - líneas: {len(crosswalk_lines)} | Velocidad previa: {throttle:.2f}")


                    return 0, 0, input_img_arr
                    
                return angle, throttle, input_img_arr


            if self.state == 'wait-at-crosswalk':
                if current_time - self.stop_start_time >= self.wait_duration:
                    config = self.route_config.get(self.detected_apriltag, {})
                    pre_turn = config.get('pre_turn_forward_duration', 0.0)

                    if self.debug:
                        print(f"[WAIT] Acción detectada: {self.detected_apriltag} → pre_turn: {pre_turn}s")

                    if self.detected_apriltag.startswith('TURN') and pre_turn > 0:
                        self.pre_turn_start_time = current_time
                        self.state = 'pre_turn_forward'
                    elif self.detected_apriltag.startswith('TURN'):
                        self.turn_manager.turn_duration = config.get('turn_duration', self.turn_manager.turn_duration)
                        self.turn_manager.start_turn()
                        self.state = 'turn_left' if 'LEFT' in self.detected_apriltag else 'turn_right'

                    elif self.detected_apriltag == 'FORWARD':
                        self.state = 'forward_idle_straight'
                        self.forward_idle_start_time = current_time
                        if self.debug:
                            print("[FORWARD] Realiza avance recto antes de idle")

                    if self.debug:
                        print(f"[WAIT] Deciding next action for {self.detected_apriltag}")
                        if pre_turn > 0:
                            print(f"[WAIT] Pre-turn forward initiated ({pre_turn}s)")
                        else:
                            print(f"[WAIT] Direct to turn or proceed")

                    return 0, 0, input_img_arr
                return 0, 0, input_img_arr  # ← importante este return final del bloque


            # BLOQUE SEPARADO ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓

            if self.state == 'forward_idle_straight':
                if current_time - self.forward_idle_start_time < self.forward_idle_duration:
                    if self.debug:
                        print("[FORWARD] Avanzando recto antes de pasar a idle")
                    return 0.0, self.forward_throttle, input_img_arr
                else:
                    if self.debug:
                        print("[FORWARD] Avance recto completo, paso a idle")
                    self.state = 'idle'
                    self.detected_apriltag = None
                    if self.debug:
                        print(f"[STEP] Avanzando al paso {self.current_step + 1}")
                    self.current_step += 1
                    self.recovery_idle_time = time.time()
                    return 0, 0, input_img_arr



            if self.state == 'pre_turn_forward':
                config = self.route_config.get(self.detected_apriltag, {})
                duration = config.get('pre_turn_forward_duration', 0.5)
                throttle = config.get('pre_turn_throttle', 0.15)

                if self.debug:
                    print(f"[PRE-TURN] Advancing straight for {duration}s with throttle {throttle}")


                if current_time - self.pre_turn_start_time < duration:
                    return 0.0, throttle, input_img_arr
                else:
                    self.turn_manager.turn_duration = config.get('turn_duration', self.turn_manager.turn_duration)
                    self.turn_manager.start_turn()
                    self.state = 'turn_left' if 'LEFT' in self.detected_apriltag else 'turn_right'
                    return 0.0, 0.0, input_img_arr


            if self.state in ['turn_left', 'turn_right']:
                config = self.route_config.get(self.detected_apriltag, {})
                angle = config.get('turn_angle', -1 if self.state == 'turn_left' else 1)
                throttle = config.get('turn_throttle', 0.15)

                if self.turn_manager.is_waiting():
                    if self.debug:
                        print(f"[TURN] Esperando {self.turn_manager.initial_wait_time}s antes de girar")

                    return 0, 0.2, input_img_arr

                if self.turn_manager.is_turning():
                    if self.debug:
                        print(f"[TURN] Girando → Ángulo: {angle}, Throttle: {throttle}")

                    return angle, throttle, input_img_arr

                # Si terminó el giro:
                if self.debug:
                    print(f"[STEP DONE] Finished: {self.detected_apriltag} → Moving to step {self.current_step + 1}")

                sequence = config.get('after_turn_sequence', [])
                if sequence:
                    if self.debug:
                        print(f"[TURN] Iniciando secuencia post-giro ({len(sequence)} pasos)")

                    self.after_turn_sequence = sequence
                    self.after_turn_step = 0
                    self.after_turn_start_time = time.time()
                    self.state = 'after_turn_sequence'
                    return 0, 0, input_img_arr
                else:

                    if self.debug:
                        print(f"[STEP] Avanzando al paso {self.current_step + 1}")

                    self.current_step += 1
                    self.state = 'idle'
                    self.detected_apriltag = None
                    return angle, throttle, input_img_arr



            if self.state == 'after_turn_sequence':
                if self.after_turn_step < len(self.after_turn_sequence):
                    step_cfg = self.after_turn_sequence[self.after_turn_step]
                    elapsed = current_time - self.after_turn_start_time
                    if elapsed < step_cfg['duration']:
                        if self.debug:
                            print(f"[AFTER_TURN] Paso {self.after_turn_step + 1}/{len(self.after_turn_sequence)} → Ángulo: {step_cfg['angle']}, Throttle: {step_cfg['throttle']}")

                        return step_cfg['angle'], step_cfg['throttle'], input_img_arr
                    else:
                        self.after_turn_step += 1
                        self.after_turn_start_time = time.time()
                        return 0, 0, input_img_arr
                else:
                    self.state = 'idle'
                    self.detected_apriltag = None

                    if self.debug:
                        print(f"[STEP] Avanzando al paso {self.current_step + 1}")

                    self.current_step += 1
                    return 0, 0, input_img_arr



            if self.state == 'proceeding':
                elapsed = current_time - self.proceed_manager.proceed_start_time
                forward_throttle = 0.15

                if elapsed < self.proceed_manager.correction_duration:
                    # Fase 1: detectar y corregir suavemente usando línea punteada
                    #correction_angle, _ = self.detect_dashed_center_line(show_img)
                    correction_angle, _ = self.detect_two_reference_lines(show_img)
                    # Limitar el ángulo para que no sea agresivo
                    correction_angle = max(min(correction_angle, 0.8), -0.6)

                    print(f"[DEBUG] raw correction_angle = {correction_angle:.3f}")
                    # Suavizado (Exponential Moving Average)
                    alpha = 0.4  # cuanto más chico, más suave
                    self.last_correction_angle = (
                        alpha * correction_angle + (1 - alpha) * self.last_correction_angle
                    )

                    return self.last_correction_angle, forward_throttle, input_img_arr

                elif elapsed < (self.proceed_manager.correction_duration + self.proceed_manager.straight_duration):
                    # Fase 2: avanzar recto sin corregir
                    return 0, forward_throttle, input_img_arr

                else:
                    self.state = 'idle'
                    self.detected_apriltag = None
                    return 0, 0, input_img_arr


            if self.state == 'idle':

                # Determine if it's time to detect AprilTags
                if (current_time - self.recovery_idle_time > 1.5 and
                    current_time - self.last_apriltag_detection_time >= 1.0 / self.apriltag_hz):

                    self.last_apriltag_detection_time = current_time

                    if self.debug:
                        self.busqueda_tag_count += 1
                        if self.busqueda_tag_count <= 5:
                            print("BUSCANDO TAGS")

                    # Detect AprilTags
                    angle, throttle, show_img = self.detect_apriltags_and_update_state(show_img, current_time, throttle, angle)
                    if angle and throttle:
                        return angle, throttle, input_img_arr

            return angle, throttle, input_img_arr
        except Exception as e:
            print(f"Error in FIRAEngine run: {e}")