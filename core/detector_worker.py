from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
import numpy as np
from core.kalman_tracker import KalmanBoxTracker 
import math # Import for math.hypot
from gui.image_saver import ImageSaverThread

# Caché de modelos YOLO a nivel de módulo
yolo_model_cache = {}

MODEL_PATHS = {
    "Embarcaciones": r"E:\\embarcaciones\\pyqt6\\ptz_tracker\\best.pt",
    "Personas": r"E:\\embarcaciones\\pyqt6\\ptz_tracker\\yolov8m.pt",
    "Autos": r"E:\\embarcaciones\\pyqt6\\ptz_tracker\\yolov8m.pt",
    "Barcos": r"E:\\embarcaciones\\pyqt6\\ptz_tracker\\yolov8m.pt"
}

MODEL_CLASSES = {
    "Embarcaciones": [0],
    "Personas": [0],
    "Autos": [2],
    "Barcos": [9]
}

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou_val = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou_val

class DetectorWorker(QThread):
    result_ready = pyqtSignal(list, str) # Modificado: ya no incluye 'object' para el frame

    def __init__(self, model_key="Personas", parent=None, frame_interval=50, confidence=0.5, imgsz=640):
        super().__init__(parent)
        self.model_key = model_key
        model_path = MODEL_PATHS.get(model_key, MODEL_PATHS["Personas"])
        model_classes = MODEL_CLASSES.get(model_key, [0])

        print(f"✨ [YOLO] Solicitando modelo '{model_key}' desde: {model_path}")
        if model_path in yolo_model_cache:
            self.model = yolo_model_cache[model_path]
            print(f"🧠 [YOLO Cache] Usando modelo '{model_key}' desde caché.")
        else:
            self.model = YOLO(model_path) 
            yolo_model_cache[model_path] = self.model
            print(f"💿 [YOLO Cache] Modelo '{model_key}' cargado desde disco y añadido a caché.")
        
        self.model_classes = model_classes
        self.confidence = confidence
        self.imgsz = imgsz

        self.frame = None # El worker aún necesita mantener una referencia al frame actual para procesarlo
        self.running = False
        self.frame_interval = frame_interval 
        self.frame_count = 0
        
        self.trackers = [] 
        self.iou_threshold = 0.3 
        self.max_movimiento_detenido = 3 
        self.max_time_invisible = 6 
        self.recently_captured_track_ids = set()

        self.lost_trackers = [] 
        self.max_time_invisible_for_reid = 15 
        self.reid_distance_threshold = 50 

    def set_frame(self, frame):
        if isinstance(frame, np.ndarray):
            self.frame = frame # El worker actualiza su frame interno aquí

    def run(self):
        self.running = True
        KalmanBoxTracker.count = 0 

        while self.running:
            if self.frame is not None: # Procesa usando self.frame
                self.frame_count += 1
                
                actual_processing_interval = self.frame_interval 
                if actual_processing_interval < 1:
                    actual_processing_interval = 1

                if self.frame_count % actual_processing_interval == 0:
                    # ... (toda la lógica de predicción y tracking que usa self.frame)
                    yolo_results = self.model.predict(
                        source=self.frame, classes=self.model_classes,
                        conf=self.confidence, imgsz=self.imgsz, verbose=False
                    )[0]

                    current_detections = []
                    for r in yolo_results.boxes:
                        x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
                        cls = int(r.cls[0])
                        conf = float(r.conf[0])
                        current_detections.append({'bbox': np.array([x1, y1, x2, y2]), 'cls': cls, 'conf': conf})
                    
                    for trk in self.trackers: trk.predict()
                    for lt in self.lost_trackers: lt.predict()

                    current_frame_active_trackers = [] 
                    original_trackers_matched_flags = [False] * len(self.trackers) 
                    temp_matched_detections_indices = set() 

                    if len(self.trackers) > 0 and len(current_detections) > 0:
                        matches = [] 
                        for d_idx, det in enumerate(current_detections):
                            for t_idx, trk in enumerate(self.trackers): 
                                predicted_bbox = trk.get_state() 
                                current_iou = iou(det['bbox'], predicted_bbox)
                                if current_iou > self.iou_threshold:
                                    matches.append((d_idx, t_idx, current_iou))
                        matches.sort(key=lambda x: x[2], reverse=True)
                        
                        for d_idx, t_idx, _iou_score in matches: 
                            if d_idx not in temp_matched_detections_indices and \
                               not original_trackers_matched_flags[t_idx]:
                                det = current_detections[d_idx]
                                trk = self.trackers[t_idx] 
                                trk.update(det['bbox'], det['cls'], det['conf']) 
                                bbox_actual = det['bbox']
                                cx_actual = (bbox_actual[0] + bbox_actual[2]) / 2
                                cy_actual = (bbox_actual[1] + bbox_actual[3]) / 2
                                if trk.last_center_position is not None:
                                    distancia = math.hypot(cx_actual - trk.last_center_position[0], 
                                                           cy_actual - trk.last_center_position[1])
                                    if distancia < self.max_movimiento_detenido:
                                        trk.frames_consecutivos_detenido += 1
                                    else:
                                        trk.frames_consecutivos_detenido = 0 
                                else:
                                    trk.frames_consecutivos_detenido = 0 
                                trk.last_center_position = (cx_actual, cy_actual)
                                current_frame_active_trackers.append(trk) 
                                temp_matched_detections_indices.add(d_idx)
                                original_trackers_matched_flags[t_idx] = True 
                    
                    for t_idx, trk in enumerate(self.trackers):
                        if not original_trackers_matched_flags[t_idx]:
                            trk.frames_consecutivos_detenido = 0 
                            current_frame_active_trackers.append(trk)
                    
                    potential_new_detection_indices = [
                        d_idx for d_idx in range(len(current_detections)) 
                        if d_idx not in temp_matched_detections_indices
                    ]
                    revived_trackers = []
                    reid_matched_detection_indices = set()
                    lost_trackers_to_remove_from_main_lost_list = []

                    eligible_lost_trackers = sorted(
                        [lt for lt in self.lost_trackers if lt.time_since_update < self.max_time_invisible_for_reid],
                        key=lambda lt: lt.time_since_update
                    )

                    for d_idx in potential_new_detection_indices:
                        det = current_detections[d_idx]
                        det_center_x = (det['bbox'][0] + det['bbox'][2]) / 2
                        det_center_y = (det['bbox'][1] + det['bbox'][3]) / 2
                        best_match_lost_tracker = None
                        min_dist = float('inf')

                        for lost_trk in eligible_lost_trackers:
                            if lost_trk in lost_trackers_to_remove_from_main_lost_list: continue
                            if det['cls'] != lost_trk.last_cls: continue
                            lost_trk_pred_bbox = lost_trk.get_state()
                            lost_trk_center_x = (lost_trk_pred_bbox[0] + lost_trk_pred_bbox[2]) / 2
                            lost_trk_center_y = (lost_trk_pred_bbox[1] + lost_trk_pred_bbox[3]) / 2
                            distance = math.hypot(det_center_x - lost_trk_center_x, det_center_y - lost_trk_center_y)
                            if distance < self.reid_distance_threshold and distance < min_dist:
                                min_dist = distance
                                best_match_lost_tracker = lost_trk
                        
                        if best_match_lost_tracker:
                            best_match_lost_tracker.update(det['bbox'], det['cls'], det['conf'])
                            bbox_actual = det['bbox']
                            cx_actual = (bbox_actual[0] + bbox_actual[2]) / 2
                            cy_actual = (bbox_actual[1] + bbox_actual[3]) / 2
                            if best_match_lost_tracker.last_center_position is not None:
                                dist_val = math.hypot(cx_actual - best_match_lost_tracker.last_center_position[0], 
                                                       cy_actual - best_match_lost_tracker.last_center_position[1])
                                if dist_val < self.max_movimiento_detenido:
                                    best_match_lost_tracker.frames_consecutivos_detenido += 1
                                else:
                                    best_match_lost_tracker.frames_consecutivos_detenido = 0
                            else:
                                best_match_lost_tracker.frames_consecutivos_detenido = 0
                            best_match_lost_tracker.last_center_position = (cx_actual, cy_actual)
                            revived_trackers.append(best_match_lost_tracker)
                            lost_trackers_to_remove_from_main_lost_list.append(best_match_lost_tracker)
                            reid_matched_detection_indices.add(d_idx)
                    
                    current_frame_active_trackers.extend(revived_trackers)
                    self.lost_trackers = [lt for lt in self.lost_trackers if lt not in lost_trackers_to_remove_from_main_lost_list]

                    for d_idx in range(len(current_detections)):
                        if d_idx not in temp_matched_detections_indices and d_idx not in reid_matched_detection_indices: 
                            det = current_detections[d_idx]
                            new_trk = KalmanBoxTracker(det['bbox'], det['cls'], det['conf'])
                            current_frame_active_trackers.append(new_trk)
                    
                    self.trackers = current_frame_active_trackers
                    self.trackers = [trk for trk in self.trackers if trk.frames_consecutivos_detenido < 3]
                    surviving_active_trackers = []
                    for trk in self.trackers:
                        if trk.time_since_update < self.max_time_invisible:
                            surviving_active_trackers.append(trk)
                        else:
                            self.lost_trackers.append(trk)
                    self.trackers = surviving_active_trackers
                    self.lost_trackers = [lt for lt in self.lost_trackers if lt.time_since_update < self.max_time_invisible_for_reid]

                    active_track_ids = {trk.id for trk in self.trackers}
                    self.recently_captured_track_ids = self.recently_captured_track_ids.intersection(active_track_ids)

                    # La lógica de captura de imágenes aún necesita self.frame para ImageSaverThread
                    for trk in self.trackers:
                        if trk.id in self.recently_captured_track_ids or trk.time_since_update != 0:
                            continue
                        should_capture = False
                        conf_history = list(trk.confidence_history)
                        current_conf = trk.last_conf
                        streak = trk.hit_streak
                        history_len = len(conf_history)
                        if current_conf >= 0.75: should_capture = True
                        elif streak >= 5 and current_conf >= 0.5:
                            if history_len == 5 and conf_history[0] < current_conf: should_capture = True
                            elif history_len > 0 and all(c >= 0.45 for c in conf_history): should_capture = True
                        elif streak >= 2 and current_conf >= 0.5 and history_len > 0:
                            if all(c >= 0.45 for c in conf_history): should_capture = True
                        if streak >= 5 and current_conf < 0.4: should_capture = False
                        if streak >= 5 and history_len == 5:
                            if conf_history[0] >= 0.55 and current_conf <= 0.5: should_capture = False
                            elif current_conf < 0.6 and (conf_history[0] >= current_conf or (current_conf - conf_history[0]) < 0.1):
                                if not (current_conf >= 0.75): 
                                    is_trending_up = True
                                    if len(conf_history) >= 3: 
                                        if not (conf_history[-1] > conf_history[-2] > conf_history[-3]):
                                            is_trending_up = False
                                    if not is_trending_up: should_capture = False
                        if should_capture:
                            # ImageSaverThread necesita el frame actual. DetectorWorker.frame es el correcto.
                            image_to_save = self.frame.copy() 
                            bbox_for_saving = trk.get_state() 
                            bbox_int_tuple = tuple(map(int, bbox_for_saving))
                            # Aquí 'bbox_int_tuple' se pasa dos veces, uno como 'bbox' y otro como 'coordenadas'.
                            # Esto podría ser un error o un uso específico. Se mantiene por ahora.
                            saver = ImageSaverThread(image_to_save, bbox_int_tuple, trk.last_cls, bbox_int_tuple, self.model_key, current_conf)
                            saver.start()
                            self.recently_captured_track_ids.add(trk.id)
                    
                    output_for_signal = []
                    for trk in self.trackers: 
                        state_bbox = trk.get_state()
                        is_predicted_state = trk.time_since_update > 0 
                        output_for_signal.append(
                            (int(state_bbox[0]), int(state_bbox[1]), int(state_bbox[2]), int(state_bbox[3]), 
                             trk.id, trk.last_cls, trk.last_conf, is_predicted_state)
                        )
                    # Emitir la señal sin la copia del frame
                    self.result_ready.emit(output_for_signal, self.model_key)
            self.msleep(10) 

    def stop(self):
        print(f"🛑 Solicitando detener hilo {self.objectName() or id(self)}")
        self.running = False
        self.wait() 
        print(f"✅ Hilo detenido correctamente")
