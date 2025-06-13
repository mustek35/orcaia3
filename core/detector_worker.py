from PyQt6.QtCore import QThread, pyqtSignal
from logging_utils import get_logger
from ultralytics import YOLO
import numpy as np
from core.advanced_tracker import AdvancedTracker
from gui.image_saver import ImageSaverThread
import os  # <--- IMPORT OS AÑADIDO
from pathlib import Path

logger = get_logger(__name__)

# Caché de modelos YOLO a nivel de módulo
yolo_model_cache = {}

# Ajustar la ruta base usando la ubicación de este archivo
_BASE_MODEL_PATH = Path(__file__).resolve().parent / "models"

MODEL_PATHS = {
    "Embarcaciones": _BASE_MODEL_PATH / "best.pt",
    "Personas": _BASE_MODEL_PATH / "yolov8m.pt",
    "Autos": _BASE_MODEL_PATH / "yolov8m.pt",
    "Barcos": _BASE_MODEL_PATH / "yolov8m.pt",
}

MODEL_CLASSES = {
    "Embarcaciones": [0],
    "Personas": [0],
    "Autos": [2],
    "Barcos": [8]
}

# Mapear las clases predichas por cada modelo a una clase unificada
CLASS_REMAP = {
    "Embarcaciones": {0: 1},
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

    result_ready = pyqtSignal(list, str, int)


    def __init__(self, model_key="Personas", parent=None, frame_interval=1, confidence=0.5, imgsz=640, device=None, track=True, lost_ttl=5):
        super().__init__(parent)
        self.model_key = model_key
        if device is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.setObjectName(f"DetectorWorker_{self.model_key}_{id(self)}")

        logger.info(f"{self.objectName()}: Usando {'GPU' if self.device.startswith('cuda') else 'CPU'} para el modelo '{self.model_key}'")

        default_model_path = _BASE_MODEL_PATH / "yolov8n.pt"
        model_path = MODEL_PATHS.get(model_key, default_model_path)

        model_classes_for_key = MODEL_CLASSES.get(model_key)
        if model_classes_for_key is None:
            logger.warning("%s: model_key '%s' no encontrado en MODEL_CLASSES. Usando default [0].", self.objectName(), model_key)
            self.model_classes = [0]
        else:
            self.model_classes = model_classes_for_key

        model_path_str = str(model_path)
        logger.info("YOLO: solicitando modelo '%s' (real %s) desde %s para %s", self.model_key, os.path.basename(model_path_str), model_path_str, self.objectName())
        if model_path_str in yolo_model_cache:
            self.model = yolo_model_cache[model_path_str]
            logger.info("YOLO Cache: usando modelo '%s' (Path: %s) desde caché para %s", self.model_key, model_path_str, self.objectName())
        else:
            try:
                self.model = YOLO(model_path_str)
                try:
                    self.model.to(self.device)
                except Exception:
                    logger.warning("%s: model.to(%s) failed; relying on predict device parameter", self.objectName(), self.device)
                yolo_model_cache[model_path_str] = self.model
                logger.info("YOLO Cache: modelo '%s' (Path: %s) cargado y añadido a caché para %s", self.model_key, model_path_str, self.objectName())
            except Exception as e:
                logger.error("Failed to load model %s for %s: %s", model_path_str, self.objectName(), e)
                raise e

        self.confidence = confidence
        self.imgsz = imgsz
        self.track = track
        self.lost_ttl = lost_ttl
        logger.debug(
            "%s: Initialized with model_key=%s path=%s classes=%s conf=%s imgsz=%s track=%s lost_ttl=%s",
            self.objectName(),
            self.model_key,
            model_path_str,
            self.model_classes,
            self.confidence,
            self.imgsz,
            self.track,
            self.lost_ttl,
        )

        self.frame = None

        self.frame_id = None

        self.running = False
        if self.track:
            self.tracker = AdvancedTracker(
                conf_threshold=self.confidence,
                device=self.device,
                lost_ttl=self.lost_ttl,
            )
        else:
            self.tracker = None
        self.recently_captured_track_ids = set()
        self.active_savers = []

    def _remove_active_saver(self, saver_instance):
        if saver_instance in self.active_savers:
            self.active_savers.remove(saver_instance)

    def set_frame(self, frame, frame_id=None):
        logger.debug("%s: set_frame called. type=%s is_ndarray=%s", self.objectName(), type(frame), isinstance(frame, np.ndarray))
        if isinstance(frame, np.ndarray):
            logger.debug("%s: Frame shape %s id=%s", self.objectName(), frame.shape, frame_id)
            self.frame = frame
            self.frame_id = frame_id

    def run(self):
        self.running = True
        if not hasattr(self, 'model'):
            logger.error("%s: Modelo no cargado. Deteniendo hilo", self.objectName())
            return

        while self.running:
            if self.frame is not None:
                logger.debug("%s: Processing new frame", self.objectName())
                current_frame_to_process = self.frame
                current_frame_id = self.frame_id if self.frame_id is not None else 0
                frame_h, frame_w = current_frame_to_process.shape[:2]
                self.frame = None
                self.frame_id = None
                try:
                    logger.debug("%s: Calling model.predict classes=%s conf=%s imgsz=%s", self.objectName(), self.model_classes, self.confidence, self.imgsz)
                    yolo_results = self.model.predict(source=current_frame_to_process, classes=self.model_classes, conf=self.confidence, imgsz=self.imgsz, verbose=False, device=self.device)[0]
                    logger.debug("%s: model.predict successful. Raw boxes count %s", self.objectName(), len(yolo_results.boxes) if yolo_results else 'N/A')
                except Exception as e:
                    logger.error("%s: error durante model.predict: %s", self.objectName(), e)
                    self.msleep(100)
                    continue

                current_detections = []
                for r in yolo_results.boxes:
                    x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
                    x1 = max(0, min(x1, frame_w - 1))
                    y1 = max(0, min(y1, frame_h - 1))
                    x2 = max(0, min(x2, frame_w - 1))
                    y2 = max(0, min(y2, frame_h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cls = int(r.cls[0])
                    conf = float(r.conf[0])
                    remap_dict = CLASS_REMAP.get(self.model_key)
                    if remap_dict:
                        cls = remap_dict.get(cls, cls)
                    current_detections.append({'bbox': [x1, y1, x2, y2], 'cls': cls, 'conf': conf})

                if self.track:
                    tracks = self.tracker.update(current_detections, frame=current_frame_to_process)

                    output_for_signal = []
                    for trk in tracks:
                        x1, y1, x2, y2 = map(int, trk['bbox'])
                        x1 = max(0, min(x1, frame_w - 1))
                        y1 = max(0, min(y1, frame_h - 1))
                        x2 = max(0, min(x2, frame_w - 1))
                        y2 = max(0, min(y2, frame_h - 1))
                        if x2 <= x1 or y2 <= y1:
                            continue
                        output_for_signal.append({
                            'bbox': (x1, y1, x2, y2),
                            'id': trk['id'],
                            'cls': trk['cls'],
                            'conf': trk['conf'],
                            'centers': trk['centers'],
                            'moving': trk.get('moving'),
                        })
                else:
                    # Emit raw detections without tracking
                    output_for_signal = [
                        {
                            'bbox': (int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                            'cls': d['cls'],
                            'conf': d['conf'],
                        }
                        for d in current_detections
                        for b in [d['bbox']]
                    ]


                self.result_ready.emit(output_for_signal, self.model_key, current_frame_id)


            self.msleep(10)

    def stop(self):
        logger.info("%s: solicitando detener hilo", self.objectName() or id(self))
        self.running = False
        logger.info("%s: Esperando %d ImageSaverThread(s) activos", self.objectName(), len(self.active_savers))
        for saver_thread in list(self.active_savers):
            if saver_thread.isRunning():
                logger.info("%s: Esperando a ImageSaver %s", self.objectName(), saver_thread.objectName())
                if saver_thread.wait(2000):
                    logger.info("%s: ImageSaver %s terminó", self.objectName(), saver_thread.objectName())
                else:
                    logger.warning("%s: Timeout esperando a ImageSaver %s", self.objectName(), saver_thread.objectName())
            else:
                self._remove_active_saver(saver_thread)
        if self.active_savers:
            logger.warning("%s: %d ImageSaverThread(s) aún activos después de esperar", self.objectName(), len(self.active_savers))
        else:
            logger.info("%s: Todos los ImageSaverThread(s) activos han sido procesados", self.objectName())
        self.wait()
        logger.info("%s: hilo detenido correctamente", self.objectName())