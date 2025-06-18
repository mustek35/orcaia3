# core/detector_worker.py - VERSIÓN CORREGIDA
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from logging_utils import get_logger
from ultralytics import YOLO
import numpy as np
import time
import threading
import queue
import torch
from pathlib import Path
import os

logger = get_logger(__name__)

def iou(boxA, boxB):
    """Compute IoU between two boxes given as [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0
    return interArea / union

# Caché de modelos YOLO
yolo_model_cache = {}

# Rutas de modelos
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

class SimpleYOLOEngine:
    """Motor YOLO simple y confiable"""
    
    def __init__(self, model_key="Personas", confidence=0.5, imgsz=640, device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_key = model_key
        self.device = device
        self.confidence = confidence
        self.imgsz = imgsz
        self.model_classes = MODEL_CLASSES.get(model_key, [0])
        
        # Determinar ruta del modelo
        default_model_path = "yolov8n.pt"  # Fallback
        model_path = MODEL_PATHS.get(model_key, default_model_path)
        self.model_path_str = str(model_path)
        
        self.model = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=3)
        self.results_queue = queue.Queue()
        self.processing_thread = None
        
        logger.info(f"SimpleYOLOEngine creado: {model_key} en {device}")
    
    def load_model(self):
        """Cargar modelo YOLO"""
        try:
            logger.info(f"Cargando modelo {self.model_key}: {self.model_path_str}")
            
            if self.model_path_str in yolo_model_cache:
                self.model = yolo_model_cache[self.model_path_str]
                logger.info(f"Modelo {self.model_key} desde caché")
            else:
                self.model = YOLO(self.model_path_str)
                yolo_model_cache[self.model_path_str] = self.model
                logger.info(f"Modelo {self.model_key} cargado")
            
            # Test básico
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = self.model(test_img, verbose=False, imgsz=self.imgsz)
            
            return results is not None
                
        except Exception as e:
            logger.error(f"Error cargando modelo {self.model_key}: {e}")
            return False
    
    def start_processing(self):
        """Iniciar procesamiento"""
        if self.running or not self.model:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info(f"Procesamiento {self.model_key} iniciado")
    
    def stop_processing(self):
        """Detener procesamiento"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        logger.info(f"Procesamiento {self.model_key} detenido")
    
    def add_frame(self, frame):
        """Añadir frame para procesamiento"""
        if not self.running or frame is None:
            return
        
        try:
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame.copy())
            else:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Empty:
                    self.frame_queue.put_nowait(frame.copy())
        except Exception as e:
            logger.warning(f"Error añadiendo frame a {self.model_key}: {e}")
    
    def get_results(self, timeout=0.01):
        """Obtener resultados disponibles"""
        results = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.results_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def _processing_loop(self):
        """Loop principal de procesamiento"""
        while self.running:
            try:
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                detections = self._process_frame(frame)
                
                if detections:
                    result = {
                        'detections': detections,
                        'timestamp': time.time(),
                        'model_key': self.model_key
                    }
                    self.results_queue.put(result)
                
                self.frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error en loop {self.model_key}: {e}")
                time.sleep(0.1)
    
    def _process_frame(self, frame):
        """Procesar un frame individual"""
        try:
            if frame is None or frame.size == 0:
                return []
            
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.imgsz,
                classes=self.model_classes,
                verbose=False,
                save=False,
                show=False
            )
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        try:
                            box = boxes[i]
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy())
                            cls = int(box.cls[0].cpu().numpy())
                            
                            detection = {
                                'bbox': (int(x1), int(y1), int(x2), int(y2)),
                                'cls': cls,
                                'conf': conf
                            }
                            
                            detections.append(detection)
                            
                        except Exception as e:
                            logger.warning(f"Error procesando detección {i}: {e}")
                            continue
            
            return detections
            
        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
            return []


class DetectorWorker(QThread):
    """DetectorWorker usando SimpleYOLOEngine"""
    
    result_ready = pyqtSignal(list, str)
    
    def __init__(self, model_key="Personas", parent=None, frame_interval=1, confidence=0.5, imgsz=640, device=None, **kwargs):
        super().__init__(parent)
        
        self.model_key = model_key
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        self.setObjectName(f"DetectorWorker_{self.model_key}_{id(self)}")
        
        self.confidence = confidence
        self.imgsz = imgsz
        self.frame_interval = frame_interval
        
        self.yolo_engine = SimpleYOLOEngine(
            model_key=model_key,
            confidence=confidence,
            imgsz=imgsz,
            device=device
        )
        
        self.frame = None
        self.running = False
        self.frame_counter = 0
        
        self.result_timer = QTimer()
        self.result_timer.timeout.connect(self._poll_results)
        
        logger.info(f"DetectorWorker creado: {model_key} en {device}")
    
    def set_frame(self, frame, *args, **kwargs):
        """Establecer frame para procesamiento (compatible con API existente)"""
        if isinstance(frame, np.ndarray) and frame.size > 0:
            self.frame = frame
            logger.debug(f"Frame establecido para {self.model_key}: {frame.shape}")
        else:
            logger.warning(f"Frame inválido para {self.model_key}: {type(frame)}")
    
    def run(self):
        """Hilo principal del DetectorWorker"""
        self.running = True
        
        if not self.yolo_engine.load_model():
            logger.error(f"Error cargando modelo en {self.objectName()}")
            return
        
        self.yolo_engine.start_processing()
        self.result_timer.start(50)
        
        logger.info(f"DetectorWorker {self.model_key} iniciado")
        
        while self.running:
            if self.frame is not None:
                self.frame_counter += 1
                
                if self.frame_counter % self.frame_interval == 0:
                    current_frame = self.frame
                    self.frame = None
                    self.yolo_engine.add_frame(current_frame)
            
            self.msleep(10)
        
        self.result_timer.stop()
        self.yolo_engine.stop_processing()
        logger.info(f"DetectorWorker {self.model_key} detenido")
    
    def _poll_results(self):
        """Polling de resultados desde el motor YOLO"""
        if not self.running:
            return
        
        try:
            results = self.yolo_engine.get_results(timeout=0.01)
            
            for result in results:
                detections = result['detections']
                
                if detections:
                    output_detections = []
                    
                    for det in detections:
                        output_det = {
                            'bbox': det['bbox'],
                            'cls': det['cls'], 
                            'conf': det['conf']
                        }
                        output_detections.append(output_det)
                    
                    self.result_ready.emit(output_detections, self.model_key)
                    
        except Exception as e:
            logger.error(f"Error en polling {self.model_key}: {e}")
    
    def stop(self):
        """Detener DetectorWorker"""
        logger.info(f"Deteniendo {self.objectName()}")
        self.running = False
        
        if hasattr(self, 'yolo_engine'):
            self.yolo_engine.stop_processing()
        
        if hasattr(self, 'result_timer'):
            self.result_timer.stop()
        
        self.wait()
        logger.info(f"DetectorWorker {self.model_key} detenido")
