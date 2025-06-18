"""
Detector Worker optimizado para RTX 3050 con pipeline CUDA eficiente.
Reemplaza el DetectorWorker original con mejor gestión de memoria y threading.
"""

import torch
import numpy as np
import time
import threading
from collections import deque
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from ultralytics import YOLO
from core.advanced_tracker import AdvancedTracker
from logging_utils import get_logger

logger = get_logger(__name__)

# Caché global de modelos optimizado
model_cache = {}
model_cache_lock = threading.Lock()

# Configuración optimizada para RTX 3050
CUDA_CONFIG = {
    'device': 'cuda:0',
    'half_precision': True,     # FP16 para mejor rendimiento
    'batch_size': 4,           # Óptimo para 4GB VRAM
    'max_det': 300,            # Máximo detecciones por imagen
    'agnostic_nms': True,      # NMS más rápido
    'augment': False,          # Sin aumentos para velocidad
    'verbose': False,          # Sin logs verbosos
    'save': False,             # No guardar resultados
    'save_txt': False,         # No guardar texto
    'save_conf': False,        # No guardar confianza
    'save_crop': False,        # No guardar crops
    'show': False,             # No mostrar ventanas
    'vid_stride': 1,           # Procesar cada frame
}

# Rutas de modelos actualizadas
BASE_MODEL_PATH = Path(__file__).resolve().parent / "models"
MODEL_PATHS = {
    "Embarcaciones": BASE_MODEL_PATH / "best.pt",
    "Personas": BASE_MODEL_PATH / "yolov8m.pt", 
    "Autos": BASE_MODEL_PATH / "yolov8m.pt",
    "Barcos": BASE_MODEL_PATH / "yolov8m.pt",
}

# Clases por modelo
MODEL_CLASSES = {
    "Embarcaciones": [0],
    "Personas": [0],
    "Autos": [2], 
    "Barcos": [8]
}

class OptimizedDetectorWorker(QThread):
    """
    Detector Worker optimizado con:
    - Pipeline CUDA eficiente
    - Gestión de memoria mejorada
    - Threading optimizado
    - Caché inteligente de modelos
    """
    
    # Señales
    result_ready = pyqtSignal(list, str)  # resultados, model_key
    performance_stats = pyqtSignal(dict)  # estadísticas de rendimiento
    error_occurred = pyqtSignal(str)      # errores
    
    def __init__(self, model_key="Personas", confidence=0.5, imgsz=640, 
                 device=None, enable_tracking=True, **kwargs):
        super().__init__()
        
        # Configuración básica
        self.model_key = model_key
        self.confidence = confidence
        self.imgsz = imgsz
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_tracking = enable_tracking
        
        # Configuración CUDA
        if self.device.startswith('cuda'):
            self._setup_cuda_optimizations()
        
        # Estado del worker
        self._running = False
        self._paused = False
        
        # Buffer de frames thread-safe
        self._frame_buffer = deque(maxlen=3)
        self._buffer_lock = threading.Lock()
        
        # Modelo y tracker
        self.model = None
        self.tracker = None
        
        # Estadísticas de rendimiento
        self._stats = {
            'fps': 0.0,
            'inference_time': 0.0,
            'tracking_time': 0.0,
            'total_time': 0.0,
            'frames_processed': 0,
            'memory_usage': 0.0,
            'gpu_utilization': 0.0,
            'queue_size': 0
        }
        
        # Contadores
        self._frame_count = 0
        self._last_stats_time = time.time()
        
        # Configurar nombre del objeto
        self.setObjectName(f"OptDetector_{model_key}_{id(self)}")
        
        logger.info(f"{self.objectName()}: Inicializado para {model_key} en {self.device}")
    
    def _setup_cuda_optimizations(self):
        """Configura optimizaciones específicas para CUDA"""
        try:
            # Configurar PyTorch para RTX 3050
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Configurar memoria CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Reservar espacio para el modelo
                torch.cuda.set_per_process_memory_fraction(0.9)
                
            logger.info(f"{self.objectName()}: Optimizaciones CUDA aplicadas")
            
        except Exception as e:
            logger.warning(f"{self.objectName()}: Error aplicando optimizaciones CUDA: {e}")
    
    def _load_model(self):
        """Carga el modelo YOLO con caché inteligente"""
        model_path = MODEL_PATHS.get(self.model_key)
        if not model_path or not model_path.exists():
            # Fallback al modelo por defecto
            model_path = BASE_MODEL_PATH / "yolov8n.pt"
            if not model_path.exists():
                raise FileNotFoundError(f"No se encontró modelo para {self.model_key}")
        
        model_path_str = str(model_path)
        
        # Usar caché thread-safe
        with model_cache_lock:
            if model_path_str in model_cache:
                self.model = model_cache[model_path_str]
                logger.info(f"{self.objectName()}: Modelo cargado desde caché")
                return
        
        try:
            logger.info(f"{self.objectName()}: Cargando modelo desde {model_path}")
            
            # Cargar modelo con configuración optimizada
            self.model = YOLO(model_path_str)
            
            # Configurar para CUDA si está disponible
            if self.device.startswith('cuda'):
                self.model.to(self.device)
                
                # Compilar modelo para mejor rendimiento (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    try:
                        self.model.model = torch.compile(self.model.model)
                        logger.info(f"{self.objectName()}: Modelo compilado con torch.compile")
                    except Exception as e:
                        logger.warning(f"No se pudo compilar el modelo: {e}")
            
            # Warmup del modelo
            self._warmup_model()
            
            # Agregar al caché
            with model_cache_lock:
                model_cache[model_path_str] = self.model
            
            logger.info(f"{self.objectName()}: Modelo cargado y optimizado")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error cargando modelo: {e}")
            raise
    
    def _warmup_model(self):
        """Realiza warmup del modelo para optimizar rendimiento"""
        try:
            logger.info(f"{self.objectName()}: Realizando warmup del modelo...")
            
            # Crear imagen dummy
            dummy_img = np.random.randint(0, 255, (self.imgsz, self.imgsz, 3), dtype=np.uint8)
            
            # Ejecutar varias inferencias de warmup
            model_classes = MODEL_CLASSES.get(self.model_key, [0])
            
            for i in range(3):
                _ = self.model.predict(
                    source=dummy_img,
                    classes=model_classes,
                    conf=self.confidence,
                    imgsz=self.imgsz,
                    device=self.device,
                    **CUDA_CONFIG
                )
            
            logger.info(f"{self.objectName()}: Warmup completado")
            
        except Exception as e:
            logger.warning(f"{self.objectName()}: Error en warmup: {e}")
    
    def _setup_tracker(self):
        """Configura el tracker optimizado"""
        if not self.enable_tracking:
            return
        
        try:
            # Configuración específica por modelo
            tracker_config = {
                'max_age': 30,
                'n_init': 3,
                'conf_threshold': self.confidence,
                'device': self.device,
                'lost_ttl': 5,
                'enable_size_control': True,
                'enable_velocity_prediction': True
            }
            
            # Ajustes específicos por tipo de objeto
            if self.model_key == "Personas":
                tracker_config.update({
                    'lost_ttl': 10,
                    'max_age': 40
                })
            elif self.model_key in ["Barcos", "Embarcaciones"]:
                tracker_config.update({
                    'lost_ttl': 15,
                    'max_age': 50
                })
            
            self.tracker = AdvancedTracker(**tracker_config)
            logger.info(f"{self.objectName()}: Tracker configurado")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error configurando tracker: {e}")
            self.tracker = None
    
    def start_detection(self):
        """Inicia el worker de detección"""
        if self._running:
            logger.warning(f"{self.objectName()}: Ya está ejecutándose")
            return
        
        try:
            # Cargar modelo y configurar tracker
            self._load_model()
            self._setup_tracker()
            
            # Iniciar thread
            self._running = True
            self.start()
            
            logger.info(f"{self.objectName()}: Detección iniciada")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error iniciando detección: {e}")
            self.error_occurred.emit(str(e))
    
    def stop_detection(self):
        """Detiene el worker de detección"""
        logger.info(f"{self.objectName()}: Deteniendo detección...")
        
        self._running = False
        
        # Esperar a que termine el thread
        if self.isRunning():
            if not self.wait(5000):  # 5 segundos de timeout
                logger.warning(f"{self.objectName()}: Timeout esperando detención")
                self.terminate()
        
        # Limpiar recursos
        self._cleanup_resources()
        
        logger.info(f"{self.objectName()}: Detección detenida")
    
    def _cleanup_resources(self):
        """Limpia recursos GPU y memoria"""
        try:
            # Limpiar buffer
            with self._buffer_lock:
                self._frame_buffer.clear()
            
            # Limpiar memoria CUDA si se usa
            if self.device.startswith('cuda') and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.debug(f"{self.objectName()}: Recursos limpiados")
            
        except Exception as e:
            logger.warning(f"{self.objectName()}: Error limpiando recursos: {e}")
    
    def pause_detection(self):
        """Pausa la detección temporalmente"""
        self._paused = True
        logger.info(f"{self.objectName()}: Detección pausada")
    
    def resume_detection(self):
        """Reanuda la detección"""
        self._paused = False
        logger.info(f"{self.objectName()}: Detección reanudada")
    
    def add_frame(self, frame):
        """Agrega un frame al buffer para procesamiento"""
        if not self._running or frame is None:
            return
        
        try:
            # Validar frame
            if not isinstance(frame, np.ndarray) or frame.size == 0:
                logger.warning(f"{self.objectName()}: Frame inválido recibido")
                return
            
            # Preparar frame data
            frame_data = {
                'frame': frame.copy(),
                'timestamp': time.time(),
                'frame_id': self._frame_count
            }
            
            # Agregar al buffer thread-safe
            with self._buffer_lock:
                # Si el buffer está lleno, remover el más antiguo
                if len(self._frame_buffer) >= self._frame_buffer.maxlen:
                    dropped = self._frame_buffer.popleft()
                    logger.debug(f"{self.objectName()}: Frame {dropped['frame_id']} descartado")
                
                self._frame_buffer.append(frame_data)
                self._stats['queue_size'] = len(self._frame_buffer)
            
            self._frame_count += 1
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error agregando frame: {e}")
    
    def run(self):
        """Loop principal del worker"""
        logger.info(f"{self.objectName()}: Iniciando loop de detección")
        
        while self._running:
            try:
                # Verificar si está pausado
                if self._paused:
                    time.sleep(0.1)
                    continue
                
                # Obtener frame del buffer
                frame_data = None
                with self._buffer_lock:
                    if self._frame_buffer:
                        frame_data = self._frame_buffer.popleft()
                        self._stats['queue_size'] = len(self._frame_buffer)
                
                if frame_data is None:
                    time.sleep(0.01)  # Sleep corto si no hay frames
                    continue
                
                # Procesar frame
                self._process_frame(frame_data)
                
            except Exception as e:
                logger.error(f"{self.objectName()}: Error en loop principal: {e}")
                time.sleep(0.1)
        
        logger.info(f"{self.objectName()}: Loop de detección terminado")
    
    def _process_frame(self, frame_data):
        """Procesa un frame individual"""
        start_time = time.time()
        
        try:
            frame = frame_data['frame']
            frame_id = frame_data['frame_id']
            
            # Preprocesar frame
            processed_frame = self._preprocess_frame(frame)
            
            # Ejecutar inferencia
            inference_start = time.time()
            detections = self._run_inference(processed_frame)
            inference_time = time.time() - inference_start
            
            # Tracking si está habilitado
            tracking_start = time.time()
            if self.tracker and detections:
                tracked_results = self._run_tracking(detections, frame)
            else:
                tracked_results = self._format_detections(detections, frame.shape[:2])
            
            tracking_time = time.time() - tracking_start
            
            # Emitir resultados
            if tracked_results:
                self.result_ready.emit(tracked_results, self.model_key)
            
            # Actualizar estadísticas
            total_time = time.time() - start_time
            self._update_performance_stats(inference_time, tracking_time, total_time)
            
            logger.debug(f"{self.objectName()}: Frame {frame_id} procesado en {total_time*1000:.1f}ms")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error procesando frame: {e}")
    
    def _preprocess_frame(self, frame):
        """Preprocesa el frame para inferencia"""
        try:
            # Redimensionar si es necesario
            if frame.shape[:2] != (self.imgsz, self.imgsz):
                # Usar interpolación rápida para redimensionar
                processed = cv2.resize(frame, (self.imgsz, self.imgsz), 
                                     interpolation=cv2.INTER_LINEAR)
            else:
                processed = frame.copy()
            
            return processed
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error en preprocesamiento: {e}")
            return frame
    
    def _run_inference(self, frame):
        """Ejecuta la inferencia del modelo"""
        try:
            model_classes = MODEL_CLASSES.get(self.model_key, [0])
            
            # Configurar parámetros de inferencia optimizados
            inference_params = {
                'source': frame,
                'classes': model_classes,
                'conf': self.confidence,
                'imgsz': self.imgsz,
                'device': self.device,
                **CUDA_CONFIG
            }
            
            # Ejecutar inferencia
            results = self.model.predict(**inference_params)
            
            # Procesar resultados
            if results and len(results) > 0:
                return self._extract_detections(results[0], frame.shape[:2])
            
            return []
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error en inferencia: {e}")
            return []
    
    def _extract_detections(self, result, original_shape):
        """Extrae detecciones del resultado YOLO"""
        detections = []
        
        try:
            if not hasattr(result, 'boxes') or result.boxes is None:
                return detections
            
            boxes = result.boxes
            if len(boxes) == 0:
                return detections
            
            # Escalar coordenadas al tamaño original
            scale_x = original_shape[1] / self.imgsz
            scale_y = original_shape[0] / self.imgsz
            
            for i, box in enumerate(boxes):
                # Extraer coordenadas
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Escalar al tamaño original
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Validar coordenadas
                x1 = max(0, min(x1, original_shape[1] - 1))
                y1 = max(0, min(y1, original_shape[0] - 1))
                x2 = max(0, min(x2, original_shape[1] - 1))
                y2 = max(0, min(y2, original_shape[0] - 1))
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                # Extraer clase y confianza
                cls = int(box.cls[0].cpu().numpy())
                conf = float(box.conf[0].cpu().numpy())
                
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'cls': cls,
                    'conf': conf
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error extrayendo detecciones: {e}")
            return []
    
    def _run_tracking(self, detections, frame):
        """Ejecuta el tracking sobre las detecciones"""
        try:
            if not self.tracker:
                return self._format_detections(detections, frame.shape[:2])
            
            # Ejecutar tracking
            tracked_results = self.tracker.update(detections, frame=frame)
            
            return tracked_results
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error en tracking: {e}")
            return self._format_detections(detections, frame.shape[:2])
    
    def _format_detections(self, detections, frame_shape):
        """Formatea detecciones cuando no hay tracking"""
        formatted = []
        
        for i, det in enumerate(detections):
            formatted.append({
                'bbox': det['bbox'],
                'id': f"det_{i}",  # ID temporal
                'cls': det['cls'],
                'conf': det['conf'],
                'centers': [],  # Sin historial
                'moving': None,  # Sin información de movimiento
            })
        
        return formatted
    
    def _update_performance_stats(self, inference_time, tracking_time, total_time):
        """Actualiza estadísticas de rendimiento"""
        current_time = time.time()
        
        # Actualizar tiempos
        self._stats['inference_time'] = inference_time * 1000  # ms
        self._stats['tracking_time'] = tracking_time * 1000   # ms
        self._stats['total_time'] = total_time * 1000         # ms
        
        # Calcular FPS
        time_diff = current_time - self._last_stats_time
        if time_diff >= 1.0:  # Actualizar cada segundo
            frame_count = getattr(self, '_frames_processed_last_sec', 0)
            self._stats['fps'] = frame_count / time_diff
            self._frames_processed_last_sec = 0
            self._last_stats_time = current_time
            
            # Obtener uso de memoria GPU si está disponible
            if self.device.startswith('cuda') and torch.cuda.is_available():
                try:
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    self._stats['memory_usage'] = memory_allocated
                    self._stats['memory_reserved'] = memory_reserved
                except:
                    pass
            
            # Emitir estadísticas
            self.performance_stats.emit(self._stats.copy())
        
        # Incrementar contador
        self._frames_processed_last_sec = getattr(self, '_frames_processed_last_sec', 0) + 1
    
    def get_stats(self):
        """Obtiene estadísticas actuales"""
        return self._stats.copy()
    
    def update_confidence(self, new_confidence):
        """Actualiza el umbral de confianza"""
        old_conf = self.confidence
        self.confidence = max(0.0, min(1.0, new_confidence))
        
        if self.tracker:
            self.tracker.conf_threshold = self.confidence
        
        logger.info(f"{self.objectName()}: Confianza actualizada de {old_conf} a {self.confidence}")
    
    def get_model_info(self):
        """Obtiene información del modelo actual"""
        info = {
            'model_key': self.model_key,
            'model_path': str(MODEL_PATHS.get(self.model_key, 'N/A')),
            'classes': MODEL_CLASSES.get(self.model_key, []),
            'confidence': self.confidence,
            'imgsz': self.imgsz,
            'device': self.device,
            'tracking_enabled': self.enable_tracking,
            'model_loaded': self.model is not None,
            'tracker_active': self.tracker is not None
        }
        return info


class DetectorManager:
    """
    Manager para múltiples detectores optimizados.
    Gestiona la creación, configuración y lifecycle de detectores.
    """
    
    def __init__(self):
        self.detectors = {}
        self.detector_configs = {}
        
    def create_detector(self, detector_id, model_key, config=None):
        """Crea un nuevo detector optimizado"""
        if detector_id in self.detectors:
            logger.warning(f"Detector {detector_id} ya existe. Reemplazando...")
            self.remove_detector(detector_id)
        
        # Configuración por defecto optimizada para RTX 3050
        default_config = {
            'confidence': 0.5,
            'imgsz': 640,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'enable_tracking': True
        }
        
        # Combinar con configuración personalizada
        final_config = default_config.copy()
        if config:
            final_config.update(config)
        
        try:
            detector = OptimizedDetectorWorker(
                model_key=model_key,
                **final_config
            )
            
            self.detectors[detector_id] = detector
            self.detector_configs[detector_id] = final_config
            
            logger.info(f"Detector {detector_id} creado para modelo {model_key}")
            return detector
            
        except Exception as e:
            logger.error(f"Error creando detector {detector_id}: {e}")
            return None
    
    def get_detector(self, detector_id):
        """Obtiene un detector por ID"""
        return self.detectors.get(detector_id)
    
    def remove_detector(self, detector_id):
        """Remueve y limpia un detector"""
        if detector_id in self.detectors:
            detector = self.detectors[detector_id]
            detector.stop_detection()
            
            del self.detectors[detector_id]
            if detector_id in self.detector_configs:
                del self.detector_configs[detector_id]
            
            logger.info(f"Detector {detector_id} removido")
    
    def stop_all_detectors(self):
        """Detiene todos los detectores"""
        logger.info("Deteniendo todos los detectores...")
        
        for detector_id in list(self.detectors.keys()):
            self.remove_detector(detector_id)
        
        logger.info("Todos los detectores detenidos")
    
    def get_active_detectors(self):
        """Obtiene lista de detectores activos"""
        return list(self.detectors.keys())
    
    def get_detector_stats(self, detector_id):
        """Obtiene estadísticas de un detector"""
        detector = self.detectors.get(detector_id)
        if detector:
            return detector.get_stats()
        return None


# Instancia global del manager
detector_manager = DetectorManager()


# Ejemplo de uso optimizado
if __name__ == "__main__":
    import cv2
    
    # Configuración de prueba
    test_config = {
        'confidence': 0.3,
        'imgsz': 640,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'enable_tracking': True
    }
    
    # Crear detector
    detector = detector_manager.create_detector(
        detector_id="test_detector",
        model_key="Personas", 
        config=test_config
    )
    
    if detector:
        # Conectar señales
        detector.result_ready.connect(lambda results, model: print(f"Detecciones: {len(results)}"))
        detector.performance_stats.connect(lambda stats: print(f"FPS: {stats['fps']:.1f}"))
        detector.error_occurred.connect(lambda err: print(f"Error: {err}"))
        
        # Iniciar detección
        detector.start_detection()
        
        # Simular frames (en aplicación real vendrían del video reader)
        test_frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        
        for i in range(10):
            detector.add_frame(test_frame)
            time.sleep(0.1)
        
        # Detener
        detector.stop_detection()
        
        # Limpiar
        detector_manager.stop_all_detectors()
        
        print("Test completado")