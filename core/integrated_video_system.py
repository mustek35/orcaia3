"""
Sistema de video integrado que combina GStreamer + CUDA Pipeline.
Implementa la integración completa de Fase 2 para máximo rendimiento.
"""

import numpy as np
import time
import threading
from enum import Enum
from typing import Optional, Dict, List, Callable
from collections import deque
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap

from core.gstreamer_video_reader import GStreamerVideoReader, GStreamerVideoReaderFactory
from core.cuda_pipeline_processor import CUDAPipelineProcessor, cuda_pipeline_manager
from core.advanced_tracker import AdvancedTracker
from logging_utils import get_logger

logger = get_logger(__name__)

class VideoSystemMode(Enum):
    """Modos de operación del sistema de video"""
    BALANCED = "balanced"           # Balance entre latencia y calidad
    ULTRA_LOW_LATENCY = "ultra_low_latency"  # Latencia mínima
    HIGH_QUALITY = "high_quality"   # Máxima calidad
    POWER_SAVE = "power_save"       # Ahorro de energía
    STRESS_TEST = "stress_test"     # Test de estrés

class IntegratedVideoSystem(QObject):
    """
    Sistema de video integrado que combina:
    - GStreamer para captura con NVDEC
    - CUDA Pipeline para procesamiento IA
    - Advanced Tracker para seguimiento
    - Gestión automática de recursos
    """
    
    # Señales principales
    detection_results = pyqtSignal(list, dict)    # detecciones, metadata
    display_frame = pyqtSignal(QPixmap)           # frame para UI
    system_stats = pyqtSignal(dict)               # estadísticas del sistema
    error_occurred = pyqtSignal(str)              # errores críticos
    mode_changed = pyqtSignal(str)                # cambio de modo
    
    def __init__(self, system_id: str, config: Dict = None):
        super().__init__()
        
        self.system_id = system_id
        self.config = config or self._get_default_config()
        
        # Componentes del sistema
        self.video_reader: Optional[GStreamerVideoReader] = None
        self.cuda_processor: Optional[CUDAPipelineProcessor] = None
        self.tracker: Optional[AdvancedTracker] = None
        
        # Estado del sistema
        self.current_mode = VideoSystemMode.BALANCED
        self.running = False
        self.rtsp_url = None
        
        # Sincronización de componentes
        self.frame_sync_queue = deque(maxlen=10)
        self.sync_lock = threading.Lock()
        
        # Estadísticas integradas
        self.integrated_stats = {
            'system_mode': self.current_mode.value,
            'total_latency_ms': 0.0,
            'video_capture_fps': 0.0,
            'cuda_processing_fps': 0.0,
            'tracking_fps': 0.0,
            'end_to_end_fps': 0.0,
            'memory_usage_total': 0.0,
            'cpu_usage': 0.0,
            'gpu_utilization': 0.0,
            'active_tracks': 0,
            'detection_accuracy': 0.0,
            'system_health': 'unknown',
            'uptime_seconds': 0.0
        }
        
        # Métricas de rendimiento
        self.performance_tracker = PerformanceTracker()
        
        # Timer para actualización de estadísticas
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_integrated_stats)
        
        # Tiempo de inicio
        self.start_time = time.time()
        
        logger.info(f"IntegratedVideoSystem '{system_id}' inicializado")
    
    def _get_default_config(self):
        """Configuración por defecto del sistema integrado"""
        return {
            # Configuración de video
            'video': {
                'performance_profile': 'balanced',
                'enable_gpu_decode': True,
                'target_latency_ms': 50,
                'buffer_size': 2,
            },
            
            # Configuración CUDA
            'cuda': {
                'batch_size': 4,
                'confidence_threshold': 0.5,
                'input_size': (640, 640),
                'half_precision': True,
                'enable_adaptive_batch': True,
            },
            
            # Configuración de tracking
            'tracking': {
                'enable_tracking': True,
                'max_age': 30,
                'n_init': 3,
                'lost_ttl': 5,
                'enable_size_control': True,
                'enable_velocity_prediction': True,
            },
            
            # Configuración del sistema
            'system': {
                'auto_mode_switching': True,
                'health_monitoring': True,
                'stats_update_interval': 2000,  # ms
                'auto_reconnect': True,
                'max_reconnect_attempts': 10,
            }
        }
    
    def initialize_system(self, rtsp_url: str, camera_type: str = "fija"):
        """Inicializa todos los componentes del sistema"""
        try:
            logger.info(f"Inicializando sistema para {rtsp_url}")
            
            self.rtsp_url = rtsp_url
            
            # 1. Inicializar video reader con GStreamer
            self._initialize_video_reader(rtsp_url, camera_type)
            
            # 2. Inicializar procesador CUDA
            self._initialize_cuda_processor()
            
            # 3. Inicializar tracker
            self._initialize_tracker()
            
            # 4. Configurar sincronización entre componentes
            self._setup_component_sync()
            
            # 5. Aplicar modo de operación
            self._apply_operation_mode(self.current_mode)
            
            logger.info("Sistema integrado inicializado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando sistema: {e}")
            self.error_occurred.emit(f"Error de inicialización: {e}")
            return False
    
    def _initialize_video_reader(self, rtsp_url: str, camera_type: str):
        """Inicializa el lector de video GStreamer"""
        try:
            # Crear video reader según el modo actual
            performance_profile = self._get_video_profile_for_mode(self.current_mode)
            
            self.video_reader = GStreamerVideoReaderFactory.create_reader(
                rtsp_url=rtsp_url,
                camera_type=camera_type,
                performance_profile=performance_profile
            )
            
            # Conectar señales
            self.video_reader.frame_ready.connect(self._on_video_frame)
            self.video_reader.display_ready.connect(self._on_display_frame)
            self.video_reader.stats_updated.connect(self._on_video_stats)
            self.video_reader.error_occurred.connect(self._on_video_error)
            
            logger.info("Video reader GStreamer inicializado")
            
        except Exception as e:
            logger.error(f"Error inicializando video reader: {e}")
            raise
    
    def _initialize_cuda_processor(self):
        """Inicializa el procesador CUDA"""
        try:
            # Configuración del modelo
            model_config = {
                **self.config['cuda'],
                'model_path': self._get_model_path_for_mode(self.current_mode),
            }
            
            # Crear procesador usando el manager
            self.cuda_processor = cuda_pipeline_manager.create_processor(
                processor_id=self.system_id,
                model_config=model_config
            )
            
            if not self.cuda_processor:
                raise RuntimeError("No se pudo crear procesador CUDA")
            
            # Conectar señales
            self.cuda_processor.results_ready.connect(self._on_cuda_results)
            self.cuda_processor.performance_stats.connect(self._on_cuda_stats)
            self.cuda_processor.error_occurred.connect(self._on_cuda_error)
            
            # Cargar modelo
            self.cuda_processor.load_model()
            
            logger.info("Procesador CUDA inicializado")
            
        except Exception as e:
            logger.error(f"Error inicializando procesador CUDA: {e}")
            raise
    
    def _initialize_tracker(self):
        """Inicializa el tracker avanzado"""
        try:
            if not self.config['tracking']['enable_tracking']:
                logger.info("Tracking deshabilitado por configuración")
                return
            
            # Configuración del tracker
            tracker_config = {
                **self.config['tracking'],
                'conf_threshold': self.config['cuda']['confidence_threshold'],
                'device': 'cuda' if self.cuda_processor else 'cpu',
            }
            
            self.tracker = AdvancedTracker(**tracker_config)
            
            logger.info("Tracker avanzado inicializado")
            
        except Exception as e:
            logger.error(f"Error inicializando tracker: {e}")
            # Tracker es opcional, no fallar el sistema completo
            self.tracker = None
    
    def _setup_component_sync(self):
        """Configura la sincronización entre componentes"""
        try:
            # Timer para estadísticas integradas
            interval = self.config['system']['stats_update_interval']
            self.stats_timer.start(interval)
            
            # Configurar performance tracker
            self.performance_tracker.start_monitoring()
            
            logger.info("Sincronización de componentes configurada")
            
        except Exception as e:
            logger.error(f"Error configurando sincronización: {e}")
            raise
    
    def start_system(self):
        """Inicia el sistema completo"""
        if self.running:
            logger.warning("Sistema ya está ejecutándose")
            return True
        
        try:
            logger.info("Iniciando sistema integrado...")
            
            # 1. Iniciar procesador CUDA
            if self.cuda_processor:
                self.cuda_processor.start_processing()
            
            # 2. Iniciar video reader
            if self.video_reader:
                self.video_reader.start()
            
            self.running = True
            self.start_time = time.time()
            
            logger.info("Sistema integrado iniciado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando sistema: {e}")
            self.error_occurred.emit(f"Error de inicio: {e}")
            return False
    
    def stop_system(self):
        """Detiene el sistema completo"""
        if not self.running:
            return
        
        logger.info("Deteniendo sistema integrado...")
        
        self.running = False
        
        try:
            # Detener timer de estadísticas
            self.stats_timer.stop()
            
            # Detener performance tracker
            self.performance_tracker.stop_monitoring()
            
            # Detener video reader
            if self.video_reader:
                self.video_reader.stop()
                self.video_reader.deleteLater()
                self.video_reader = None
            
            # Detener procesador CUDA
            if self.cuda_processor:
                cuda_pipeline_manager.remove_processor(self.system_id)
                self.cuda_processor = None
            
            # Limpiar tracker
            self.tracker = None
            
            # Limpiar datos
            with self.sync_lock:
                self.frame_sync_queue.clear()
            
            logger.info("Sistema integrado detenido")
            
        except Exception as e:
            logger.error(f"Error deteniendo sistema: {e}")
    
    def change_mode(self, new_mode: VideoSystemMode):
        """Cambia el modo de operación del sistema"""
        try:
            if new_mode == self.current_mode:
                return True
            
            logger.info(f"Cambiando modo: {self.current_mode.value} -> {new_mode.value}")
            
            old_mode = self.current_mode
            self.current_mode = new_mode
            
            # Aplicar nueva configuración
            success = self._apply_operation_mode(new_mode)
            
            if success:
                self.mode_changed.emit(new_mode.value)
                logger.info(f"Modo cambiado exitosamente a {new_mode.value}")
            else:
                # Revertir si falló
                self.current_mode = old_mode
                logger.error(f"Error cambiando modo, revirtiendo a {old_mode.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error cambiando modo: {e}")
            return False
    
    def _apply_operation_mode(self, mode: VideoSystemMode):
        """Aplica configuración específica del modo de operación"""
        try:
            mode_configs = {
                VideoSystemMode.ULTRA_LOW_LATENCY: {
                    'video_profile': 'ultra_low_latency',
                    'cuda_batch_size': 1,
                    'cuda_precision': True,  # FP16
                    'buffer_size': 1,
                    'tracking_enabled': False,  # Disable tracking for min latency
                },
                VideoSystemMode.BALANCED: {
                    'video_profile': 'balanced',
                    'cuda_batch_size': 4,
                    'cuda_precision': True,
                    'buffer_size': 2,
                    'tracking_enabled': True,
                },
                VideoSystemMode.HIGH_QUALITY: {
                    'video_profile': 'quality',
                    'cuda_batch_size': 6,
                    'cuda_precision': False,  # FP32
                    'buffer_size': 3,
                    'tracking_enabled': True,
                },
                VideoSystemMode.POWER_SAVE: {
                    'video_profile': 'balanced',
                    'cuda_batch_size': 2,
                    'cuda_precision': True,
                    'buffer_size': 1,
                    'tracking_enabled': False,
                },
                VideoSystemMode.STRESS_TEST: {
                    'video_profile': 'ultra_low_latency',
                    'cuda_batch_size': 8,
                    'cuda_precision': True,
                    'buffer_size': 4,
                    'tracking_enabled': True,
                }
            }
            
            config = mode_configs.get(mode, mode_configs[VideoSystemMode.BALANCED])
            
            # Aplicar configuración a componentes
            if self.cuda_processor:
                self.cuda_processor.update_model_config({
                    'batch_size': config['cuda_batch_size'],
                    'half_precision': config['cuda_precision'],
                })
            
            # Actualizar configuración local
            self.config['tracking']['enable_tracking'] = config['tracking_enabled']
            
            logger.info(f"Configuración de modo {mode.value} aplicada")
            return True
            
        except Exception as e:
            logger.error(f"Error aplicando modo {mode.value}: {e}")
            return False
    
    def _get_video_profile_for_mode(self, mode: VideoSystemMode):
        """Obtiene el perfil de video para un modo específico"""
        profiles = {
            VideoSystemMode.ULTRA_LOW_LATENCY: 'ultra_low_latency',
            VideoSystemMode.BALANCED: 'balanced', 
            VideoSystemMode.HIGH_QUALITY: 'quality',
            VideoSystemMode.POWER_SAVE: 'balanced',
            VideoSystemMode.STRESS_TEST: 'ultra_low_latency',
        }
        return profiles.get(mode, 'balanced')
    
    def _get_model_path_for_mode(self, mode: VideoSystemMode):
        """Obtiene la ruta del modelo para un modo específico"""
        models = {
            VideoSystemMode.ULTRA_LOW_LATENCY: 'yolov8n.pt',  # Modelo más rápido
            VideoSystemMode.BALANCED: 'yolov8s.pt',           # Balance
            VideoSystemMode.HIGH_QUALITY: 'yolov8m.pt',       # Mejor precisión
            VideoSystemMode.POWER_SAVE: 'yolov8n.pt',         # Eficiencia energética
            VideoSystemMode.STRESS_TEST: 'yolov8s.pt',        # Modelo intermedio
        }
        return models.get(mode, 'yolov8s.pt')
    
    # Callbacks de componentes
    def _on_video_frame(self, frame: np.ndarray):
        """Callback para frames de video del GStreamer"""
        try:
            if not self.running or not self.cuda_processor:
                return
            
            # Agregar timestamp para medir latencia
            frame_metadata = {
                'video_timestamp': time.time(),
                'system_id': self.system_id,
                'frame_source': 'gstreamer'
            }
            
            # Enviar frame al procesador CUDA
            success = self.cuda_processor.add_frame(frame, frame_metadata)
            
            if not success:
                logger.debug("Frame descartado por procesador CUDA")
            
            # Actualizar métricas de performance
            self.performance_tracker.record_video_frame()
            
        except Exception as e:
            logger.error(f"Error procesando frame de video: {e}")
    
    def _on_display_frame(self, pixmap: QPixmap):
        """Callback para frames de display"""
        try:
            # Emitir directamente para UI
            self.display_frame.emit(pixmap)
            
        except Exception as e:
            logger.error(f"Error procesando frame de display: {e}")
    
    def _on_cuda_results(self, detections: List, metadata: Dict):
        """Callback para resultados del procesador CUDA"""
        try:
            if not self.running:
                return
            
            # Calcular latencia end-to-end
            if 'video_timestamp' in metadata:
                end_to_end_latency = (time.time() - metadata['video_timestamp']) * 1000
                metadata['end_to_end_latency_ms'] = end_to_end_latency
                self.integrated_stats['total_latency_ms'] = end_to_end_latency
            
            # Aplicar tracking si está habilitado
            if self.tracker and self.config['tracking']['enable_tracking']:
                # Convertir detecciones al formato del tracker
                tracker_detections = self._convert_detections_for_tracker(detections)
                
                # Obtener frame original si está disponible (para tracking)
                frame = None  # En implementación completa, obtener del sync_queue
                
                # Aplicar tracking
                tracked_results = self.tracker.update(tracker_detections, frame=frame)
                
                # Actualizar estadísticas de tracking
                self.integrated_stats['active_tracks'] = len(tracked_results)
                self.performance_tracker.record_tracking_result(tracked_results)
                
                # Emitir resultados con tracking
                self.detection_results.emit(tracked_results, metadata)
            else:
                # Emitir detecciones sin tracking
                formatted_detections = self._format_detections_without_tracking(detections)
                self.detection_results.emit(formatted_detections, metadata)
            
            # Actualizar métricas de performance
            self.performance_tracker.record_detection_result(detections)
            
        except Exception as e:
            logger.error(f"Error procesando resultados CUDA: {e}")
    
    def _convert_detections_for_tracker(self, detections: List) -> List:
        """Convierte detecciones CUDA al formato del tracker"""
        tracker_detections = []
        
        for detection in detections:
            tracker_detection = {
                'bbox': detection['bbox'],
                'cls': detection['cls'],
                'conf': detection['conf']
            }
            tracker_detections.append(tracker_detection)
        
        return tracker_detections
    
    def _format_detections_without_tracking(self, detections: List) -> List:
        """Formatea detecciones cuando no hay tracking"""
        formatted = []
        
        for i, detection in enumerate(detections):
            formatted_detection = {
                'bbox': detection['bbox'],
                'id': f'det_{i}',  # ID temporal
                'cls': detection['cls'],
                'conf': detection['conf'],
                'centers': [],  # Sin historial
                'moving': None,  # Sin información de movimiento
            }
            formatted.append(formatted_detection)
        
        return formatted
    
    def _on_video_stats(self, stats: Dict):
        """Callback para estadísticas de video"""
        try:
            self.integrated_stats.update({
                'video_capture_fps': stats.get('fps_received', 0),
                'video_connection_stable': stats.get('connection_stable', False),
                'video_latency_ms': stats.get('latency_ms', 0),
            })
            
        except Exception as e:
            logger.error(f"Error procesando estadísticas de video: {e}")
    
    def _on_cuda_stats(self, stats: Dict):
        """Callback para estadísticas CUDA"""
        try:
            self.integrated_stats.update({
                'cuda_processing_fps': stats.get('fps', 0),
                'cuda_memory_usage': stats.get('memory_usage', 0),
                'cuda_inference_time': stats.get('inference_time', 0),
                'cuda_batch_efficiency': stats.get('batch_efficiency', 0),
            })
            
        except Exception as e:
            logger.error(f"Error procesando estadísticas CUDA: {e}")
    
    def _on_video_error(self, error_msg: str):
        """Callback para errores de video"""
        logger.error(f"Error de video: {error_msg}")
        self.error_occurred.emit(f"Video: {error_msg}")
        
        # Auto-reconexión si está habilitada
        if self.config['system']['auto_reconnect']:
            self._attempt_auto_reconnect()
    
    def _on_cuda_error(self, error_msg: str):
        """Callback para errores CUDA"""
        logger.error(f"Error CUDA: {error_msg}")
        self.error_occurred.emit(f"CUDA: {error_msg}")
    
    def _attempt_auto_reconnect(self):
        """Intenta reconectar automáticamente"""
        try:
            if not self.config['system']['auto_reconnect']:
                return
            
            logger.info("Intentando reconexión automática...")
            
            # Detener componentes actuales
            if self.video_reader:
                self.video_reader.stop()
            
            # Esperar un momento
            time.sleep(2.0)
            
            # Reinicializar video reader
            if self.rtsp_url:
                self._initialize_video_reader(self.rtsp_url, "fija")  # Asumir fija por defecto
                if self.video_reader:
                    self.video_reader.start()
                    logger.info("Reconexión automática exitosa")
            
        except Exception as e:
            logger.error(f"Error en reconexión automática: {e}")
    
    def _update_integrated_stats(self):
        """Actualiza estadísticas integradas del sistema"""
        try:
            # Actualizar tiempo de actividad
            self.integrated_stats['uptime_seconds'] = time.time() - self.start_time
            
            # Obtener estadísticas del performance tracker
            perf_stats = self.performance_tracker.get_stats()
            self.integrated_stats.update(perf_stats)
            
            # Calcular FPS end-to-end
            video_fps = self.integrated_stats.get('video_capture_fps', 0)
            cuda_fps = self.integrated_stats.get('cuda_processing_fps', 0)
            self.integrated_stats['end_to_end_fps'] = min(video_fps, cuda_fps)
            
            # Evaluar salud del sistema
            self.integrated_stats['system_health'] = self._evaluate_system_health()
            
            # Emitir estadísticas
            self.system_stats.emit(self.integrated_stats.copy())
            
        except Exception as e:
            logger.error(f"Error actualizando estadísticas integradas: {e}")
    
    def _evaluate_system_health(self) -> str:
        """Evalúa la salud general del sistema"""
        try:
            video_fps = self.integrated_stats.get('video_capture_fps', 0)
            cuda_fps = self.integrated_stats.get('cuda_processing_fps', 0)
            latency = self.integrated_stats.get('total_latency_ms', 0)
            memory_usage = self.integrated_stats.get('cuda_memory_usage', 0)
            
            # Criterios de salud basados en el modo actual
            if self.current_mode == VideoSystemMode.ULTRA_LOW_LATENCY:
                target_fps = 15
                max_latency = 50
            elif self.current_mode == VideoSystemMode.HIGH_QUALITY:
                target_fps = 8
                max_latency = 200
            else:  # BALANCED y otros
                target_fps = 10
                max_latency = 100
            
            # Evaluar métricas
            issues = []
            
            if video_fps < target_fps * 0.7:
                issues.append('low_video_fps')
            
            if cuda_fps < target_fps * 0.7:
                issues.append('low_cuda_fps')
            
            if latency > max_latency * 1.5:
                issues.append('high_latency')
            
            if memory_usage > 3.5:  # > 3.5GB para RTX 3050
                issues.append('high_memory')
            
            # Determinar estado de salud
            if not issues:
                return 'excellent'
            elif len(issues) == 1:
                return 'good'
            elif len(issues) == 2:
                return 'warning'
            else:
                return 'critical'
                
        except Exception as e:
            logger.error(f"Error evaluando salud del sistema: {e}")
            return 'unknown'
    
    # Métodos públicos de control
    def pause_system(self):
        """Pausa el sistema temporalmente"""
        try:
            if self.video_reader:
                self.video_reader.pause()
            
            if self.cuda_processor:
                # CUDA processor no tiene pause, pero podemos detener temporalmente
                pass
            
            logger.info("Sistema pausado")
            
        except Exception as e:
            logger.error(f"Error pausando sistema: {e}")
    
    def resume_system(self):
        """Reanuda el sistema"""
        try:
            if self.video_reader:
                self.video_reader.resume()
            
            logger.info("Sistema reanudado")
            
        except Exception as e:
            logger.error(f"Error reanudando sistema: {e}")
    
    def get_current_stats(self) -> Dict:
        """Obtiene estadísticas actuales del sistema"""
        return self.integrated_stats.copy()
    
    def get_system_info(self) -> Dict:
        """Obtiene información completa del sistema"""
        info = {
            'system_id': self.system_id,
            'current_mode': self.current_mode.value,
            'running': self.running,
            'rtsp_url': self.rtsp_url,
            'components': {
                'video_reader': self.video_reader is not None,
                'cuda_processor': self.cuda_processor is not None,
                'tracker': self.tracker is not None,
            },
            'config': self.config.copy(),
            'stats': self.integrated_stats.copy(),
        }
        
        if self.cuda_processor:
            info['cuda_info'] = self.cuda_processor.get_model_info()
        
        return info
    
    def update_config(self, new_config: Dict):
        """Actualiza configuración del sistema"""
        try:
            self.config.update(new_config)
            
            # Aplicar cambios relevantes
            if 'cuda' in new_config and self.cuda_processor:
                self.cuda_processor.update_model_config(new_config['cuda'])
            
            logger.info("Configuración del sistema actualizada")
            
        except Exception as e:
            logger.error(f"Error actualizando configuración: {e}")


class PerformanceTracker:
    """Tracker de rendimiento para el sistema integrado"""
    
    def __init__(self):
        self.video_frame_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        self.tracking_times = deque(maxlen=100)
        
        self.total_video_frames = 0
        self.total_detections = 0
        self.total_tracks = 0
        
        self.start_time = time.time()
        self.monitoring = False
    
    def start_monitoring(self):
        """Inicia el monitoreo de rendimiento"""
        self.monitoring = True
        self.start_time = time.time()
    
    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.monitoring = False
    
    def record_video_frame(self):
        """Registra un frame de video procesado"""
        if self.monitoring:
            self.video_frame_times.append(time.time())
            self.total_video_frames += 1
    
    def record_detection_result(self, detections):
        """Registra resultado de detección"""
        if self.monitoring:
            self.detection_times.append(time.time())
            self.total_detections += len(detections)
    
    def record_tracking_result(self, tracks):
        """Registra resultado de tracking"""
        if self.monitoring:
            self.tracking_times.append(time.time())
            self.total_tracks += len(tracks)
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas de rendimiento"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calcular FPS para los últimos 10 segundos
        recent_time = current_time - 10.0
        
        recent_video_frames = sum(1 for t in self.video_frame_times if t > recent_time)
        recent_detections = sum(1 for t in self.detection_times if t > recent_time)
        recent_tracking = sum(1 for t in self.tracking_times if t > recent_time)
        
        return {
            'performance_video_fps': recent_video_frames / 10.0,
            'performance_detection_fps': recent_detections / 10.0,
            'performance_tracking_fps': recent_tracking / 10.0,
            'total_frames_processed': self.total_video_frames,
            'total_detections_made': self.total_detections,
            'total_tracks_created': self.total_tracks,
            'uptime_seconds': uptime,
        }


class IntegratedVideoSystemManager:
    """Manager para múltiples sistemas de video integrados"""
    
    def __init__(self):
        self.systems = {}
        self.global_stats = {
            'total_systems': 0,
            'active_systems': 0,
            'total_fps': 0.0,
            'total_memory_usage': 0.0,
            'average_latency': 0.0,
            'system_health_distribution': {},
        }
    
    def create_system(self, system_id: str, config: Dict = None) -> IntegratedVideoSystem:
        """Crea un nuevo sistema de video integrado"""
        try:
            if system_id in self.systems:
                logger.warning(f"Sistema {system_id} ya existe. Reemplazando...")
                self.remove_system(system_id)
            
            system = IntegratedVideoSystem(system_id, config)
            self.systems[system_id] = system
            
            # Conectar señales para estadísticas globales
            system.system_stats.connect(
                lambda stats, sid=system_id: self._update_global_stats(sid, stats)
            )
            
            logger.info(f"Sistema integrado {system_id} creado")
            return system
            
        except Exception as e:
            logger.error(f"Error creando sistema {system_id}: {e}")
            return None
    
    def get_system(self, system_id: str) -> Optional[IntegratedVideoSystem]:
        """Obtiene un sistema por ID"""
        return self.systems.get(system_id)
    
    def remove_system(self, system_id: str):
        """Remueve un sistema"""
        try:
            if system_id in self.systems:
                system = self.systems[system_id]
                system.stop_system()
                del self.systems[system_id]
                
                logger.info(f"Sistema {system_id} removido")
                
        except Exception as e:
            logger.error(f"Error removiendo sistema {system_id}: {e}")
    
    def stop_all_systems(self):
        """Detiene todos los sistemas"""
        logger.info("Deteniendo todos los sistemas integrados...")
        
        for system_id in list(self.systems.keys()):
            self.remove_system(system_id)
        
        # Limpiar procesadores CUDA globalmente
        cuda_pipeline_manager.stop_all_processors()
        
        logger.info("Todos los sistemas detenidos")
    
    def get_global_stats(self) -> Dict:
        """Obtiene estadísticas globales"""
        return self.global_stats.copy()
    
    def _update_global_stats(self, system_id: str, stats: Dict):
        """Actualiza estadísticas globales"""
        try:
            self.global_stats['total_systems'] = len(self.systems)
            self.global_stats['active_systems'] = sum(
                1 for s in self.systems.values() if s.running
            )
            
            # Agregar estadísticas de todos los sistemas activos
            total_fps = 0.0
            total_memory = 0.0
            total_latency = 0.0
            health_distribution = {}
            active_count = 0
            
            for system in self.systems.values():
                if system.running:
                    sys_stats = system.get_current_stats()
                    total_fps += sys_stats.get('end_to_end_fps', 0)
                    total_memory += sys_stats.get('cuda_memory_usage', 0)
                    total_latency += sys_stats.get('total_latency_ms', 0)
                    
                    health = sys_stats.get('system_health', 'unknown')
                    health_distribution[health] = health_distribution.get(health, 0) + 1
                    active_count += 1
            
            self.global_stats.update({
                'total_fps': total_fps,
                'total_memory_usage': total_memory,
                'average_latency': total_latency / max(1, active_count),
                'system_health_distribution': health_distribution,
            })
            
        except Exception as e:
            logger.error(f"Error actualizando estadísticas globales: {e}")


# Instancia global del manager
integrated_video_manager = IntegratedVideoSystemManager()


# Ejemplo de uso para testing
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Crear sistema de prueba
    config = {
        'video': {
            'performance_profile': 'balanced',
            'enable_gpu_decode': True,
        },
        'cuda': {
            'batch_size': 4,
            'confidence_threshold': 0.5,
        },
        'tracking': {
            'enable_tracking': True,
        }
    }
    
    system = integrated_video_manager.create_system('test_system', config)
    
    if system:
        # Conectar señales para testing
        system.detection_results.connect(
            lambda dets, meta: print(f"Detecciones: {len(dets)}")
        )
        system.system_stats.connect(
            lambda stats: print(f"FPS: {stats.get('end_to_end_fps', 0):.1f}")
        )
        system.error_occurred.connect(
            lambda err: print(f"Error: {err}")
        )
        
        # Inicializar y ejecutar (descomenta para test real)
        # rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream"
        # if system.initialize_system(rtsp_url, "fija"):
        #     system.start_system()
        #     
        #     # Ejecutar por 30 segundos
        #     QTimer.singleShot(30000, lambda: [
        #         system.stop_system(),
        #         integrated_video_manager.stop_all_systems(),
        #         app.quit()
        #     ])
        
        print("Sistema de video integrado creado exitosamente")
        print("Descomenta el código de test para ejecutar con RTSP real")
    
    else:
        print("Error creando sistema de video integrado")
    
    # sys.exit(app.exec())