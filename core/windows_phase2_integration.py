"""
Integración completa de Fase 2 para Windows con CUDA Pipeline optimizado.
Combina Windows Native Video Reader + CUDA Pipeline sin dependencias de GStreamer.
"""

import numpy as np
import time
import threading
from enum import Enum
from typing import Optional, Dict, List
from collections import deque
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap

from core.windows_native_video_reader import WindowsNativeVideoReader, WindowsVideoReaderFactory
from core.cuda_pipeline_processor import CUDAPipelineProcessor, cuda_pipeline_manager
from core.advanced_tracker import AdvancedTracker
from logging_utils import get_logger

logger = get_logger(__name__)

class WindowsVideoSystemMode(Enum):
    """Modos de operación optimizados para Windows"""
    BALANCED = "balanced"                    # Balance óptimo para RTX 3050
    ULTRA_LOW_LATENCY = "ultra_low_latency"  # Latencia mínima Windows
    HIGH_QUALITY = "high_quality"            # Máxima calidad
    POWER_EFFICIENT = "power_efficient"     # Eficiencia energética
    DEBUG_MODE = "debug_mode"                # Modo debug con logs detallados

class WindowsIntegratedVideoSystem(QObject):
    """
    Sistema de video integrado optimizado para Windows que combina:
    - Windows Native Video Reader (OpenCV optimizado)
    - CUDA Pipeline Processor (RTX 3050 optimizado)
    - Advanced Tracker (seguimiento inteligente)
    - Gestión automática de recursos Windows
    """
    
    # Señales principales
    detection_results = pyqtSignal(list, dict)    # detecciones, metadata
    display_frame = pyqtSignal(QPixmap)           # frame para UI
    system_stats = pyqtSignal(dict)               # estadísticas del sistema
    error_occurred = pyqtSignal(str)              # errores críticos
    mode_changed = pyqtSignal(str)                # cambio de modo
    connection_status = pyqtSignal(str)           # estado de conexión
    
    def __init__(self, system_id: str, config: Dict = None):
        super().__init__()
        
        self.system_id = system_id
        self.config = config or self._get_default_config()
        
        # Componentes del sistema
        self.video_reader: Optional[WindowsNativeVideoReader] = None
        self.cuda_processor: Optional[CUDAPipelineProcessor] = None
        self.tracker: Optional[AdvancedTracker] = None
        
        # Estado del sistema
        self.current_mode = WindowsVideoSystemMode.BALANCED
        self.running = False
        self.rtsp_url = None
        
        # Performance monitoring
        self.performance_monitor = WindowsPerformanceMonitor()
        
        # Estadísticas integradas
        self.integrated_stats = {
            'system_mode': self.current_mode.value,
            'platform': 'windows',
            'total_latency_ms': 0.0,
            'video_fps': 0.0,
            'cuda_fps': 0.0,
            'tracking_fps': 0.0,
            'end_to_end_fps': 0.0,
            'cpu_usage': 0.0,
            'gpu_memory_usage': 0.0,
            'gpu_utilization': 0.0,
            'active_tracks': 0,
            'system_health': 'unknown',
            'uptime_seconds': 0.0,
            'connection_stable': False,
            'windows_backend': 'unknown',
            'decode_method': 'software'
        }
        
        # Timer para estadísticas
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self._update_integrated_stats)
        
        # Tiempo de inicio
        self.start_time = time.time()
        
        logger.info(f"WindowsIntegratedVideoSystem '{system_id}' inicializado")
    
    def _get_default_config(self):
        """Configuración por defecto optimizada para Windows + RTX 3050"""
        return {
            # Configuración de video Windows
            'video': {
                'performance_profile': 'balanced',
                'use_gpu_decode': True,
                'preferred_backend': 'auto',  # Auto-detect best backend
                'buffer_size': 3,
                'enable_threading': True,
                'windows_optimizations': True,
            },
            
            # Configuración CUDA (optimizada para RTX 3050)
            'cuda': {
                'batch_size': 4,              # Óptimo para 4GB VRAM
                'confidence_threshold': 0.5,
                'input_size': (640, 640),
                'half_precision': True,       # FP16 para RTX 3050
                'enable_adaptive_batch': True,
                'model_path': 'yolov8s.pt',   # Balance velocidad/precisión
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
            
            # Configuración del sistema Windows
            'system': {
                'auto_mode_switching': True,
                'health_monitoring': True,
                'stats_update_interval': 2000,  # ms
                'auto_reconnect': True,
                'max_reconnect_attempts': 10,
                'windows_performance_mode': True,
                'enable_gpu_monitoring': True,
            }
        }
    
    def initialize_system(self, rtsp_url: str, camera_type: str = "fija"):
        """Inicializa todos los componentes del sistema Windows"""
        try:
            logger.info(f"Inicializando sistema Windows para {rtsp_url}")
            
            self.rtsp_url = rtsp_url
            
            # 1. Verificar requisitos Windows
            if not self._check_windows_requirements():
                raise RuntimeError("Requisitos de Windows no cumplidos")
            
            # 2. Inicializar video reader Windows
            self._initialize_windows_video_reader(rtsp_url, camera_type)
            
            # 3. Inicializar procesador CUDA
            self._initialize_cuda_processor()
            
            # 4. Inicializar tracker
            self._initialize_tracker()
            
            # 5. Configurar monitoreo de rendimiento
            self._setup_performance_monitoring()
            
            # 6. Aplicar modo de operación
            self._apply_operation_mode(self.current_mode)
            
            logger.info("Sistema Windows integrado inicializado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error inicializando sistema Windows: {e}")
            self.error_occurred.emit(f"Error de inicialización: {e}")
            return False
    
    def _check_windows_requirements(self):
        """Verifica requisitos específicos de Windows"""
        try:
            from core.windows_native_video_reader import check_windows_video_capabilities
            from core.cuda_pipeline_processor import verify_cuda_pipeline_requirements
            
            # Verificar capacidades de video Windows
            video_caps, video_issues = check_windows_video_capabilities()
            if video_issues:
                logger.error(f"Problemas con video Windows: {video_issues}")
                return False
            
            # Verificar CUDA
            cuda_reqs, cuda_issues = verify_cuda_pipeline_requirements()
            if cuda_issues:
                logger.error(f"Problemas con CUDA: {cuda_issues}")
                return False
            
            # Actualizar stats con info del sistema
            self.integrated_stats.update({
                'windows_backend': ', '.join(video_caps.get('backends_available', [])),
                'decode_method': 'hardware' if video_caps.get('hardware_decode_support') else 'software',
                'gpu_info': cuda_reqs.get('gpu_info', 'unknown')
            })
            
            logger.info("Requisitos Windows verificados exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error verificando requisitos Windows: {e}")
            return False
    
    def _initialize_windows_video_reader(self, rtsp_url: str, camera_type: str):
        """Inicializa el video reader nativo de Windows"""
        try:
            # Crear video reader optimizado para Windows
            performance_profile = self._get_video_profile_for_mode(self.current_mode)
            
            self.video_reader = WindowsVideoReaderFactory.create_reader(
                rtsp_url=rtsp_url,
                camera_type=camera_type,
                performance_profile=performance_profile
            )
            
            # Conectar señales
            self.video_reader.frame_ready.connect(self._on_video_frame)
            self.video_reader.display_ready.connect(self._on_display_frame)
            self.video_reader.stats_updated.connect(self._on_video_stats)
            self.video_reader.error_occurred.connect(self._on_video_error)
            self.video_reader.connection_state_changed.connect(self._on_connection_state)
            
            logger.info("Windows video reader inicializado")
            
        except Exception as e:
            logger.error(f"Error inicializando Windows video reader: {e}")
            raise
    
    def _initialize_cuda_processor(self):
        """Inicializa el procesador CUDA"""
        try:
            # Configuración del modelo optimizada para Windows + RTX 3050
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
            
            logger.info("Procesador CUDA inicializado para Windows")
            
        except Exception as e:
            logger.error(f"Error inicializando procesador CUDA: {e}")
            raise
    
    def _initialize_tracker(self):
        """Inicializa el tracker avanzado"""
        try:
            if not self.config['tracking']['enable_tracking']:
                logger.info("Tracking deshabilitado por configuración")
                return
            
            # Configuración del tracker optimizada para Windows
            tracker_config = {
                **self.config['tracking'],
                'conf_threshold': self.config['cuda']['confidence_threshold'],
                'device': 'cuda' if self.cuda_processor else 'cpu',
            }
            
            self.tracker = AdvancedTracker(**tracker_config)
            
            logger.info("Tracker avanzado inicializado para Windows")
            
        except Exception as e:
            logger.error(f"Error inicializando tracker: {e}")
            # Tracker es opcional
            self.tracker = None
    
    def _setup_performance_monitoring(self):
        """Configura el monitoreo de rendimiento Windows"""
        try:
            # Iniciar monitoreo específico de Windows
            self.performance_monitor.start_monitoring()
            
            # Timer para estadísticas integradas
            interval = self.config['system']['stats_update_interval']
            self.stats_timer.start(interval)
            
            logger.info("Monitoreo de rendimiento Windows configurado")
            
        except Exception as e:
            logger.error(f"Error configurando monitoreo: {e}")
    
    def start_system(self):
        """Inicia el sistema completo"""
        if self.running:
            logger.warning("Sistema ya está ejecutándose")
            return True
        
        try:
            logger.info("Iniciando sistema Windows integrado...")
            
            # 1. Iniciar procesador CUDA
            if self.cuda_processor:
                self.cuda_processor.start_processing()
            
            # 2. Iniciar video reader Windows
            if self.video_reader:
                self.video_reader.start()
            
            self.running = True
            self.start_time = time.time()
            
            logger.info("Sistema Windows integrado iniciado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"Error iniciando sistema Windows: {e}")
            self.error_occurred.emit(f"Error de inicio: {e}")
            return False
    
    def stop_system(self):
        """Detiene el sistema completo"""
        if not self.running:
            return
        
        logger.info("Deteniendo sistema Windows integrado...")
        
        self.running = False
        
        try:
            # Detener timers
            self.stats_timer.stop()
            
            # Detener monitoreo
            self.performance_monitor.stop_monitoring()
            
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
            
            logger.info("Sistema Windows integrado detenido")
            
        except Exception as e:
            logger.error(f"Error deteniendo sistema Windows: {e}")
    
    def change_mode(self, new_mode: WindowsVideoSystemMode):
        """Cambia el modo de operación del sistema"""
        try:
            if new_mode == self.current_mode:
                return True
            
            logger.info(f"Cambiando modo Windows: {self.current_mode.value} -> {new_mode.value}")
            
            old_mode = self.current_mode
            self.current_mode = new_mode
            
            # Aplicar nueva configuración
            success = self._apply_operation_mode(new_mode)
            
            if success:
                self.mode_changed.emit(new_mode.value)
                logger.info(f"Modo Windows cambiado exitosamente a {new_mode.value}")
            else:
                # Revertir si falló
                self.current_mode = old_mode
                logger.error(f"Error cambiando modo, revirtiendo a {old_mode.value}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error cambiando modo Windows: {e}")
            return False
    
    def _apply_operation_mode(self, mode: WindowsVideoSystemMode):
        """Aplica configuración específica del modo de operación Windows"""
        try:
            mode_configs = {
                WindowsVideoSystemMode.ULTRA_LOW_LATENCY: {
                    'video_profile': 'ultra_low_latency',
                    'cuda_batch_size': 1,
                    'cuda_precision': True,  # FP16
                    'tracking_enabled': False,  # Disable para min latencia
                    'windows_performance_mode': True,
                },
                WindowsVideoSystemMode.BALANCED: {
                    'video_profile': 'balanced',
                    'cuda_batch_size': 4,
                    'cuda_precision': True,
                    'tracking_enabled': True,
                    'windows_performance_mode': True,
                },
                WindowsVideoSystemMode.HIGH_QUALITY: {
                    'video_profile': 'quality',
                    'cuda_batch_size': 6,
                    'cuda_precision': False,  # FP32 para mejor precisión
                    'tracking_enabled': True,
                    'windows_performance_mode': True,
                },
                WindowsVideoSystemMode.POWER_EFFICIENT: {
                    'video_profile': 'power_save',
                    'cuda_batch_size': 2,
                    'cuda_precision': True,
                    'tracking_enabled': False,
                    'windows_performance_mode': False,
                },
                WindowsVideoSystemMode.DEBUG_MODE: {
                    'video_profile': 'balanced',
                    'cuda_batch_size': 2,
                    'cuda_precision': True,
                    'tracking_enabled': True,
                    'windows_performance_mode': False,
                }
            }
            
            config = mode_configs.get(mode, mode_configs[WindowsVideoSystemMode.BALANCED])
            
            # Aplicar configuración a componentes
            if self.cuda_processor:
                self.cuda_processor.update_model_config({
                    'batch_size': config['cuda_batch_size'],
                    'half_precision': config['cuda_precision'],
                })
            
            # Configurar modo de rendimiento Windows
            if config.get('windows_performance_mode'):
                self._set_windows_performance_mode(True)
            
            # Actualizar configuración local
            self.config['tracking']['enable_tracking'] = config['tracking_enabled']
            
            # Actualizar estadísticas
            self.integrated_stats['system_mode'] = mode.value
            
            logger.info(f"Configuración de modo Windows {mode.value} aplicada")
            return True
            
        except Exception as e:
            logger.error(f"Error aplicando modo Windows {mode.value}: {e}")
            return False
    
    def _set_windows_performance_mode(self, enable: bool):
        """Configura modo de rendimiento específico de Windows"""
        try:
            if enable:
                # Configuraciones específicas de Windows para máximo rendimiento
                import os
                
                # Configurar prioridad del proceso
                try:
                    import psutil
                    current_process = psutil.Process()
                    current_process.nice(psutil.HIGH_PRIORITY_CLASS)
                    logger.info("Prioridad de proceso Windows configurada a HIGH")
                except:
                    logger.warning("No se pudo configurar prioridad del proceso")
                
                # Configurar variables de entorno para OpenCV
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
                os.environ['OPENCV_VIDEOIO_PRIORITY_FFMPEG'] = '100'
                
            logger.info(f"Modo de rendimiento Windows: {'habilitado' if enable else 'deshabilitado'}")
            
        except Exception as e:
            logger.warning(f"Error configurando modo de rendimiento Windows: {e}")
    
    def _get_video_profile_for_mode(self, mode: WindowsVideoSystemMode):
        """Obtiene el perfil de video para un modo específico"""
        profiles = {
            WindowsVideoSystemMode.ULTRA_LOW_LATENCY: 'ultra_low_latency',
            WindowsVideoSystemMode.BALANCED: 'balanced',
            WindowsVideoSystemMode.HIGH_QUALITY: 'quality',
            WindowsVideoSystemMode.POWER_EFFICIENT: 'power_save',
            WindowsVideoSystemMode.DEBUG_MODE: 'balanced',
        }
        return profiles.get(mode, 'balanced')
    
    def _get_model_path_for_mode(self, mode: WindowsVideoSystemMode):
        """Obtiene la ruta del modelo para un modo específico"""
        models = {
            WindowsVideoSystemMode.ULTRA_LOW_LATENCY: 'yolov8n.pt',  # Más rápido
            WindowsVideoSystemMode.BALANCED: 'yolov8s.pt',           # Balance
            WindowsVideoSystemMode.HIGH_QUALITY: 'yolov8m.pt',       # Mejor precisión
            WindowsVideoSystemMode.POWER_EFFICIENT: 'yolov8n.pt',    # Eficiencia
            WindowsVideoSystemMode.DEBUG_MODE: 'yolov8n.pt',         # Rápido para debug
        }
        return models.get(mode, 'yolov8s.pt')
    
    # Callbacks de componentes
    def _on_video_frame(self, frame: np.ndarray):
        """Callback para frames de video del Windows reader"""
        try:
            if not self.running or not self.cuda_processor:
                return
            
            # Agregar metadata Windows específico
            frame_metadata = {
                'video_timestamp': time.time(),
                'system_id': self.system_id,
                'platform': 'windows',
                'frame_source': 'windows_native'
            }
            
            # Enviar frame al procesador CUDA
            success = self.cuda_processor.add_frame(frame, frame_metadata)
            
            if not success:
                logger.debug("Frame descartado por procesador CUDA")
            
            # Actualizar métricas
            self.performance_monitor.record_video_frame()
            
        except Exception as e:
            logger.error(f"Error procesando frame de video Windows: {e}")
    
    def _on_display_frame(self, pixmap: QPixmap):
        """Callback para frames de display"""
        try:
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
                # Convertir detecciones para tracker
                tracker_detections = [
                    {'bbox': det['bbox'], 'cls': det['cls'], 'conf': det['conf']}
                    for det in detections
                ]
                
                # Aplicar tracking
                tracked_results = self.tracker.update(tracker_detections, frame=None)
                
                # Actualizar estadísticas
                self.integrated_stats['active_tracks'] = len(tracked_results)
                self.performance_monitor.record_tracking_result(tracked_results)
                
                # Emitir resultados
                self.detection_results.emit(tracked_results, metadata)
            else:
                # Formatear detecciones sin tracking
                formatted_detections = [
                    {
                        'bbox': det['bbox'],
                        'id': f'det_{i}',
                        'cls': det['cls'],
                        'conf': det['conf'],
                        'centers': [],
                        'moving': None,
                    }
                    for i, det in enumerate(detections)
                ]
                self.detection_results.emit(formatted_detections, metadata)
            
            # Actualizar métricas
            self.performance_monitor.record_detection_result(detections)
            
        except Exception as e:
            logger.error(f"Error procesando resultados CUDA: {e}")
    
    def _on_video_stats(self, stats: Dict):
        """Callback para estadísticas de video Windows"""
        try:
            self.integrated_stats.update({
                'video_fps': stats.get('fps_capture', 0),
                'connection_stable': stats.get('connection_stable', False),
                'windows_backend': stats.get('backend_info', 'unknown'),
                'decode_method': stats.get('decode_method', 'software'),
            })
        except Exception as e:
            logger.error(f"Error procesando estadísticas de video: {e}")
    
    def _on_cuda_stats(self, stats: Dict):
        """Callback para estadísticas CUDA"""
        try:
            self.integrated_stats.update({
                'cuda_fps': stats.get('fps', 0),
                'gpu_memory_usage': stats.get('memory_usage', 0),
                'cuda_inference_time': stats.get('inference_time', 0),
            })
        except Exception as e:
            logger.error(f"Error procesando estadísticas CUDA: {e}")
    
    def _on_video_error(self, error_msg: str):
        """Callback para errores de video"""
        logger.error(f"Error de video Windows: {error_msg}")
        self.error_occurred.emit(f"Video: {error_msg}")
    
    def _on_cuda_error(self, error_msg: str):
        """Callback para errores CUDA"""
        logger.error(f"Error CUDA: {error_msg}")
        self.error_occurred.emit(f"CUDA: {error_msg}")
    
    def _on_connection_state(self, state: str):
        """Callback para cambios de estado de conexión"""
        logger.info(f"Estado de conexión Windows: {state}")
        self.connection_status.emit(state)
        self.integrated_stats['connection_stable'] = (state in ['connected', 'reconnected'])
    
    def _update_integrated_stats(self):
        """Actualiza estadísticas integradas del sistema Windows"""
        try:
            # Actualizar tiempo de actividad
            self.integrated_stats['uptime_seconds'] = time.time() - self.start_time
            
            # Obtener estadísticas del performance monitor
            perf_stats = self.performance_monitor.get_stats()
            self.integrated_stats.update(perf_stats)
            
            # Calcular FPS end-to-end
            video_fps = self.integrated_stats.get('video_fps', 0)
            cuda_fps = self.integrated_stats.get('cuda_fps', 0)
            self.integrated_stats['end_to_end_fps'] = min(video_fps, cuda_fps) if cuda_fps > 0 else 0
            
            # Evaluar salud del sistema
            self.integrated_stats['system_health'] = self._evaluate_system_health()
            
            # Emitir estadísticas
            self.system_stats.emit(self.integrated_stats.copy())
            
        except Exception as e:
            logger.error(f"Error actualizando estadísticas integradas: {e}")
    
    def _evaluate_system_health(self) -> str:
        """Evalúa la salud general del sistema Windows"""
        try:
            video_fps = self.integrated_stats.get('video_fps', 0)
            cuda_fps = self.integrated_stats.get('cuda_fps', 0)
            latency = self.integrated_stats.get('total_latency_ms', 0)
            memory_usage = self.integrated_stats.get('gpu_memory_usage', 0)
            connection_stable = self.integrated_stats.get('connection_stable', False)
            
            # Criterios de salud basados en el modo actual
            target_thresholds = {
                WindowsVideoSystemMode.ULTRA_LOW_LATENCY: {'fps': 15, 'latency': 50},
                WindowsVideoSystemMode.BALANCED: {'fps': 10, 'latency': 100},
                WindowsVideoSystemMode.HIGH_QUALITY: {'fps': 8, 'latency': 200},
                WindowsVideoSystemMode.POWER_EFFICIENT: {'fps': 5, 'latency': 150},
                WindowsVideoSystemMode.DEBUG_MODE: {'fps': 5, 'latency': 200},
            }
            
            thresholds = target_thresholds.get(
                self.current_mode, 
                target_thresholds[WindowsVideoSystemMode.BALANCED]
            )
            
            # Evaluar métricas
            issues = []
            
            if not connection_stable:
                issues.append('connection_unstable')
            
            if video_fps < thresholds['fps'] * 0.7:
                issues.append('low_video_fps')
            
            if cuda_fps < thresholds['fps'] * 0.7:
                issues.append('low_cuda_fps')
            
            if latency > thresholds['latency'] * 1.5:
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
            logger.error(f"Error evaluando salud del sistema Windows: {e}")
            return 'unknown'
    
    # Métodos públicos de control
    def pause_system(self):
        """Pausa el sistema temporalmente"""
        try:
            if self.video_reader:
                self.video_reader.pause()
            logger.info("Sistema Windows pausado")
        except Exception as e:
            logger.error(f"Error pausando sistema Windows: {e}")
    
    def resume_system(self):
        """Reanuda el sistema"""
        try:
            if self.video_reader:
                self.video_reader.resume()
            logger.info("Sistema Windows reanudado")
        except Exception as e:
            logger.error(f"Error reanudando sistema Windows: {e}")
    
    def get_current_stats(self) -> Dict:
        """Obtiene estadísticas actuales del sistema"""
        return self.integrated_stats.copy()
    
    def get_system_info(self) -> Dict:
        """Obtiene información completa del sistema Windows"""
        info = {
            'system_id': self.system_id,
            'platform': 'windows',
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
        
        if self.video_reader:
            info['video_info'] = self.video_reader.get_stats()
        
        return info


class WindowsPerformanceMonitor:
    """Monitor de rendimiento específico para Windows"""
    
    def __init__(self):
        self.monitoring = False
        self.start_time = time.time()
        
        # Contadores
        self.video_frames = 0
        self.detection_results = 0
        self.tracking_results = 0
        
        # Historial
        self.video_frame_times = deque(maxlen=100)
        self.detection_times = deque(maxlen=100)
        self.tracking_times = deque(maxlen=100)
        
        # Performance Windows específico
        self.cpu_monitor = None
        self.gpu_monitor = None
        
        try:
            import psutil
            self.cpu_monitor = psutil
        except ImportError:
            logger.warning("psutil no disponible, monitoreo de CPU limitado")
    
    def start_monitoring(self):
        """Inicia el monitoreo de rendimiento Windows"""
        self.monitoring = True
        self.start_time = time.time()
        logger.info("Monitoreo de rendimiento Windows iniciado")
    
    def stop_monitoring(self):
        """Detiene el monitoreo"""
        self.monitoring = False
        logger.info("Monitoreo de rendimiento Windows detenido")
    
    def record_video_frame(self):
        """Registra un frame de video procesado"""
        if self.monitoring:
            self.video_frame_times.append(time.time())
            self.video_frames += 1
    
    def record_detection_result(self, detections):
        """Registra resultado de detección"""
        if self.monitoring:
            self.detection_times.append(time.time())
            self.detection_results += len(detections)
    
    def record_tracking_result(self, tracks):
        """Registra resultado de tracking"""
        if self.monitoring:
            self.tracking_times.append(time.time())
            self.tracking_results += len(tracks)
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas de rendimiento Windows"""
        try:
            current_time = time.time()
            uptime = current_time - self.start_time
            
            # Calcular FPS para los últimos 10 segundos
            recent_time = current_time - 10.0
            
            recent_video = sum(1 for t in self.video_frame_times if t > recent_time)
            recent_detections = sum(1 for t in self.detection_times if t > recent_time)
            recent_tracking = sum(1 for t in self.tracking_times if t > recent_time)
            
            stats = {
                'performance_video_fps': recent_video / 10.0,
                'performance_detection_fps': recent_detections / 10.0,
                'performance_tracking_fps': recent_tracking / 10.0,
                'total_frames_processed': self.video_frames,
                'total_detections_made': self.detection_results,
                'total_tracks_created': self.tracking_results,
                'uptime_seconds': uptime,
            }
            
            # Estadísticas específicas de Windows
            if self.cpu_monitor:
                try:
                    stats['cpu_usage'] = self.cpu_monitor.cpu_percent(interval=None)
                    memory_info = self.cpu_monitor.virtual_memory()
                    stats['system_memory_usage'] = memory_info.percent
                except:
                    pass
            
            # GPU stats (si está disponible)
            try:
                import torch
                if torch.cuda.is_available():
                    stats['gpu_memory_allocated'] = torch.cuda.memory_allocated() / (1024**3)
                    stats['gpu_memory_reserved'] = torch.cuda.memory_reserved() / (1024**3)
            except:
                pass
            
            return stats
            
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas Windows: {e}")
            return {}


class WindowsIntegratedVideoSystemManager:
    """Manager para múltiples sistemas de video integrados Windows"""
    
    def __init__(self):
        self.systems = {}
        self.global_stats = {
            'platform': 'windows',
            'total_systems': 0,
            'active_systems': 0,
            'total_fps': 0.0,
            'total_memory_usage': 0.0,
            'average_latency': 0.0,
            'system_health_distribution': {},
            'windows_backend_distribution': {},
        }
    
    def create_system(self, system_id: str, config: Dict = None) -> WindowsIntegratedVideoSystem:
        """Crea un nuevo sistema de video integrado Windows"""
        try:
            if system_id in self.systems:
                logger.warning(f"Sistema Windows {system_id} ya existe. Reemplazando...")
                self.remove_system(system_id)
            
            system = WindowsIntegratedVideoSystem(system_id, config)
            self.systems[system_id] = system
            
            # Conectar señales para estadísticas globales
            system.system_stats.connect(
                lambda stats, sid=system_id: self._update_global_stats(sid, stats)
            )
            
            logger.info(f"Sistema Windows integrado {system_id} creado")
            return system
            
        except Exception as e:
            logger.error(f"Error creando sistema Windows {system_id}: {e}")
            return None
    
    def get_system(self, system_id: str) -> Optional[WindowsIntegratedVideoSystem]:
        """Obtiene un sistema por ID"""
        return self.systems.get(system_id)
    
    def remove_system(self, system_id: str):
        """Remueve un sistema"""
        try:
            if system_id in self.systems:
                system = self.systems[system_id]
                system.stop_system()
                del self.systems[system_id]
                logger.info(f"Sistema Windows {system_id} removido")
        except Exception as e:
            logger.error(f"Error removiendo sistema Windows {system_id}: {e}")
    
    def stop_all_systems(self):
        """Detiene todos los sistemas Windows"""
        logger.info("Deteniendo todos los sistemas Windows integrados...")
        
        for system_id in list(self.systems.keys()):
            self.remove_system(system_id)
        
        # Limpiar procesadores CUDA globalmente
        cuda_pipeline_manager.stop_all_processors()
        
        logger.info("Todos los sistemas Windows detenidos")
    
    def get_global_stats(self) -> Dict:
        """Obtiene estadísticas globales de Windows"""
        return self.global_stats.copy()
    
    def _update_global_stats(self, system_id: str, stats: Dict):
        """Actualiza estadísticas globales de Windows"""
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
            backend_distribution = {}
            active_count = 0
            
            for system in self.systems.values():
                if system.running:
                    sys_stats = system.get_current_stats()
                    total_fps += sys_stats.get('end_to_end_fps', 0)
                    total_memory += sys_stats.get('gpu_memory_usage', 0)
                    total_latency += sys_stats.get('total_latency_ms', 0)
                    
                    health = sys_stats.get('system_health', 'unknown')
                    health_distribution[health] = health_distribution.get(health, 0) + 1
                    
                    backend = sys_stats.get('windows_backend', 'unknown')
                    backend_distribution[backend] = backend_distribution.get(backend, 0) + 1
                    
                    active_count += 1
            
            self.global_stats.update({
                'total_fps': total_fps,
                'total_memory_usage': total_memory,
                'average_latency': total_latency / max(1, active_count),
                'system_health_distribution': health_distribution,
                'windows_backend_distribution': backend_distribution,
            })
            
        except Exception as e:
            logger.error(f"Error actualizando estadísticas globales Windows: {e}")


# Instancia global del manager Windows
windows_integrated_video_manager = WindowsIntegratedVideoSystemManager()


def create_windows_phase2_system(system_id: str, rtsp_url: str, camera_type: str = "fija", 
                                 mode: WindowsVideoSystemMode = WindowsVideoSystemMode.BALANCED):
    """
    Función de conveniencia para crear un sistema Fase 2 completo para Windows.
    
    Args:
        system_id: ID único del sistema
        rtsp_url: URL del stream RTSP
        camera_type: Tipo de cámara (fija, ptz, nvr)
        mode: Modo de operación inicial
        
    Returns:
        WindowsIntegratedVideoSystem configurado y listo para usar
    """
    try:
        # Configuración optimizada para Windows + RTX 3050
        config = {
            'video': {
                'performance_profile': mode.value,
                'use_gpu_decode': True,
                'enable_threading': True,
                'windows_optimizations': True,
            },
            'cuda': {
                'batch_size': 4 if mode == WindowsVideoSystemMode.BALANCED else 2,
                'confidence_threshold': 0.5,
                'half_precision': True,
                'enable_adaptive_batch': True,
            },
            'tracking': {
                'enable_tracking': mode != WindowsVideoSystemMode.ULTRA_LOW_LATENCY,
                'enable_size_control': True,
                'enable_velocity_prediction': True,
            },
            'system': {
                'auto_reconnect': True,
                'windows_performance_mode': True,
                'enable_gpu_monitoring': True,
            }
        }
        
        # Crear sistema
        system = windows_integrated_video_manager.create_system(system_id, config)
        
        if system:
            # Configurar modo inicial
            system.change_mode(mode)
            
            # Inicializar
            if system.initialize_system(rtsp_url, camera_type):
                logger.info(f"Sistema Windows Fase 2 '{system_id}' creado exitosamente")
                return system
            else:
                logger.error(f"Error inicializando sistema Windows '{system_id}'")
                windows_integrated_video_manager.remove_system(system_id)
                return None
        else:
            logger.error(f"Error creando sistema Windows '{system_id}'")
            return None
            
    except Exception as e:
        logger.error(f"Error en create_windows_phase2_system: {e}")
        return None


# Ejemplo de uso para testing
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Verificar capacidades Windows
    from core.windows_native_video_reader import check_windows_video_capabilities
    from core.cuda_pipeline_processor import verify_cuda_pipeline_requirements
    
    print("=== VERIFICACIÓN SISTEMA WINDOWS FASE 2 ===")
    
    # Video Windows
    video_caps, video_issues = check_windows_video_capabilities()
    print(f"OpenCV: {video_caps.get('opencv_version', 'N/A')}")
    print(f"Backends: {video_caps.get('backends_available', [])}")
    print(f"Hardware decode: {video_caps.get('hardware_decode_support', False)}")
    
    # CUDA
    cuda_reqs, cuda_issues = verify_cuda_pipeline_requirements()
    print(f"GPU: {cuda_reqs.get('gpu_info', 'N/A')}")
    print(f"Memoria GPU: {cuda_reqs.get('memory_available', 0):.1f} GB")
    
    if video_issues or cuda_issues:
        print("\n⚠️ PROBLEMAS DETECTADOS:")
        for issue in video_issues + cuda_issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Sistema Windows Fase 2 listo!")
        
        # Test básico (descomenta para test real)
        # print("\n🧪 Creando sistema de prueba...")
        # system = create_windows_phase2_system(
        #     system_id="test_windows",
        #     rtsp_url="rtsp://admin:password@192.168.1.100:554/stream",
        #     camera_type="fija",
        #     mode=WindowsVideoSystemMode.BALANCED
        # )
        # 
        # if system:
        #     print("✅ Sistema creado exitosamente")
        #     # Conectar señales para ver resultados
        #     system.detection_results.connect(
        #         lambda dets, meta: print(f"Detecciones: {len(dets)}")
        #     )
        #     system.system_stats.connect(
        #         lambda stats: print(f"FPS: {stats.get('end_to_end_fps', 0):.1f}")
        #     )
        #     
        #     # Iniciar sistema
        #     if system.start_system():
        #         print("✅ Sistema iniciado")
        #         
        #         # Ejecutar por 30 segundos
        #         QTimer.singleShot(30000, lambda: [
        #             system.stop_system(),
        #             windows_integrated_video_manager.stop_all_systems(),
        #             app.quit()
        #         ])
        #     else:
        #         print("❌ Error iniciando sistema")
        # else:
        #     print("❌ Error creando sistema")
    
    print("\nDescomenta el código de test para ejecutar con RTSP real")
    # sys.exit(app.exec())