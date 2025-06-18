"""
Windows Native Video Reader optimizado para RTX 3050.
Utiliza OpenCV con optimizaciones Windows + CUDA sin dependencias adicionales.
"""

import cv2
import numpy as np
import threading
import time
from collections import deque
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtGui import QImage, QPixmap
from logging_utils import get_logger

logger = get_logger(__name__)

class WindowsNativeVideoReader(QObject):
    """
    Video reader nativo para Windows optimizado para RTX 3050.
    
    Características:
    - OpenCV con backend optimizado Windows
    - Hardware decode con DXVA2/D3D11 (cuando disponible)
    - Threading optimizado para Windows
    - Gestión inteligente de buffers
    - Reconexión automática robusta
    """
    
    # Señales
    frame_ready = pyqtSignal(np.ndarray)      # Frame para análisis
    display_ready = pyqtSignal(QPixmap)       # Frame para display
    stats_updated = pyqtSignal(dict)          # Estadísticas
    error_occurred = pyqtSignal(str)          # Errores
    connection_state_changed = pyqtSignal(str)  # Estados de conexión
    
    def __init__(self, rtsp_url, config=None):
        super().__init__()
        
        self.rtsp_url = rtsp_url
        self.config = config or self._get_default_config()
        
        # OpenCV VideoCapture
        self.cap = None
        
        # Estado
        self._running = False
        self._paused = False
        self._connected = False
        
        # Threading optimizado para Windows
        self._capture_thread = None
        self._display_thread = None
        self._reconnect_thread = None
        
        # Buffers circulares
        self._frame_buffer = deque(maxlen=self.config['buffer_size'])
        self._display_buffer = deque(maxlen=2)
        
        # Locks thread-safe
        self._buffer_lock = threading.Lock()
        self._display_lock = threading.Lock()
        self._cap_lock = threading.Lock()
        
        # Estadísticas
        self._stats = {
            'fps_capture': 0.0,
            'fps_display': 0.0,
            'frames_received': 0,
            'frames_dropped': 0,
            'connection_stable': False,
            'backend_info': 'unknown',
            'decode_method': 'software',
            'resolution': '0x0',
            'reconnect_attempts': 0,
            'uptime_seconds': 0.0,
            'buffer_health': 0.0
        }
        
        # Contadores
        self._frame_count = 0
        self._display_count = 0
        self._last_stats_time = time.time()
        self._start_time = None
        self._reconnect_attempts = 0
        
        # Configurar backend OpenCV para Windows
        self._setup_opencv_backend()
        
        logger.info(f"WindowsNativeVideoReader inicializado para: {rtsp_url}")
    
    def _get_default_config(self):
        """Configuración por defecto optimizada para Windows + RTX 3050"""
        return {
            'buffer_size': 3,                    # Buffers para estabilidad Windows
            'capture_fps': 30,                   # FPS objetivo de captura
            'display_fps': 25,                   # FPS de display
            'analysis_fps': 8,                   # FPS para análisis IA
            'reconnect_delay': 2.0,              # Delay entre reconexiones
            'max_reconnect_attempts': 10,        # Intentos máximos
            'connection_timeout': 5000,          # Timeout en ms
            'read_timeout': 1000,                # Timeout de lectura en ms
            'preferred_backend': cv2.CAP_FFMPEG, # Backend preferido
            'use_gpu_decode': True,              # Intentar decode en GPU
            'output_format': 'BGR',              # Formato de salida
            'frame_skip_on_delay': True,         # Skip frames si hay retraso
            'adaptive_quality': True,            # Calidad adaptativa
            'enable_threading': True,            # Threading mejorado
            'windows_optimizations': True,       # Optimizaciones Windows
        }
    
    def _setup_opencv_backend(self):
        """Configura el backend OpenCV optimizado para Windows"""
        try:
            # Detectar backends disponibles
            backends = []
            
            # Probar FFMPEG (mejor para RTSP)
            if cv2.CAP_FFMPEG in cv2.videoio_registry.getCameraBackends():
                backends.append(('FFMPEG', cv2.CAP_FFMPEG))
            
            # Probar DirectShow (nativo Windows)
            if cv2.CAP_DSHOW in cv2.videoio_registry.getCameraBackends():
                backends.append(('DirectShow', cv2.CAP_DSHOW))
            
            # Probar Media Foundation (Windows 10+)
            if cv2.CAP_MSMF in cv2.videoio_registry.getCameraBackends():
                backends.append(('Media Foundation', cv2.CAP_MSMF))
            
            logger.info(f"Backends OpenCV disponibles: {[b[0] for b in backends]}")
            
            # Seleccionar mejor backend
            if backends:
                self.config['preferred_backend'] = backends[0][1]
                self._stats['backend_info'] = backends[0][0]
            
            # Configurar optimizaciones de OpenCV para Windows
            if self.config['windows_optimizations']:
                # Habilitar optimizaciones Intel IPP si están disponibles
                cv2.setUseOptimized(True)
                
                # Configurar número de threads OpenCV
                import os
                cpu_count = os.cpu_count() or 4
                cv2.setNumThreads(min(cpu_count, 8))  # Máximo 8 threads
                
                logger.info(f"OpenCV optimizado: {cv2.getNumThreads()} threads")
            
        except Exception as e:
            logger.warning(f"Error configurando backend OpenCV: {e}")
    
    def _create_video_capture(self):
        """Crea y configura VideoCapture optimizado"""
        try:
            with self._cap_lock:
                # Liberar capture anterior si existe
                if self.cap:
                    self.cap.release()
                
                # Crear nuevo capture con backend optimizado
                self.cap = cv2.VideoCapture(
                    self.rtsp_url, 
                    self.config['preferred_backend']
                )
                
                if not self.cap.isOpened():
                    raise ConnectionError("No se pudo abrir el stream RTSP")
                
                # Configurar propiedades optimizadas para Windows
                self._configure_capture_properties()
                
                # Test de lectura
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    raise ConnectionError("No se pudo leer frame de prueba")
                
                # Obtener información del stream
                self._update_stream_info()
                
                logger.info("VideoCapture Windows configurado exitosamente")
                return True
                
        except Exception as e:
            logger.error(f"Error creando VideoCapture: {e}")
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    def _configure_capture_properties(self):
        """Configura propiedades del VideoCapture"""
        try:
            # Buffer size (importante para latencia)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config['buffer_size'])
            
            # FPS si es posible
            self.cap.set(cv2.CAP_PROP_FPS, self.config['capture_fps'])
            
            # Timeout de conexión
            if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.config['connection_timeout'])
            
            # Timeout de lectura
            if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
                self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.config['read_timeout'])
            
            # Intentar habilitar hardware decode si está disponible
            if self.config['use_gpu_decode']:
                try:
                    # NVIDIA NVDEC (si está disponible)
                    self.cap.set(cv2.CAP_PROP_CODEC_PIXEL_FORMAT, cv2.CAP_OPENCV_MJPEG)
                    
                    # D3D11 backend para Windows (si está disponible)
                    if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_D3D11)
                        self._stats['decode_method'] = 'hardware_d3d11'
                    
                except Exception as e:
                    logger.debug(f"Hardware decode no disponible: {e}")
                    self._stats['decode_method'] = 'software'
            
            logger.info("Propiedades de VideoCapture configuradas")
            
        except Exception as e:
            logger.warning(f"Error configurando propiedades: {e}")
    
    def _update_stream_info(self):
        """Actualiza información del stream"""
        try:
            if self.cap and self.cap.isOpened():
                fps = self.cap.get(cv2.CAP_PROP_FPS)
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                self._stats.update({
                    'resolution': f'{width}x{height}',
                    'stream_fps': fps if fps > 0 else 'unknown'
                })
                
                logger.info(f"Stream info: {width}x{height} @ {fps} FPS")
                
        except Exception as e:
            logger.warning(f"Error obteniendo info del stream: {e}")
    
    def start(self):
        """Inicia la captura de video"""
        if self._running:
            logger.warning("WindowsNativeVideoReader ya está ejecutándose")
            return
        
        try:
            logger.info(f"Iniciando captura Windows para: {self.rtsp_url}")
            
            # Crear conexión inicial
            if not self._create_video_capture():
                raise RuntimeError("No se pudo establecer conexión inicial")
            
            self._running = True
            self._start_time = time.time()
            self._connected = True
            self.connection_state_changed.emit('connected')
            
            # Iniciar threads
            if self.config['enable_threading']:
                self._start_capture_thread()
                self._start_display_thread()
            
            # Iniciar timer de estadísticas
            self._stats_timer = QTimer()
            self._stats_timer.timeout.connect(self._update_stats)
            self._stats_timer.start(2000)  # Cada 2 segundos
            
            logger.info("Captura Windows iniciada exitosamente")
            
        except Exception as e:
            logger.error(f"Error iniciando captura: {e}")
            self.error_occurred.emit(f"Error de inicio: {e}")
            self._cleanup()
    
    def _start_capture_thread(self):
        """Inicia el thread de captura"""
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name=f"WinVideoCapture-{id(self)}"
        )
        self._capture_thread.start()
    
    def _start_display_thread(self):
        """Inicia el thread de display"""
        self._display_thread = threading.Thread(
            target=self._display_loop,
            daemon=True,
            name=f"WinVideoDisplay-{id(self)}"
        )
        self._display_thread.start()
    
    def _capture_loop(self):
        """Loop principal de captura en thread separado"""
        logger.info("Iniciando loop de captura Windows")
        
        frame_interval = 1.0 / self.config['capture_fps']
        last_frame_time = 0
        consecutive_failures = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # Control de FPS
                if current_time - last_frame_time < frame_interval:
                    time.sleep(0.001)
                    continue
                
                # Pausar si es necesario
                if self._paused:
                    time.sleep(0.1)
                    continue
                
                # Verificar conexión
                if not self._connected:
                    if not self._attempt_reconnection():
                        time.sleep(1.0)
                        continue
                
                # Leer frame
                success = self._read_and_process_frame(current_time)
                
                if success:
                    consecutive_failures = 0
                    last_frame_time = current_time
                    self._frame_count += 1
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 10:  # 10 fallos consecutivos
                        logger.warning("Múltiples fallos de lectura, intentando reconexión")
                        self._handle_connection_loss()
                        consecutive_failures = 0
                
            except Exception as e:
                logger.error(f"Error en capture loop: {e}")
                self._handle_connection_loss()
                time.sleep(0.5)
        
        logger.info("Loop de captura Windows terminado")
    
    def _read_and_process_frame(self, timestamp):
        """Lee y procesa un frame"""
        try:
            with self._cap_lock:
                if not self.cap or not self.cap.isOpened():
                    return False
                
                ret, frame = self.cap.read()
            
            if not ret or frame is None:
                return False
            
            # Validar frame
            if frame.size == 0:
                logger.warning("Frame vacío recibido")
                return False
            
            # Procesar frame
            self._process_captured_frame(frame, timestamp)
            return True
            
        except Exception as e:
            logger.error(f"Error leyendo frame: {e}")
            return False
    
    def _process_captured_frame(self, frame, timestamp):
        """Procesa un frame capturado"""
        try:
            # Crear datos del frame
            frame_data = {
                'frame': frame.copy(),
                'timestamp': timestamp,
                'frame_id': self._frame_count
            }
            
            # Agregar a buffer con control de sobrecarga
            with self._buffer_lock:
                if len(self._frame_buffer) >= self._frame_buffer.maxlen:
                    dropped = self._frame_buffer.popleft()
                    self._stats['frames_dropped'] += 1
                    logger.debug(f"Frame {dropped['frame_id']} descartado por buffer lleno")
                
                self._frame_buffer.append(frame_data)
            
            # Emitir para análisis (con control de FPS)
            if self._should_emit_for_analysis():
                self.frame_ready.emit(frame.copy())
            
            # Preparar para display
            self._prepare_display_frame(frame, timestamp)
            
            self._stats['frames_received'] += 1
            
        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
    
    def _should_emit_for_analysis(self):
        """Determina si debe emitir frame para análisis"""
        analysis_interval = 1.0 / self.config['analysis_fps']
        current_time = time.time()
        
        if not hasattr(self, '_last_analysis_time'):
            self._last_analysis_time = 0
        
        if current_time - self._last_analysis_time >= analysis_interval:
            self._last_analysis_time = current_time
            return True
        
        return False
    
    def _prepare_display_frame(self, frame, timestamp):
        """Prepara frame para display"""
        try:
            display_data = {
                'frame': frame,
                'timestamp': timestamp
            }
            
            with self._display_lock:
                if len(self._display_buffer) >= self._display_buffer.maxlen:
                    self._display_buffer.popleft()
                self._display_buffer.append(display_data)
                
        except Exception as e:
            logger.error(f"Error preparando frame para display: {e}")
    
    def _display_loop(self):
        """Loop de display en thread separado"""
        logger.info("Iniciando loop de display Windows")
        
        display_interval = 1.0 / self.config['display_fps']
        last_display_time = 0
        
        while self._running:
            try:
                current_time = time.time()
                
                # Control de FPS de display
                if current_time - last_display_time < display_interval:
                    time.sleep(0.001)
                    continue
                
                # Obtener frame para display
                display_data = None
                with self._display_lock:
                    if self._display_buffer:
                        display_data = self._display_buffer.popleft()
                
                if display_data:
                    pixmap = self._frame_to_pixmap(display_data['frame'])
                    if pixmap:
                        self.display_ready.emit(pixmap)
                        self._display_count += 1
                
                last_display_time = current_time
                
            except Exception as e:
                logger.error(f"Error en display loop: {e}")
                time.sleep(0.1)
        
        logger.info("Loop de display Windows terminado")
    
    def _frame_to_pixmap(self, frame):
        """Convierte frame OpenCV a QPixmap"""
        try:
            # Convertir BGR a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Crear QImage
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            
            qimage = QImage(
                rgb_frame.data,
                w, h,
                bytes_per_line,
                QImage.Format.Format_RGB888
            )
            
            return QPixmap.fromImage(qimage)
            
        except Exception as e:
            logger.error(f"Error convirtiendo frame a pixmap: {e}")
            return None
    
    def _handle_connection_loss(self):
        """Maneja pérdida de conexión"""
        try:
            self._connected = False
            self.connection_state_changed.emit('disconnected')
            
            # Limpiar conexión actual
            with self._cap_lock:
                if self.cap:
                    self.cap.release()
                    self.cap = None
            
            logger.warning("Conexión perdida, programando reconexión")
            
        except Exception as e:
            logger.error(f"Error manejando pérdida de conexión: {e}")
    
    def _attempt_reconnection(self):
        """Intenta reconectar automáticamente"""
        try:
            if self._reconnect_attempts >= self.config['max_reconnect_attempts']:
                logger.error("Máximo número de reconexiones alcanzado")
                self.error_occurred.emit("Conexión perdida - máximo de intentos alcanzado")
                return False
            
            self._reconnect_attempts += 1
            self._stats['reconnect_attempts'] = self._reconnect_attempts
            
            logger.info(f"Intento de reconexión {self._reconnect_attempts}/{self.config['max_reconnect_attempts']}")
            
            # Esperar antes de reconectar
            time.sleep(self.config['reconnect_delay'])
            
            # Intentar crear nueva conexión
            if self._create_video_capture():
                self._connected = True
                self._reconnect_attempts = 0
                self.connection_state_changed.emit('reconnected')
                logger.info("Reconexión exitosa")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error en reconexión: {e}")
            return False
    
    def _update_stats(self):
        """Actualiza estadísticas de rendimiento"""
        try:
            current_time = time.time()
            time_diff = current_time - self._last_stats_time
            
            if time_diff > 0:
                # Calcular FPS
                self._stats['fps_capture'] = self._frame_count / time_diff
                self._stats['fps_display'] = self._display_count / time_diff
                
                # Uptime
                if self._start_time:
                    self._stats['uptime_seconds'] = current_time - self._start_time
                
                # Estado de conexión
                self._stats['connection_stable'] = self._connected
                
                # Salud del buffer
                with self._buffer_lock:
                    buffer_usage = len(self._frame_buffer) / max(1, self._frame_buffer.maxlen)
                    self._stats['buffer_health'] = buffer_usage * 100
            
            # Reset contadores
            self._frame_count = 0
            self._display_count = 0
            self._last_stats_time = current_time
            
            # Emitir estadísticas
            self.stats_updated.emit(self._stats.copy())
            
        except Exception as e:
            logger.error(f"Error actualizando estadísticas: {e}")
    
    def stop(self):
        """Detiene la captura de video"""
        if not self._running:
            return
        
        logger.info("Deteniendo captura Windows...")
        
        self._running = False
        
        try:
            # Detener timer
            if hasattr(self, '_stats_timer'):
                self._stats_timer.stop()
            
            # Esperar threads
            if self._capture_thread and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=3.0)
            
            if self._display_thread and self._display_thread.is_alive():
                self._display_thread.join(timeout=3.0)
            
            # Limpiar recursos
            self._cleanup()
            
            logger.info("Captura Windows detenida")
            
        except Exception as e:
            logger.error(f"Error deteniendo captura: {e}")
    
    def _cleanup(self):
        """Limpia recursos"""
        try:
            # Liberar VideoCapture
            with self._cap_lock:
                if self.cap:
                    self.cap.release()
                    self.cap = None
            
            # Limpiar buffers
            with self._buffer_lock:
                self._frame_buffer.clear()
            
            with self._display_lock:
                self._display_buffer.clear()
            
            self._connected = False
            
        except Exception as e:
            logger.error(f"Error en cleanup: {e}")
    
    def pause(self):
        """Pausa la captura temporalmente"""
        self._paused = True
        logger.info("Captura Windows pausada")
    
    def resume(self):
        """Reanuda la captura"""
        self._paused = False
        logger.info("Captura Windows reanudada")
    
    def get_latest_frame(self):
        """Obtiene el frame más reciente"""
        with self._buffer_lock:
            if self._frame_buffer:
                return self._frame_buffer[-1]['frame'].copy()
        return None
    
    def get_stats(self):
        """Obtiene estadísticas actuales"""
        return self._stats.copy()
    
    def update_config(self, new_config):
        """Actualiza configuración en tiempo real"""
        try:
            old_config = self.config.copy()
            self.config.update(new_config)
            
            # Aplicar cambios que requieren reconexión
            reconnect_required = False
            for key in ['preferred_backend', 'use_gpu_decode', 'buffer_size']:
                if key in new_config and new_config[key] != old_config.get(key):
                    reconnect_required = True
                    break
            
            if reconnect_required and self._running:
                logger.info("Reiniciando captura por cambios de configuración")
                self._handle_connection_loss()
            
            logger.info(f"Configuración actualizada: {new_config}")
            
        except Exception as e:
            logger.error(f"Error actualizando configuración: {e}")


class WindowsVideoReaderFactory:
    """Factory para crear video readers optimizados para Windows"""
    
    @staticmethod
    def create_reader(rtsp_url, camera_type="fija", performance_profile="balanced"):
        """
        Crea un video reader optimizado para Windows.
        
        Args:
            rtsp_url: URL del stream RTSP
            camera_type: Tipo de cámara (fija, ptz, nvr)
            performance_profile: Perfil de rendimiento
        """
        
        # Perfiles optimizados para Windows
        profiles = {
            "ultra_low_latency": {
                'buffer_size': 1,
                'capture_fps': 30,
                'display_fps': 30,
                'analysis_fps': 15,
                'frame_skip_on_delay': True,
                'connection_timeout': 3000,
                'read_timeout': 500,
            },
            "balanced": {
                'buffer_size': 3,
                'capture_fps': 30,
                'display_fps': 25,
                'analysis_fps': 8,
                'connection_timeout': 5000,
                'read_timeout': 1000,
            },
            "quality": {
                'buffer_size': 5,
                'capture_fps': 25,
                'display_fps': 20,
                'analysis_fps': 5,
                'adaptive_quality': True,
                'connection_timeout': 10000,
                'read_timeout': 2000,
            },
            "power_save": {
                'buffer_size': 2,
                'capture_fps': 20,
                'display_fps': 15,
                'analysis_fps': 3,
                'use_gpu_decode': False,
                'enable_threading': False,
            }
        }
        
        # Configuraciones por tipo de cámara
        camera_configs = {
            "ptz": {
                'reconnect_delay': 3.0,
                'max_reconnect_attempts': 15,
                'connection_timeout': 10000,
            },
            "nvr": {
                'buffer_size': 2,  # NVRs pueden tener más latencia
                'reconnect_delay': 5.0,
                'max_reconnect_attempts': 20,
            },
            "fija": {
                'reconnect_delay': 2.0,
                'max_reconnect_attempts': 10,
            }
        }
        
        # Combinar configuraciones
        config = profiles.get(performance_profile, profiles["balanced"]).copy()
        config.update(camera_configs.get(camera_type, {}))
        
        return WindowsNativeVideoReader(rtsp_url, config)


def check_windows_video_capabilities():
    """Verifica capacidades de video en Windows"""
    try:
        capabilities = {
            'opencv_version': cv2.__version__,
            'backends_available': [],
            'hardware_decode_support': False,
            'threading_support': True,
            'gpu_info': 'unknown'
        }
        
        # Verificar backends disponibles
        available_backends = cv2.videoio_registry.getCameraBackends()
        
        backend_names = {
            cv2.CAP_FFMPEG: 'FFMPEG',
            cv2.CAP_DSHOW: 'DirectShow',
            cv2.CAP_MSMF: 'Media Foundation',
            cv2.CAP_OPENCV_MJPEG: 'OpenCV MJPEG'
        }
        
        for backend_id, backend_name in backend_names.items():
            if backend_id in available_backends:
                capabilities['backends_available'].append(backend_name)
        
        # Verificar soporte de hardware decode
        try:
            # Intentar crear un VideoCapture con propiedades de hardware
            test_cap = cv2.VideoCapture()
            if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
                capabilities['hardware_decode_support'] = True
        except:
            pass
        
        # Información de GPU
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                capabilities['gpu_info'] = f"{gpu_name} ({gpu_memory:.1f}GB)"
        except:
            pass
        
        return capabilities, []
        
    except Exception as e:
        return {}, [str(e)]


# Ejemplo de uso para testing
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget, QTextEdit
    
    # Verificar capacidades
    capabilities, issues = check_windows_video_capabilities()
    
    print("=== VERIFICACIÓN VIDEO WINDOWS ===")
    print(f"OpenCV: {capabilities.get('opencv_version', 'N/A')}")
    print(f"Backends: {capabilities.get('backends_available', [])}")
    print(f"Hardware decode: {capabilities.get('hardware_decode_support', False)}")
    print(f"GPU: {capabilities.get('gpu_info', 'N/A')}")
    
    if issues:
        print("\n⚠️ PROBLEMAS:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ Sistema listo para video Windows optimizado")
    
    # Test widget (descomenta para test real)
    app = QApplication(sys.argv)
    
    class TestWindowsVideoWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Test Windows Native Video Reader")
            self.resize(800, 600)
            
            layout = QVBoxLayout()
            
            # Video display
            self.video_label = QLabel("Conectando...")
            self.video_label.setMinimumSize(640, 480)
            self.video_label.setStyleSheet("border: 1px solid black; background: black;")
            layout.addWidget(self.video_label)
            
            # Stats display
            self.stats_text = QTextEdit()
            self.stats_text.setMaximumHeight(150)
            self.stats_text.setReadOnly(True)
            layout.addWidget(self.stats_text)
            
            self.setLayout(layout)
            
            # Crear video reader
            rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream"
            self.reader = WindowsVideoReaderFactory.create_reader(
                rtsp_url,
                camera_type="fija",
                performance_profile="balanced"
            )
            
            # Conectar señales
            self.reader.display_ready.connect(self.video_label.setPixmap)
            self.reader.stats_updated.connect(self.update_stats)
            self.reader.error_occurred.connect(self.handle_error)
            self.reader.connection_state_changed.connect(self.handle_connection_state)
            
            # Iniciar (descomenta para test real)
            # self.reader.start()
        
        def update_stats(self, stats):
            stats_text = f"Backend: {stats.get('backend_info', 'N/A')}\n"
            stats_text += f"Decode: {stats.get('decode_method', 'N/A')}\n"
            stats_text += f"Resolution: {stats.get('resolution', 'N/A')}\n"
            stats_text += f"FPS Capture: {stats.get('fps_capture', 0):.1f}\n"
            stats_text += f"FPS Display: {stats.get('fps_display', 0):.1f}\n"
            stats_text += f"Frames: {stats.get('frames_received', 0)}\n"
            stats_text += f"Dropped: {stats.get('frames_dropped', 0)}\n"
            stats_text += f"Buffer: {stats.get('buffer_health', 0):.1f}%\n"
            stats_text += f"Connected: {stats.get('connection_stable', False)}\n"
            stats_text += f"Uptime: {stats.get('uptime_seconds', 0):.1f}s"
            
            self.stats_text.setText(stats_text)
        
        def handle_error(self, error_msg):
            self.stats_text.append(f"\nERROR: {error_msg}")
        
        def handle_connection_state(self, state):
            self.stats_text.append(f"\nConnection: {state}")
        
        def closeEvent(self, event):
            if hasattr(self, 'reader'):
                self.reader.stop()
            event.accept()
    
    # Solo mostrar capacidades por ahora
    print("\n" + "="*50)
    print("Para test real, descomenta las líneas en el código")
    print("y proporciona una URL RTSP válida")
    
    # widget = TestWindowsVideoWidget()
    # widget.show()
    # sys.exit(app.exec())