"""
Sistema de Seguimiento PTZ Profesional
Maneja el seguimiento automático de objetos con cámaras PTZ
"""

import threading
import time
import queue
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from core.light_api import LightAPI
from core.grid_utils import GridUtils

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PTZTrackingMode(Enum):
    """Modos de seguimiento PTZ"""
    DISABLED = "disabled"
    TRACKING = "tracking"
    ANALYTICS_ONLY = "analytics_only"

@dataclass
class DetectionEvent:
    """Evento de detección para seguimiento"""
    object_id: str
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    center: Tuple[int, int]  # cx, cy
    grid_cell: Tuple[int, int]  # fila, columna
    timestamp: float
    source_camera_ip: str
    frame_dimensions: Tuple[int, int]  # width, height

@dataclass
class PTZCameraConfig:
    """Configuración de cámara PTZ"""
    ip: str
    port: int
    username: str
    password: str
    tracking_mode: PTZTrackingMode
    tracking_enabled: bool = False
    # Configuración de seguimiento
    tracking_sensitivity: float = 0.005
    max_pt_speed: float = 0.5
    deadzone_x: float = 0.03
    deadzone_y: float = 0.03
    confirmation_frames: int = 3
    # Configuración de zona de seguimiento
    tracking_grid_cells: List[Tuple[int, int]] = None
    # API Light
    light_api: Optional[LightAPI] = None

class PTZTrackingThread(threading.Thread):
    """Hilo principal para el seguimiento PTZ"""
    
    def __init__(self, config: PTZCameraConfig):
        super().__init__(daemon=True)
        self.config = config
        self.detection_queue = queue.Queue(maxsize=100)
        self.running = False
        self.current_target = None
        self.confirmation_counter = 0
        self.last_movement_time = 0
        
        # Inicializar API Light
        self.light_api = LightAPI(
            ip=config.ip,
            port=config.port,
            username=config.username,
            password=config.password
        )
        
    def start_tracking(self):
        """Iniciar el seguimiento"""
        if not self.config.tracking_enabled:
            logger.warning(f"Seguimiento no habilitado para cámara {self.config.ip}")
            return False
            
        try:
            # Verificar conectividad con la cámara
            if not self.light_api.test_connection():
                logger.error(f"No se puede conectar a la cámara PTZ {self.config.ip}")
                return False
                
            self.running = True
            self.start()
            logger.info(f"Seguimiento PTZ iniciado para {self.config.ip}")
            return True
            
        except Exception as e:
            logger.error(f"Error al iniciar seguimiento PTZ: {e}")
            return False
    
    def stop_tracking(self):
        """Detener el seguimiento"""
        self.running = False
        # Detener movimiento de la cámara
        self.light_api.stop_movement()
        logger.info(f"Seguimiento PTZ detenido para {self.config.ip}")
    
    def add_detection(self, detection: DetectionEvent):
        """Agregar detección a la cola de procesamiento"""
        try:
            # Verificar si la detección está en las celdas de seguimiento
            if self.config.tracking_grid_cells:
                if detection.grid_cell not in self.config.tracking_grid_cells:
                    return  # Ignorar detecciones fuera de la zona de seguimiento
            
            self.detection_queue.put(detection, timeout=0.1)
        except queue.Full:
            logger.warning("Cola de detecciones llena, descartando detección más antigua")
            try:
                self.detection_queue.get_nowait()
                self.detection_queue.put(detection, timeout=0.1)
            except queue.Empty:
                pass
    
    def run(self):
        """Bucle principal del hilo de seguimiento"""
        logger.info(f"Iniciando bucle de seguimiento para {self.config.ip}")
        
        while self.running:
            try:
                # Procesar detecciones
                detection = self.detection_queue.get(timeout=1.0)
                self.process_detection(detection)
                
            except queue.Empty:
                # Sin detecciones, verificar si hay que detener seguimiento
                self.handle_no_detection()
                continue
                
            except Exception as e:
                logger.error(f"Error en bucle de seguimiento: {e}")
                time.sleep(0.1)
    
    def process_detection(self, detection: DetectionEvent):
        """Procesar una detección y actualizar seguimiento"""
        current_time = time.time()
        
        # Verificar si es el mismo objetivo
        if self.current_target and self.current_target.object_id == detection.object_id:
            self.confirmation_counter += 1
        else:
            # Nuevo objetivo
            self.current_target = detection
            self.confirmation_counter = 1
        
        # Requiere confirmación antes de seguir
        if self.confirmation_counter < self.config.confirmation_frames:
            logger.debug(f"Esperando confirmación {self.confirmation_counter}/{self.config.confirmation_frames}")
            return
        
        # Calcular movimiento PTZ
        self.calculate_and_execute_movement(detection)
        self.last_movement_time = current_time
    
    def calculate_and_execute_movement(self, detection: DetectionEvent):
        """Calcular y ejecutar movimiento PTZ basado en la detección"""
        frame_w, frame_h = detection.frame_dimensions
        cx, cy = detection.center
        
        # Calcular centro del frame
        center_x = frame_w / 2
        center_y = frame_h / 2
        
        # Calcular diferencias
        dx = cx - center_x
        dy = cy - center_y
        
        # Aplicar zona muerta
        deadzone_x_pixels = frame_w * self.config.deadzone_x
        deadzone_y_pixels = frame_h * self.config.deadzone_y
        
        if abs(dx) < deadzone_x_pixels:
            dx = 0
        if abs(dy) < deadzone_y_pixels:
            dy = 0
        
        # Si está centrado, detener movimiento
        if dx == 0 and dy == 0:
            self.light_api.stop_movement()
            logger.info("🎯 Objetivo centrado - Deteniendo movimiento")
            return
        
        # Calcular velocidades
        pan_speed = self._calculate_speed(dx, frame_w, self.config.tracking_sensitivity)
        tilt_speed = self._calculate_speed(-dy, frame_h, self.config.tracking_sensitivity)  # Invertir Y
        
        # Limitar velocidades
        pan_speed = max(-self.config.max_pt_speed, min(self.config.max_pt_speed, pan_speed))
        tilt_speed = max(-self.config.max_pt_speed, min(self.config.max_pt_speed, tilt_speed))
        
        # Ejecutar movimiento
        success = self.light_api.continuous_move(pan_speed, tilt_speed, 0.0)
        
        if success:
            logger.info(f"🎯 Seguimiento PTZ: pan={pan_speed:.3f}, tilt={tilt_speed:.3f}")
        else:
            logger.error("❌ Error ejecutando movimiento PTZ")
    
    def _calculate_speed(self, delta, frame_size, sensitivity):
        """Calcular velocidad basada en el delta y sensibilidad"""
        normalized_delta = delta / frame_size
        return normalized_delta * sensitivity
    
    def handle_no_detection(self):
        """Manejar caso cuando no hay detecciones"""
        current_time = time.time()
        
        # Si ha pasado tiempo sin detecciones, detener seguimiento
        if self.last_movement_time > 0 and (current_time - self.last_movement_time) > 2.0:
            self.light_api.stop_movement()
            self.current_target = None
            self.confirmation_counter = 0
            self.last_movement_time = 0
            logger.debug("⏹️ Timeout sin detecciones - Deteniendo seguimiento")

class PTZTrackingManager:
    """Gestor principal del sistema de seguimiento PTZ"""
    
    def __init__(self):
        self.tracking_threads: Dict[str, PTZTrackingThread] = {}
        self.camera_configs: Dict[str, PTZCameraConfig] = {}
        self.grid_utils = GridUtils()
        
    def load_configuration(self, config_file: str = "ptz_tracking_config.json"):
        """Cargar configuración desde archivo"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            for cam_data in config_data.get("ptz_cameras", []):
                config = PTZCameraConfig(
                    ip=cam_data["ip"],
                    port=cam_data.get("port", 80),
                    username=cam_data["username"],
                    password=cam_data["password"],
                    tracking_mode=PTZTrackingMode(cam_data.get("tracking_mode", "disabled")),
                    tracking_enabled=cam_data.get("tracking_enabled", False),
                    tracking_sensitivity=cam_data.get("tracking_sensitivity", 0.005),
                    max_pt_speed=cam_data.get("max_pt_speed", 0.5),
                    deadzone_x=cam_data.get("deadzone_x", 0.03),
                    deadzone_y=cam_data.get("deadzone_y", 0.03),
                    confirmation_frames=cam_data.get("confirmation_frames", 3),
                    tracking_grid_cells=cam_data.get("tracking_grid_cells", [])
                )
                
                self.camera_configs[config.ip] = config
                logger.info(f"Configuración cargada para cámara PTZ: {config.ip}")
                
        except FileNotFoundError:
            logger.warning(f"Archivo de configuración {config_file} no encontrado")
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
    
    def start_tracking_for_camera(self, camera_ip: str) -> bool:
        """Iniciar seguimiento para una cámara específica"""
        if camera_ip not in self.camera_configs:
            logger.error(f"No hay configuración para la cámara {camera_ip}")
            return False
        
        config = self.camera_configs[camera_ip]
        
        if not config.tracking_enabled:
            logger.warning(f"Seguimiento no habilitado para {camera_ip}")
            return False
        
        # Crear y iniciar hilo de seguimiento
        tracking_thread = PTZTrackingThread(config)
        
        if tracking_thread.start_tracking():
            self.tracking_threads[camera_ip] = tracking_thread
            return True
        
        return False
    
    def stop_tracking_for_camera(self, camera_ip: str):
        """Detener seguimiento para una cámara específica"""
        if camera_ip in self.tracking_threads:
            self.tracking_threads[camera_ip].stop_tracking()
            del self.tracking_threads[camera_ip]
            logger.info(f"Seguimiento detenido para {camera_ip}")
    
    def send_detection_to_tracker(self, detection_data: dict, source_camera_ip: str):
        """Enviar detección al sistema de seguimiento"""
        # Convertir datos de detección a evento
        detection = DetectionEvent(
            object_id=detection_data.get("object_id", "unknown"),
            object_type=detection_data.get("object_type", "unknown"),
            confidence=detection_data.get("confidence", 0.0),
            bbox=tuple(detection_data.get("bbox", [0, 0, 0, 0])),
            center=tuple(detection_data.get("center", [0, 0])),
            grid_cell=tuple(detection_data.get("grid_cell", [0, 0])),
            timestamp=time.time(),
            source_camera_ip=source_camera_ip,
            frame_dimensions=tuple(detection_data.get("frame_dimensions", [640, 480]))
        )
        
        # Enviar a todas las cámaras PTZ configuradas para seguimiento
        for camera_ip, thread in self.tracking_threads.items():
            if thread.running:
                thread.add_detection(detection)
    
    def get_tracking_status(self) -> Dict[str, dict]:
        """Obtener estado actual del seguimiento"""
        status = {}
        
        for camera_ip, thread in self.tracking_threads.items():
            status[camera_ip] = {
                "running": thread.running,
                "current_target": thread.current_target.object_id if thread.current_target else None,
                "confirmation_counter": thread.confirmation_counter,
                "queue_size": thread.detection_queue.qsize()
            }
        
        return status
    
    def shutdown(self):
        """Cerrar todos los hilos de seguimiento"""
        for camera_ip in list(self.tracking_threads.keys()):
            self.stop_tracking_for_camera(camera_ip)
        
        logger.info("Sistema de seguimiento PTZ cerrado")

# Instancia global del gestor
ptz_tracking_manager = PTZTrackingManager()