"""
Integración del Sistema PTZ con el Sistema Principal
Conecta el seguimiento PTZ con la detección de objetos y la interfaz principal
"""

import json
import logging
import threading
from typing import Dict, List, Optional, Callable
from dataclasses import asdict

from core.ptz_tracking_system import (
    PTZTrackingManager, 
    PTZCameraConfig, 
    PTZTrackingMode,
    DetectionEvent,
    ptz_tracking_manager
)
from core.light_api import LightAPI
from gui.ptz_config_widget import PTZConfigWidget

logger = logging.getLogger(__name__)

class PTZSystemIntegration:
    """Clase principal de integración del sistema PTZ"""
    
    def __init__(self, main_app=None):
        self.main_app = main_app
        self.tracking_manager = ptz_tracking_manager
        self.config_widgets: Dict[str, PTZConfigWidget] = {}
        self.detection_callbacks: List[Callable] = []
        
        # Cargar configuración al inicializar
        self.load_configuration()
        
    def load_configuration(self, config_file: str = "ptz_tracking_config.json"):
        """Cargar configuración PTZ desde archivo"""
        try:
            self.tracking_manager.load_configuration(config_file)
            logger.info(f"Configuración PTZ cargada desde {config_file}")
        except Exception as e:
            logger.error(f"Error cargando configuración PTZ: {e}")
            self.create_default_config(config_file)
    
    def create_default_config(self, config_file: str):
        """Crear configuración por defecto"""
        default_config = {
            "ptz_cameras": [],
            "global_settings": {
                "auto_start_tracking": False,
                "log_level": "INFO",
                "max_tracking_distance": 0.3,
                "tracking_timeout": 10.0
            }
        }
        
        try:
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
            logger.info(f"Configuración por defecto creada: {config_file}")
        except Exception as e:
            logger.error(f"Error creando configuración por defecto: {e}")
    
    def add_ptz_camera(self, camera_config: Dict) -> bool:
        """
        Agregar nueva cámara PTZ al sistema
        
        Args:
            camera_config: Diccionario con configuración de cámara
            
        Returns:
            bool: True si se agregó exitosamente
        """
        try:
            # Crear configuración PTZ
            config = PTZCameraConfig(
                ip=camera_config["ip"],
                port=camera_config.get("port", 80),
                username=camera_config["username"],
                password=camera_config["password"],
                tracking_mode=PTZTrackingMode(camera_config.get("tracking_mode", "disabled")),
                tracking_enabled=camera_config.get("tracking_enabled", False),
                tracking_sensitivity=camera_config.get("tracking_sensitivity", 0.005),
                max_pt_speed=camera_config.get("max_pt_speed", 0.5),
                deadzone_x=camera_config.get("deadzone_x", 0.03),
                deadzone_y=camera_config.get("deadzone_y", 0.03),
                confirmation_frames=camera_config.get("confirmation_frames", 3),
                tracking_grid_cells=camera_config.get("tracking_grid_cells", [])
            )
            
            # Agregar al gestor de seguimiento
            self.tracking_manager.camera_configs[config.ip] = config
            
            # Si el seguimiento está habilitado, iniciarlo
            if config.tracking_enabled:
                success = self.tracking_manager.start_tracking_for_camera(config.ip)
                if success:
                    logger.info(f"Seguimiento iniciado para cámara PTZ {config.ip}")
                else:
                    logger.warning(f"No se pudo iniciar seguimiento para {config.ip}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error agregando cámara PTZ: {e}")
            return False
    
    def remove_ptz_camera(self, camera_ip: str) -> bool:
        """
        Remover cámara PTZ del sistema
        
        Args:
            camera_ip: IP de la cámara a remover
            
        Returns:
            bool: True si se removió exitosamente
        """
        try:
            # Detener seguimiento si está activo
            self.tracking_manager.stop_tracking_for_camera(camera_ip)
            
            # Remover configuración
            if camera_ip in self.tracking_manager.camera_configs:
                del self.tracking_manager.camera_configs[camera_ip]
            
            # Cerrar widget de configuración si existe
            if camera_ip in self.config_widgets:
                self.config_widgets[camera_ip].close()
                del self.config_widgets[camera_ip]
            
            logger.info(f"Cámara PTZ {camera_ip} removida del sistema")
            return True
            
        except Exception as e:
            logger.error(f"Error removiendo cámara PTZ {camera_ip}: {e}")
            return False
    
    def open_ptz_config_window(self, camera_ip: str) -> Optional[PTZConfigWidget]:
        """
        Abrir ventana de configuración PTZ para una cámara
        
        Args:
            camera_ip: IP de la cámara
            
        Returns:
            PTZConfigWidget: Widget de configuración o None si hay error
        """
        try:
            # Verificar si ya existe una ventana abierta
            if camera_ip in self.config_widgets:
                widget = self.config_widgets[camera_ip]
                widget.raise_()
                widget.activateWindow()
                return widget
            
            # Crear nueva ventana de configuración
            config_widget = PTZConfigWidget()
            
            # Configurar cámara si existe en la configuración
            if camera_ip in self.tracking_manager.camera_configs:
                config = self.tracking_manager.camera_configs[camera_ip]
                config_widget.set_camera_config(config)
            
            # Conectar señales
            config_widget.configuration_changed.connect(
                lambda config_data: self.on_configuration_changed(camera_ip, config_data)
            )
            config_widget.tracking_toggled.connect(
                lambda ip, enabled: self.on_tracking_toggled(ip, enabled)
            )
            
            # Mostrar ventana
            config_widget.show()
            
            # Guardar referencia
            self.config_widgets[camera_ip] = config_widget
            
            # Conectar señal de cierre para limpiar referencia
            config_widget.finished.connect(
                lambda: self.config_widgets.pop(camera_ip, None)
            )
            
            logger.info(f"Ventana de configuración PTZ abierta para {camera_ip}")
            return config_widget
            
        except Exception as e:
            logger.error(f"Error abriendo configuración PTZ para {camera_ip}: {e}")
            return None
    
    def send_detection_to_trackers(self, detection_data: Dict, source_camera_ip: str):
        """
        Enviar detección a todas las cámaras PTZ configuradas para seguimiento
        
        Args:
            detection_data: Datos de la detección
            source_camera_ip: IP de la cámara que generó la detección
        """
        try:
            # Convertir datos a formato estándar si es necesario
            standardized_detection = self.standardize_detection_data(detection_data, source_camera_ip)
            
            # Enviar a gestor de seguimiento
            self.tracking_manager.send_detection_to_tracker(standardized_detection, source_camera_ip)
            
            # Ejecutar callbacks registrados
            for callback in self.detection_callbacks:
                try:
                    callback(standardized_detection, source_camera_ip)
                except Exception as e:
                    logger.error(f"Error en callback de detección: {e}")
                    
        except Exception as e:
            logger.error(f"Error enviando detección a trackers: {e}")
    
    def standardize_detection_data(self, detection_data: Dict, source_camera_ip: str) -> Dict:
        """
        Estandarizar datos de detección para el sistema PTZ
        
        Args:
            detection_data: Datos originales de detección
            source_camera_ip: IP de la cámara fuente
            
        Returns:
            Dict: Datos estandarizados
        """
        try:
            # Extraer información básica
            bbox = detection_data.get("bbox", [0, 0, 0, 0])
            
            # Calcular centro si no está presente
            if "center" not in detection_data:
                x, y, w, h = bbox
                center = (x + w // 2, y + h // 2)
            else:
                center = detection_data["center"]
            
            # Calcular celda de grilla si no está presente
            if "grid_cell" not in detection_data:
                frame_w = detection_data.get("frame_width", 640)
                frame_h = detection_data.get("frame_height", 480)
                grid_cell = self.calculate_grid_cell(center, frame_w, frame_h)
            else:
                grid_cell = detection_data["grid_cell"]
            
            # Crear datos estandarizados
            standardized = {
                "object_id": detection_data.get("object_id", f"obj_{int(time.time())}"),
                "object_type": detection_data.get("object_type", "unknown"),
                "confidence": detection_data.get("confidence", 0.0),
                "bbox": bbox,
                "center": center,
                "grid_cell": grid_cell,
                "frame_dimensions": (
                    detection_data.get("frame_width", 640),
                    detection_data.get("frame_height", 480)
                ),
                "timestamp": detection_data.get("timestamp", time.time()),
                "source_camera_ip": source_camera_ip
            }
            
            return standardized
            
        except Exception as e:
            logger.error(f"Error estandarizando datos de detección: {e}")
            return detection_data
    
    def calculate_grid_cell(self, center: tuple, frame_w: int, frame_h: int, 
                          grid_rows: int = 12, grid_cols: int = 16) -> tuple:
        """
        Calcular celda de grilla basada en el centro del objeto
        
        Args:
            center: Coordenadas del centro (x, y)
            frame_w: Ancho del frame
            frame_h: Alto del frame
            grid_rows: Número de filas de la grilla
            grid_cols: Número de columnas de la grilla
            
        Returns:
            tuple: (fila, columna) de la celda
        """
        try:
            cx, cy = center
            
            # Calcular tamaño de celda
            cell_w = frame_w / grid_cols
            cell_h = frame_h / grid_rows
            
            # Calcular posición en la grilla
            col = min(int(cx / cell_w), grid_cols - 1)
            row = min(int(cy / cell_h), grid_rows - 1)
            
            return (row, col)
            
        except Exception:
            return (0, 0)
    
    def get_tracking_status(self) -> Dict[str, Dict]:
        """
        Obtener estado actual de todas las cámaras PTZ
        
        Returns:
            Dict: Estado de seguimiento por cámara
        """
        try:
            status = self.tracking_manager.get_tracking_status()
            
            # Agregar información adicional
            enhanced_status = {}
            for camera_ip, camera_status in status.items():
                config = self.tracking_manager.camera_configs.get(camera_ip)
                
                enhanced_status[camera_ip] = {
                    **camera_status,
                    "tracking_mode": config.tracking_mode.value if config else "unknown",
                    "tracking_enabled": config.tracking_enabled if config else False,
                    "has_config_window": camera_ip in self.config_widgets
                }
            
            return enhanced_status
            
        except Exception as e:
            logger.error(f"Error obteniendo estado de seguimiento: {e}")
            return {}

    def get_status(self) -> Dict[str, Dict]:
        """Alias para compatibilidad retroactiva.

        Algunos módulos antiguos aún llaman a ``get_status`` para
        obtener el estado de las cámaras PTZ.  Desde versiones
        recientes el método público se denomina
        :func:`get_tracking_status`.  Esta función simplemente delega
        la llamada para mantener compatibilidad con esos módulos.
        """

        return self.get_tracking_status()
    
    def register_detection_callback(self, callback: Callable):
        """
        Registrar callback para recibir notificaciones de detección
        
        Args:
            callback: Función que recibe (detection_data, source_camera_ip)
        """
        if callback not in self.detection_callbacks:
            self.detection_callbacks.append(callback)
            logger.info("Callback de detección registrado")
    
    def unregister_detection_callback(self, callback: Callable):
        """
        Desregistrar callback de detección
        
        Args:
            callback: Función a desregistrar
        """
        if callback in self.detection_callbacks:
            self.detection_callbacks.remove(callback)
            logger.info("Callback de detección desregistrado")
    
    def save_configuration(self, config_file: str = "ptz_tracking_config.json"):
        """
        Guardar configuración actual en archivo
        
        Args:
            config_file: Ruta del archivo de configuración
        """
        try:
            # Recopilar configuraciones de cámaras
            ptz_cameras = []
            
            for camera_ip, config in self.tracking_manager.camera_configs.items():
                camera_data = {
                    "ip": config.ip,
                    "port": config.port,
                    "username": config.username,
                    "password": config.password,
                    "tracking_mode": config.tracking_mode.value,
                    "tracking_enabled": config.tracking_enabled,
                    "tracking_sensitivity": config.tracking_sensitivity,
                    "max_pt_speed": config.max_pt_speed,
                    "deadzone_x": config.deadzone_x,
                    "deadzone_y": config.deadzone_y,
                    "confirmation_frames": config.confirmation_frames,
                    "tracking_grid_cells": config.tracking_grid_cells or []
                }
                ptz_cameras.append(camera_data)
            
            # Crear estructura de configuración completa
            config_data = {
                "ptz_cameras": ptz_cameras,
                "global_settings": {
                    "auto_start_tracking": True,
                    "log_level": "INFO",
                    "max_tracking_distance": 0.3,
                    "tracking_timeout": 10.0
                },
                "version": "1.0",
                "last_updated": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Guardar en archivo
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            logger.info(f"Configuración PTZ guardada en {config_file}")
            
        except Exception as e:
            logger.error(f"Error guardando configuración PTZ: {e}")
    
    def start_all_tracking(self) -> Dict[str, bool]:
        """
        Iniciar seguimiento para todas las cámaras habilitadas
        
        Returns:
            Dict: Resultado por cámara {ip: success}
        """
        results = {}
        
        try:
            for camera_ip, config in self.tracking_manager.camera_configs.items():
                if config.tracking_enabled:
                    success = self.tracking_manager.start_tracking_for_camera(camera_ip)
                    results[camera_ip] = success
                    
                    if success:
                        logger.info(f"Seguimiento iniciado para {camera_ip}")
                    else:
                        logger.error(f"Error iniciando seguimiento para {camera_ip}")
                else:
                    results[camera_ip] = False
                    logger.info(f"Seguimiento no habilitado para {camera_ip}")
            
        except Exception as e:
            logger.error(f"Error iniciando seguimiento masivo: {e}")
        
        return results
    
    def stop_all_tracking(self):
        """Detener seguimiento para todas las cámaras"""
        try:
            for camera_ip in list(self.tracking_manager.tracking_threads.keys()):
                self.tracking_manager.stop_tracking_for_camera(camera_ip)
            
            logger.info("Seguimiento detenido para todas las cámaras")
            
        except Exception as e:
            logger.error(f"Error deteniendo seguimiento masivo: {e}")
    
    def emergency_stop_all(self):
        """Parada de emergencia - detener todo movimiento PTZ inmediatamente"""
        try:
            logger.warning("PARADA DE EMERGENCIA PTZ ACTIVADA")
            
            # Detener todos los seguimientos
            self.stop_all_tracking()
            
            # Enviar comando STOP a todas las cámaras PTZ configuradas
            for camera_ip, config in self.tracking_manager.camera_configs.items():
                try:
                    light_api = LightAPI(config.ip, config.port, config.username, config.password)
                    light_api.stop_movement()
                    logger.info(f"Comando STOP enviado a {camera_ip}")
                except Exception as e:
                    logger.error(f"Error enviando STOP a {camera_ip}: {e}")
            
        except Exception as e:
            logger.error(f"Error en parada de emergencia: {e}")
    
    # =============================================================================
    # MÉTODOS DE CALLBACK PARA SEÑALES
    # =============================================================================
    
    def on_configuration_changed(self, camera_ip: str, config_data: Dict):
        """
        Manejar cambio de configuración desde widget
        
        Args:
            camera_ip: IP de la cámara
            config_data: Nuevos datos de configuración
        """
        try:
            # Actualizar configuración
            if "ptz_cameras" in config_data and config_data["ptz_cameras"]:
                cam_data = config_data["ptz_cameras"][0]
                self.add_ptz_camera(cam_data)
            
            # Guardar configuración automáticamente
            self.save_configuration()
            
            logger.info(f"Configuración actualizada para {camera_ip}")
            
        except Exception as e:
            logger.error(f"Error actualizando configuración para {camera_ip}: {e}")
    
    def on_tracking_toggled(self, camera_ip: str, enabled: bool):
        """
        Manejar cambio de estado de seguimiento
        
        Args:
            camera_ip: IP de la cámara
            enabled: Nuevo estado de seguimiento
        """
        try:
            if enabled:
                success = self.tracking_manager.start_tracking_for_camera(camera_ip)
                if success:
                    logger.info(f"Seguimiento habilitado para {camera_ip}")
                else:
                    logger.error(f"Error habilitando seguimiento para {camera_ip}")
            else:
                self.tracking_manager.stop_tracking_for_camera(camera_ip)
                logger.info(f"Seguimiento deshabilitado para {camera_ip}")
                
        except Exception as e:
            logger.error(f"Error cambiando estado de seguimiento para {camera_ip}: {e}")
    
    def shutdown(self):
        """Cerrar sistema PTZ limpiamente"""
        try:
            logger.info("Cerrando sistema PTZ...")
            
            # Cerrar todas las ventanas de configuración
            for widget in list(self.config_widgets.values()):
                widget.close()
            
            # Detener todos los seguimientos
            self.stop_all_tracking()
            
            # Cerrar gestor de seguimiento
            self.tracking_manager.shutdown()
            
            # Guardar configuración final
            self.save_configuration()
            
            logger.info("Sistema PTZ cerrado correctamente")
            
        except Exception as e:
            logger.error(f"Error cerrando sistema PTZ: {e}")

# =============================================================================
# FUNCIÓN DE INTEGRACIÓN CON SISTEMA PRINCIPAL
# =============================================================================

def integrate_ptz_with_main_system(main_app, existing_cameras: List[Dict] = None):
    """
    Integrar sistema PTZ con la aplicación principal
    
    Args:
        main_app: Instancia de la aplicación principal
        existing_cameras: Lista de cámaras existentes para verificar cuáles son PTZ
        
    Returns:
        PTZSystemIntegration: Instancia del sistema integrado
    """
    try:
        # Crear instancia de integración
        ptz_integration = PTZSystemIntegration(main_app)
        
        # Si hay cámaras existentes, identificar cuáles son PTZ
        if existing_cameras:
            ptz_cameras = [cam for cam in existing_cameras if cam.get("tipo") == "ptz"]
            
            for cam in ptz_cameras:
                # Agregar configuración PTZ básica
                ptz_config = {
                    "ip": cam["ip"],
                    "port": cam.get("puerto", 80),
                    "username": cam["usuario"],
                    "password": cam["contrasena"],
                    "tracking_mode": "tracking",
                    "tracking_enabled": False  # Deshabilitado por defecto
                }
                
                ptz_integration.add_ptz_camera(ptz_config)
        
        # Conectar con sistema de detección si existe
        if hasattr(main_app, 'detection_system'):
            main_app.detection_system.register_callback(
                ptz_integration.send_detection_to_trackers
            )
        
        logger.info("Sistema PTZ integrado exitosamente")
        return ptz_integration
        
    except Exception as e:
        logger.error(f"Error integrando sistema PTZ: {e}")
        return None

# =============================================================================
# CLASE HELPER PARA INTERFACE PRINCIPAL
# =============================================================================

class PTZControlInterface:
    """Interface simplificada para controlar PTZ desde la GUI principal"""
    
    def __init__(self, ptz_integration: PTZSystemIntegration):
        self.ptz_integration = ptz_integration
    
    def add_ptz_buttons_to_camera(self, camera_widget, camera_ip: str):
        """
        Agregar botones PTZ a widget de cámara
        
        Args:
            camera_widget: Widget de cámara donde agregar botones
            camera_ip: IP de la cámara
        """
        try:
            from PyQt6.QtWidgets import QPushButton, QHBoxLayout
            
            # Verificar si es cámara PTZ
            if camera_ip not in self.ptz_integration.tracking_manager.camera_configs:
                return
            
            # Crear layout para botones PTZ
            ptz_layout = QHBoxLayout()
            
            # Botón de configuración PTZ
            config_btn = QPushButton("⚙️ PTZ Config")
            config_btn.clicked.connect(
                lambda: self.ptz_integration.open_ptz_config_window(camera_ip)
            )
            
            # Botón de seguimiento
            tracking_btn = QPushButton("🎯 Tracking")
            tracking_btn.setCheckable(True)
            tracking_btn.clicked.connect(
                lambda checked: self.toggle_tracking(camera_ip, checked)
            )
            
            # Botón de parada de emergencia
            stop_btn = QPushButton("🛑 STOP")
            stop_btn.clicked.connect(
                lambda: self.emergency_stop_camera(camera_ip)
            )
            stop_btn.setStyleSheet("background-color: red; color: white;")
            
            ptz_layout.addWidget(config_btn)
            ptz_layout.addWidget(tracking_btn)
            ptz_layout.addWidget(stop_btn)
            
            # Agregar al widget de cámara
            if hasattr(camera_widget, 'layout'):
                camera_widget.layout().addLayout(ptz_layout)
            
        except Exception as e:
            logger.error(f"Error agregando botones PTZ: {e}")
    
    def toggle_tracking(self, camera_ip: str, enabled: bool):
        """Toggle seguimiento para una cámara"""
        try:
            self.ptz_integration.on_tracking_toggled(camera_ip, enabled)
        except Exception as e:
            logger.error(f"Error toggle tracking: {e}")
    
    def emergency_stop_camera(self, camera_ip: str):
        """Parada de emergencia para una cámara específica"""
        try:
            config = self.ptz_integration.tracking_manager.camera_configs.get(camera_ip)
            if config:
                light_api = LightAPI(config.ip, config.port, config.username, config.password)
                light_api.stop_movement()
                logger.warning(f"STOP de emergencia enviado a {camera_ip}")
        except Exception as e:
            logger.error(f"Error en STOP de emergencia: {e}")

# Instancia global para fácil acceso
global_ptz_integration: Optional[PTZSystemIntegration] = None

def get_ptz_integration() -> Optional[PTZSystemIntegration]:
    """Obtener instancia global del sistema PTZ"""
    return global_ptz_integration

def set_ptz_integration(integration: PTZSystemIntegration):
    """Establecer instancia global del sistema PTZ"""
    global global_ptz_integration
    global_ptz_integration = integration