"""
LightAPI - Interfaz completa para cámaras PTZ
Implementa todas las funcionalidades de control PTZ usando la API Light
"""

import requests
import json
import base64
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from requests.auth import HTTPDigestAuth
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class PTZDirection(Enum):
    """Direcciones de movimiento PTZ"""
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    UP_LEFT = "up_left"
    UP_RIGHT = "up_right"
    DOWN_LEFT = "down_left"
    DOWN_RIGHT = "down_right"
    STOP = "stop"

class ZoomDirection(Enum):
    """Direcciones de zoom"""
    IN = "in"
    OUT = "out"
    STOP = "stop"

@dataclass
class PTZPosition:
    """Posición PTZ"""
    pan: float
    tilt: float
    zoom: float

@dataclass
class PresetInfo:
    """Información de preset"""
    id: int
    name: str
    position: Optional[PTZPosition] = None

class LightAPI:
    """Clase principal para controlar cámaras PTZ mediante Light API"""
    
    def __init__(self, ip: str, port: int = 80, username: str = "admin", password: str = ""):
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        self.base_url = f"http://{ip}:{port}/LAPI/V1.0"
        self.channel_id = 0  # Para IPC siempre es 0
        self.session = requests.Session()
        self.session.auth = HTTPDigestAuth(username, password)
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Cache para información del dispositivo
        self._device_info = None
        self._presets_cache = None
        self._last_preset_update = 0
    
    def test_connection(self) -> bool:
        """Probar conexión con la cámara"""
        try:
            response = self._make_request("GET", "System/DeviceInfo")
            return response is not None
        except Exception as e:
            logger.error(f"Error probando conexión con {self.ip}: {e}")
            return False
    
    def get_device_info(self) -> Optional[Dict[str, Any]]:
        """Obtener información del dispositivo"""
        if not self._device_info:
            response = self._make_request("GET", "System/DeviceInfo")
            if response:
                self._device_info = response.get("Response", {}).get("Data", {})
        
        return self._device_info
    
    # =============================================================================
    # CONTROL DE MOVIMIENTO PTZ
    # =============================================================================
    
    def continuous_move(self, pan_speed: float, tilt_speed: float, zoom_speed: float = 0.0) -> bool:
        """
        Movimiento continuo de la cámara PTZ
        
        Args:
            pan_speed: Velocidad de paneo (-1.0 a 1.0)
            tilt_speed: Velocidad de inclinación (-1.0 a 1.0)  
            zoom_speed: Velocidad de zoom (-1.0 a 1.0)
            
        Returns:
            bool: True si el comando fue exitoso
        """
        try:
            # Limitar velocidades al rango válido
            pan_speed = max(-1.0, min(1.0, pan_speed))
            tilt_speed = max(-1.0, min(1.0, tilt_speed))
            zoom_speed = max(-1.0, min(1.0, zoom_speed))
            
            data = {
                "Command": "ContinuousMove",
                "PanTiltSpeed": {
                    "x": float(pan_speed),
                    "y": float(tilt_speed)
                },
                "ZoomSpeed": float(zoom_speed)
            }
            
            response = self._make_request("PUT", f"Channels/{self.channel_id}/PTZ/PTZCtrl", data)
            return response is not None
            
        except Exception as e:
            logger.error(f"Error en movimiento continuo: {e}")
            return False
    
    def move_direction(self, direction: PTZDirection, speed: float = 0.5, duration: float = 0.0) -> bool:
        """
        Mover la cámara en una dirección específica
        
        Args:
            direction: Dirección del movimiento
            speed: Velocidad del movimiento (0.0 a 1.0)
            duration: Duración del movimiento en segundos (0 = continuo)
            
        Returns:
            bool: True si el comando fue exitoso
        """
        try:
            speed = max(0.0, min(1.0, speed))
            
            # Mapear direcciones a velocidades
            direction_map = {
                PTZDirection.UP: (0.0, speed),
                PTZDirection.DOWN: (0.0, -speed),
                PTZDirection.LEFT: (-speed, 0.0),
                PTZDirection.RIGHT: (speed, 0.0),
                PTZDirection.UP_LEFT: (-speed, speed),
                PTZDirection.UP_RIGHT: (speed, speed),
                PTZDirection.DOWN_LEFT: (-speed, -speed),
                PTZDirection.DOWN_RIGHT: (speed, -speed),
                PTZDirection.STOP: (0.0, 0.0)
            }
            
            if direction not in direction_map:
                logger.error(f"Dirección no válida: {direction}")
                return False
            
            pan_speed, tilt_speed = direction_map[direction]
            
            # Ejecutar movimiento
            success = self.continuous_move(pan_speed, tilt_speed, 0.0)
            
            # Si se especifica duración, programar parada
            if success and duration > 0:
                def stop_after_duration():
                    time.sleep(duration)
                    self.stop_movement()
                
                import threading
                threading.Thread(target=stop_after_duration, daemon=True).start()
            
            return success
            
        except Exception as e:
            logger.error(f"Error moviendo en dirección {direction}: {e}")
            return False
    
    def stop_movement(self) -> bool:
        """Detener todo movimiento PTZ"""
        try:
            data = {
                "Command": "Stop",
                "StopPanTilt": True,
                "StopZoom": True
            }
            
            response = self._make_request("PUT", f"Channels/{self.channel_id}/PTZ/PTZCtrl", data)
            return response is not None
            
        except Exception as e:
            logger.error(f"Error deteniendo movimiento: {e}")
            return False
    
    # =============================================================================
    # CONTROL DE ZOOM
    # =============================================================================
    
    def zoom_continuous(self, zoom_speed: float) -> bool:
        """
        Zoom continuo
        
        Args:
            zoom_speed: Velocidad de zoom (-1.0 a 1.0, positivo = acercar)
            
        Returns:
            bool: True si el comando fue exitoso
        """
        return self.continuous_move(0.0, 0.0, zoom_speed)
    
    def zoom_in(self, speed: float = 0.5) -> bool:
        """Acercar zoom"""
        return self.zoom_continuous(abs(speed))
    
    def zoom_out(self, speed: float = 0.5) -> bool:
        """Alejar zoom"""
        return self.zoom_continuous(-abs(speed))
    
    def zoom_stop(self) -> bool:
        """Detener zoom"""
        return self.zoom_continuous(0.0)
    
    # =============================================================================
    # PRESETS
    # =============================================================================
    
    def get_presets(self, force_refresh: bool = False) -> List[PresetInfo]:
        """
        Obtener lista de presets disponibles
        
        Args:
            force_refresh: Forzar actualización desde la cámara
            
        Returns:
            List[PresetInfo]: Lista de presets disponibles
        """
        current_time = time.time()
        
        # Usar cache si no ha pasado mucho tiempo
        if (not force_refresh and 
            self._presets_cache and 
            (current_time - self._last_preset_update) < 30):
            return self._presets_cache
        
        try:
            response = self._make_request("GET", f"Channels/{self.channel_id}/PTZ/Presets")
            
            if not response:
                return []
            
            presets_data = response.get("Response", {}).get("Data", {}).get("PresetList", [])
            presets = []
            
            for preset_data in presets_data:
                preset = PresetInfo(
                    id=preset_data.get("ID", 0),
                    name=preset_data.get("Name", f"Preset {preset_data.get('ID', 0)}")
                )
                presets.append(preset)
            
            self._presets_cache = presets
            self._last_preset_update = current_time
            
            return presets
            
        except Exception as e:
            logger.error(f"Error obteniendo presets: {e}")
            return []
    
    def goto_preset(self, preset_id: int) -> bool:
        """
        Ir a un preset específico
        
        Args:
            preset_id: ID del preset
            
        Returns:
            bool: True si el comando fue exitoso
        """
        try:
            data = {
                "Command": "GotoPreset",
                "PresetID": preset_id
            }
            
            response = self._make_request("PUT", f"Channels/{self.channel_id}/PTZ/PTZCtrl", data)
            return response is not None
            
        except Exception as e:
            logger.error(f"Error yendo al preset {preset_id}: {e}")
            return False
    
    def create_preset(self, preset_id: int, preset_name: str) -> bool:
        """
        Crear un nuevo preset en la posición actual
        
        Args:
            preset_id: ID del nuevo preset
            preset_name: Nombre del preset
            
        Returns:
            bool: True si el preset fue creado exitosamente
        """
        try:
            data = {
                "PresetList": [{
                    "ID": preset_id,
                    "Name": preset_name
                }]
            }
            
            response = self._make_request("PUT", f"Channels/{self.channel_id}/PTZ/Presets", data)
            
            if response:
                # Limpiar cache para forzar actualización
                self._presets_cache = None
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error creando preset {preset_id}: {e}")
            return False
    
    def delete_preset(self, preset_id: int) -> bool:
        """
        Eliminar un preset
        
        Args:
            preset_id: ID del preset a eliminar
            
        Returns:
            bool: True si el preset fue eliminado exitosamente
        """
        try:
            response = self._make_request("DELETE", f"Channels/{self.channel_id}/PTZ/Presets/{preset_id}")
            
            if response:
                # Limpiar cache para forzar actualización
                self._presets_cache = None
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error eliminando preset {preset_id}: {e}")
            return False
    
    # =============================================================================
    # CAPTURA DE IMÁGENES
    # =============================================================================
    
    def capture_snapshot(self, save_path: Optional[str] = None) -> Optional[bytes]:
        """
        Capturar una imagen de la cámara
        
        Args:
            save_path: Ruta donde guardar la imagen (opcional)
            
        Returns:
            bytes: Datos de la imagen o None si hay error
        """
        try:
            # Usar endpoint específico para snapshot
            url = f"{self.base_url}/Media/Snapshot"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                image_data = response.content
                
                # Guardar si se especifica ruta
                if save_path:
                    with open(save_path, 'wb') as f:
                        f.write(image_data)
                    logger.info(f"Imagen guardada en: {save_path}")
                
                return image_data
            else:
                logger.error(f"Error capturando imagen: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error en captura de imagen: {e}")
            return None
    
    # =============================================================================
    # FUNCIONES ESPECIALES
    # =============================================================================
    
    def enable_wiper(self) -> bool:
        """Activar limpiaparabrisas"""
        try:
            data = {
                "Command": "WiperOn"
            }
            
            response = self._make_request("PUT", f"Channels/{self.channel_id}/PTZ/PTZCtrl", data)
            return response is not None
            
        except Exception as e:
            logger.error(f"Error activando wiper: {e}")
            return False
    
    def set_defog_mode(self, enable: bool) -> bool:
        """
        Activar/desactivar modo defog (antiempañamiento)
        
        Args:
            enable: True para activar, False para desactivar
            
        Returns:
            bool: True si el comando fue exitoso
        """
        try:
            data = {
                "Command": "DefogOn" if enable else "DefogOff"
            }
            
            response = self._make_request("PUT", f"Channels/{self.channel_id}/PTZ/PTZCtrl", data)
            return response is not None
            
        except Exception as e:
            logger.error(f"Error {'activando' if enable else 'desactivando'} defog: {e}")
            return False
    
    def get_current_position(self) -> Optional[PTZPosition]:
        """
        Obtener posición actual de la cámara PTZ
        
        Returns:
            PTZPosition: Posición actual o None si hay error
        """
        try:
            response = self._make_request("GET", f"Channels/{self.channel_id}/PTZ/Status")
            
            if response:
                status_data = response.get("Response", {}).get("Data", {})
                position_data = status_data.get("Position", {})
                
                return PTZPosition(
                    pan=position_data.get("Pan", 0.0),
                    tilt=position_data.get("Tilt", 0.0),
                    zoom=position_data.get("Zoom", 0.0)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error obteniendo posición actual: {e}")
            return None
    
    def goto_absolute_position(self, pan: float, tilt: float, zoom: float = None) -> bool:
        """
        Mover a una posición absoluta
        
        Args:
            pan: Posición de paneo (-1.0 a 1.0)
            tilt: Posición de inclinación (-1.0 a 1.0)
            zoom: Posición de zoom (-1.0 a 1.0, opcional)
            
        Returns:
            bool: True si el comando fue exitoso
        """
        try:
            # Limitar valores al rango válido
            pan = max(-1.0, min(1.0, pan))
            tilt = max(-1.0, min(1.0, tilt))
            
            data = {
                "Command": "AbsoluteMove",
                "Position": {
                    "Pan": float(pan),
                    "Tilt": float(tilt)
                }
            }
            
            if zoom is not None:
                zoom = max(-1.0, min(1.0, zoom))
                data["Position"]["Zoom"] = float(zoom)
            
            response = self._make_request("PUT", f"Channels/{self.channel_id}/PTZ/PTZCtrl", data)
            return response is not None
            
        except Exception as e:
            logger.error(f"Error moviendo a posición absoluta: {e}")
            return False
    
    # =============================================================================
    # MÉTODOS AUXILIARES
    # =============================================================================
    
    def _make_request(self, method: str, endpoint: str, data: dict = None) -> Optional[dict]:
        """
        Realizar petición HTTP a la API
        
        Args:
            method: Método HTTP (GET, PUT, POST, DELETE)
            endpoint: Endpoint de la API
            data: Datos a enviar (para PUT/POST)
            
        Returns:
            dict: Respuesta de la API o None si hay error
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            
            if method.upper() == "GET":
                response = self.session.get(url, timeout=10)
            elif method.upper() == "PUT":
                response = self.session.put(url, json=data, timeout=10)
            elif method.upper() == "POST":
                response = self.session.post(url, json=data, timeout=10)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=10)
            else:
                logger.error(f"Método HTTP no soportado: {method}")
                return None
            
            # Verificar código de respuesta
            if response.status_code in [200, 201]:
                # Intentar parsear JSON
                try:
                    return response.json()
                except json.JSONDecodeError:
                    # Respuesta exitosa pero sin JSON válido
                    return {"status": "success"}
            else:
                logger.error(f"Error HTTP {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout en petición a {endpoint}")
            return None
        except requests.exceptions.ConnectionError:
            logger.error(f"Error de conexión a {self.ip}")
            return None
        except Exception as e:
            logger.error(f"Error en petición a {endpoint}: {e}")
            return None
    
    def get_api_info(self) -> Dict[str, Any]:
        """Obtener información de la API y capacidades de la cámara"""
        info = {
            "ip": self.ip,
            "port": self.port,
            "username": self.username,
            "connected": self.test_connection(),
            "device_info": self.get_device_info(),
            "presets_count": len(self.get_presets()),
            "current_position": self.get_current_position()
        }
        
        return info
    
    def __str__(self) -> str:
        """Representación en string de la instancia"""
        return f"LightAPI({self.ip}:{self.port}, user={self.username})"
    
    def __repr__(self) -> str:
        """Representación para debug"""
        return self.__str__()