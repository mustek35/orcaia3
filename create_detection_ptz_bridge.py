#!/usr/bin/env python3
"""
Crear el archivo detection_ptz_bridge.py que falta
"""

import os

def create_detection_ptz_bridge():
    """Crear archivo detection_ptz_bridge.py"""
    print("📁 Creando core/detection_ptz_bridge.py...")
    
    content = '''"""
Puente de Integración entre Sistema de Detección y PTZ
Conecta las detecciones de YOLO con el sistema de seguimiento PTZ
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np

logger = logging.getLogger(__name__)

class DetectionPTZBridge:
    """
    Puente entre el sistema de detección existente y el sistema PTZ
    """
    
    def __init__(self, grid_rows: int = 12, grid_cols: int = 16):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        
        # Importar grid_utils dinámicamente para evitar errores circulares
        try:
            from .grid_utils import get_default_grid_utils
            self.grid_utils = get_default_grid_utils()
        except ImportError:
            self.grid_utils = None
            logger.warning("GridUtils no disponible")
        
        # Cache para tracking de objetos
        self.object_tracker = {}
        self.last_detection_time = {}
        
        # Configuración
        self.min_confidence_for_tracking = 0.3
        self.object_timeout = 5.0  # segundos
        
    def process_detections_for_ptz(self, detections: List, frame: np.ndarray, 
                                  camera_ip: str, modelo: str = "unknown") -> bool:
        """
        Procesar detecciones del sistema YOLO para envío al sistema PTZ
        
        Args:
            detections: Lista de detecciones [(x, y, w, h, confidence, class_id), ...]
            frame: Frame actual de la cámara
            camera_ip: IP de la cámara que generó las detecciones
            modelo: Tipo de modelo de detección usado
            
        Returns:
            bool: True si se procesaron correctamente
        """
        try:
            if not detections:
                return True
            
            frame_h, frame_w = frame.shape[:2] if frame is not None else (480, 640)
            current_time = time.time()
            
            # Inicializar grilla si es necesario
            if self.grid_utils:
                self.grid_utils.initialize_grid(frame_w, frame_h)
            
            # Procesar cada detección
            for i, detection in enumerate(detections):
                try:
                    # Extraer datos de detección
                    if len(detection) >= 6:
                        x, y, w, h, confidence, class_id = detection[:6]
                    elif len(detection) >= 5:
                        x, y, w, h, confidence = detection[:5]
                        class_id = 0  # Default
                    else:
                        continue
                    
                    # Filtrar por confianza
                    if confidence < self.min_confidence_for_tracking:
                        continue
                    
                    # Convertir coordenadas si es necesario
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    
                    # Calcular centro
                    center_x = x + w // 2
                    center_y = y + h // 2
                    
                    # Calcular celda de grilla
                    if self.grid_utils:
                        grid_cell = self.grid_utils.pixel_to_grid_cell(
                            center_x, center_y, frame_w, frame_h
                        )
                    else:
                        # Cálculo básico si no hay grid_utils
                        row = min(int(center_y * self.grid_rows / frame_h), self.grid_rows - 1)
                        col = min(int(center_x * self.grid_cols / frame_w), self.grid_cols - 1)
                        grid_cell = (row, col)
                    
                    # Determinar tipo de objeto
                    object_type = self.get_object_type_from_class(class_id, modelo)
                    
                    # Crear ID único para el objeto
                    object_id = self.generate_object_id(
                        camera_ip, center_x, center_y, current_time
                    )
                    
                    # Crear datos de detección estándar
                    detection_data = {
                        "object_id": object_id,
                        "object_type": object_type,
                        "confidence": float(confidence),
                        "bbox": (x, y, w, h),
                        "center": (center_x, center_y),
                        "grid_cell": grid_cell,
                        "timestamp": current_time,
                        "frame_width": frame_w,
                        "frame_height": frame_h,
                        "model_used": modelo,
                        "class_id": int(class_id)
                    }
                    
                    # Enviar al sistema PTZ
                    self.send_to_ptz_system(detection_data, camera_ip)
                    
                    # Actualizar tracking
                    self.update_object_tracking(object_id, detection_data, current_time)
                    
                    # Actualizar estadísticas de grilla
                    if self.grid_utils:
                        self.grid_utils.increment_detection_count(grid_cell[0], grid_cell[1])
                    
                except Exception as e:
                    logger.error(f"Error procesando detección individual: {e}")
                    continue
            
            # Limpiar objetos antiguos
            self.cleanup_old_objects(current_time)
            
            return True
            
        except Exception as e:
            logger.error(f"Error procesando detecciones para PTZ: {e}")
            return False
    
    def send_to_ptz_system(self, detection_data: Dict, source_camera_ip: str):
        """
        Enviar detección al sistema PTZ
        
        Args:
            detection_data: Datos de la detección
            source_camera_ip: IP de la cámara fuente
        """
        try:
            # Importar dinámicamente para evitar errores circulares
            from .ptz_integration import get_ptz_integration
            
            ptz_integration = get_ptz_integration()
            if ptz_integration:
                ptz_integration.send_detection_to_tracker(detection_data, source_camera_ip)
            
        except ImportError:
            logger.debug("Sistema PTZ no disponible para envío de detección")
        except Exception as e:
            logger.error(f"Error enviando detección al sistema PTZ: {e}")
    
    def get_object_type_from_class(self, class_id: int, modelo: str) -> str:
        """
        Convertir class_id a tipo de objeto legible
        
        Args:
            class_id: ID de clase del modelo
            modelo: Nombre del modelo usado
            
        Returns:
            str: Tipo de objeto
        """
        try:
            # Mapeos por modelo
            class_mappings = {
                "Personas": {0: "person"},
                "Autos": {0: "car", 1: "truck", 2: "bus", 3: "motorcycle"},
                "Barcos": {0: "boat", 1: "ship"},
                "Embarcaciones": {0: "boat", 1: "ship"},
                "yolov8": {  # COCO classes
                    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 
                    4: "airplane", 5: "bus", 6: "train", 7: "truck", 8: "boat"
                }
            }
            
            mapping = class_mappings.get(modelo, {})
            return mapping.get(class_id, f"class_{class_id}")
            
        except Exception:
            return "unknown"
    
    def generate_object_id(self, camera_ip: str, center_x: int, center_y: int, 
                          timestamp: float) -> str:
        """
        Generar ID único para un objeto detectado
        
        Args:
            camera_ip: IP de la cámara
            center_x: Coordenada X del centro
            center_y: Coordenada Y del centro
            timestamp: Timestamp de la detección
            
        Returns:
            str: ID único del objeto
        """
        try:
            # Crear ID basado en posición y tiempo
            clean_ip = camera_ip.replace(".", "_")
            time_id = int(timestamp * 1000) % 100000  # últimos 5 dígitos
            
            return f"obj_{clean_ip}_{center_x}_{center_y}_{time_id}"
            
        except Exception:
            return f"obj_{int(timestamp * 1000)}"
    
    def update_object_tracking(self, object_id: str, detection_data: Dict, timestamp: float):
        """
        Actualizar tracking de objetos
        
        Args:
            object_id: ID del objeto
            detection_data: Datos de la detección
            timestamp: Timestamp actual
        """
        try:
            self.object_tracker[object_id] = {
                "last_seen": timestamp,
                "position": detection_data["center"],
                "grid_cell": detection_data["grid_cell"],
                "confidence": detection_data["confidence"],
                "object_type": detection_data["object_type"]
            }
            
            self.last_detection_time[object_id] = timestamp
            
        except Exception as e:
            logger.error(f"Error actualizando tracking: {e}")
    
    def cleanup_old_objects(self, current_time: float):
        """
        Limpiar objetos que no se han visto recientemente
        
        Args:
            current_time: Timestamp actual
        """
        try:
            expired_objects = []
            
            for object_id, last_time in self.last_detection_time.items():
                if current_time - last_time > self.object_timeout:
                    expired_objects.append(object_id)
            
            for object_id in expired_objects:
                self.object_tracker.pop(object_id, None)
                self.last_detection_time.pop(object_id, None)
            
            if expired_objects:
                logger.debug(f"Limpiados {len(expired_objects)} objetos expirados")
                
        except Exception as e:
            logger.error(f"Error limpiando objetos antiguos: {e}")
    
    def get_active_objects(self) -> Dict:
        """
        Obtener lista de objetos actualmente trackeados
        
        Returns:
            Dict: Objetos activos
        """
        return self.object_tracker.copy()
    
    def process_detection_from_gui(self, detection_data: Dict, camera_ip: str):
        """
        Procesar detección que viene desde la GUI
        
        Args:
            detection_data: Datos de detección desde GUI
            camera_ip: IP de la cámara
        """
        try:
            # Enviar directamente al sistema PTZ
            self.send_to_ptz_system(detection_data, camera_ip)
            
        except Exception as e:
            logger.error(f"Error procesando detección desde GUI: {e}")

# Instancia global del puente
detection_ptz_bridge = DetectionPTZBridge()

def integrate_detection_with_ptz(detection_function: Callable) -> Callable:
    """
    Decorador para integrar función de detección existente con PTZ
    
    Args:
        detection_function: Función de detección existente
        
    Returns:
        Callable: Función decorada con integración PTZ
    """
    def wrapper(*args, **kwargs):
        try:
            # Ejecutar función de detección original
            result = detection_function(*args, **kwargs)
            
            # Intentar extraer datos para PTZ
            if len(args) >= 3:  # Asumiendo (frame, detections, camera_info)
                frame = args[0]
                detections = args[1] if len(args) > 1 else []
                camera_info = args[2] if len(args) > 2 else {}
                
                # Extraer IP de cámara
                camera_ip = camera_info.get("ip", "unknown")
                modelo = camera_info.get("modelo", "unknown")
                
                # Procesar para PTZ
                detection_ptz_bridge.process_detections_for_ptz(
                    detections, frame, camera_ip, modelo
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error en wrapper de detección PTZ: {e}")
            return detection_function(*args, **kwargs)
    
    return wrapper

def patch_gestor_alertas_for_ptz():
    """
    Parchear GestorAlertas existente para enviar detecciones al sistema PTZ
    """
    try:
        from core.gestor_alertas import GestorAlertas
        
        # Guardar método original
        original_procesar = GestorAlertas._procesar_y_guardar
        
        def enhanced_procesar(self, boxes, frame, log_callback, tipo, cam_data):
            try:
                # Ejecutar procesamiento original
                result = original_procesar(self, boxes, frame, log_callback, tipo, cam_data)
                
                # Enviar al sistema PTZ
                camera_ip = cam_data.get("ip", "unknown")
                modelo = cam_data.get("modelo", "unknown")
                
                # Convertir boxes al formato esperado
                detections = []
                for box in boxes:
                    if len(box) >= 5:
                        x, y, w, h, class_id = box[:5]
                        confidence = 0.8  # Default si no está disponible
                        detections.append((x, y, w, h, confidence, class_id))
                
                # Procesar para PTZ
                detection_ptz_bridge.process_detections_for_ptz(
                    detections, frame, camera_ip, modelo
                )
                
                return result
                
            except Exception as e:
                logger.error(f"Error en enhanced_procesar: {e}")
                return original_procesar(self, boxes, frame, log_callback, tipo, cam_data)
        
        # Reemplazar método
        GestorAlertas._procesar_y_guardar = enhanced_procesar
        
        logger.info("✅ GestorAlertas parcheado para integración PTZ")
        return True
        
    except Exception as e:
        logger.error(f"Error parcheando GestorAlertas: {e}")
        return False

def setup_ptz_integration_hooks():
    """
    Configurar todos los hooks de integración PTZ
    """
    logger.info("🔗 Configurando hooks de integración PTZ...")
    
    success_count = 0
    
    # Parchear GestorAlertas
    if patch_gestor_alertas_for_ptz():
        success_count += 1
    
    logger.info(f"✅ Integración PTZ configurada: {success_count}/1 componentes")
    
    return success_count > 0

def auto_initialize_ptz_integration():
    """
    Inicialización automática de la integración PTZ
    """
    try:
        logger.info("🚀 Inicialización automática de integración PTZ...")
        
        # Configurar hooks
        setup_ptz_integration_hooks()
        
        # Inicializar grid por defecto
        if detection_ptz_bridge.grid_utils:
            detection_ptz_bridge.grid_utils.initialize_grid(640, 480)
        
        logger.info("✅ Integración PTZ inicializada automáticamente")
        return True
        
    except Exception as e:
        logger.error(f"Error en inicialización automática PTZ: {e}")
        return False

# Ejecutar inicialización automática al importar
if __name__ != "__main__":
    auto_initialize_ptz_integration()
'''
    
    try:
        # Crear directorio si no existe
        os.makedirs("core", exist_ok=True)
        
        # Escribir archivo
        with open("core/detection_ptz_bridge.py", 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ core/detection_ptz_bridge.py creado exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error creando archivo: {e}")
        return False

def update_ptz_integration():
    """Actualizar ptz_integration.py para incluir get_ptz_integration"""
    print("🔧 Actualizando core/ptz_integration.py...")
    
    try:
        file_path = "core/ptz_integration.py"
        
        if not os.path.exists(file_path):
            print(f"⚠️ Archivo {file_path} no encontrado")
            return False
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Agregar función get_ptz_integration si no existe
        if 'def get_ptz_integration()' not in content:
            addition = '''

# Instancia global para fácil acceso
global_ptz_integration: Optional[PTZSystemIntegration] = None

def get_ptz_integration() -> Optional[PTZSystemIntegration]:
    """Obtener instancia global del sistema PTZ"""
    return global_ptz_integration

def set_ptz_integration(integration: PTZSystemIntegration):
    """Establecer instancia global del sistema PTZ"""
    global global_ptz_integration
    global_ptz_integration = integration
'''
            
            content += addition
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Función get_ptz_integration agregada")
            return True
        else:
            print("ℹ️ Función get_ptz_integration ya existe")
            return True
            
    except Exception as e:
        print(f"❌ Error actualizando ptz_integration.py: {e}")
        return False

def test_all_imports():
    """Probar todos los imports después de crear el archivo"""
    print("\n🧪 Probando todos los imports...")
    
    import sys
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    tests = [
        ("core.detection_ptz_bridge", "DetectionPTZBridge"),
        ("core.ptz_integration", "PTZSystemIntegration"),
        ("core.light_api", "LightAPI"),
        ("gui.ptz_config_widget", "PTZConfigWidget"),
        ("core.grid_utils", "GridUtils")
    ]
    
    success_count = 0
    for module_name, class_name in tests:
        try:
            # Limpiar cache si existe
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✅ {module_name}.{class_name}")
            success_count += 1
        except Exception as e:
            print(f"  ❌ {module_name}.{class_name}: {e}")
    
    print(f"\n📊 Resultado: {success_count}/{len(tests)} imports exitosos")
    return success_count == len(tests)

def main():
    """Función principal"""
    print("📁 CREANDO ARCHIVO FALTANTE: detection_ptz_bridge.py")
    print("=" * 55)
    
    # Paso 1: Crear archivo faltante
    if create_detection_ptz_bridge():
        print("✅ Archivo detection_ptz_bridge.py creado")
    else:
        print("❌ Error creando archivo")
        return False
    
    # Paso 2: Actualizar ptz_integration.py
    if update_ptz_integration():
        print("✅ ptz_integration.py actualizado")
    else:
        print("⚠️ ptz_integration.py no se pudo actualizar")
    
    # Paso 3: Probar todos los imports
    if test_all_imports():
        print("\n🎉 ¡TODOS LOS IMPORTS FUNCIONAN!")
        print("\n🚀 Ahora ejecuta:")
        print("  python scripts/ptz_diagnostics.py")
        print("  python run_ptz_tracker.py")
        print("  python main_window.py")
        return True
    else:
        print("\n⚠️ Algunos imports aún fallan")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)