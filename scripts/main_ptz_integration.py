"""
Sistema Principal de Integración PTZ
Punto de entrada principal para el sistema de seguimiento PTZ profesional
"""

import sys
import os
import logging
import argparse
import signal
import atexit
from typing import Optional, Dict, List
from PyQt6.QtWidgets import QApplication, QMessageBox, QSystemTrayIcon, QMenu
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QIcon, QAction

# Agregar el directorio actual al path para imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from core.ptz_integration import PTZSystemIntegration, PTZControlInterface, set_ptz_integration
from core.light_api import LightAPI
from gui.ptz_config_widget import PTZConfigWidget
from core.grid_utils import GridUtils

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ptz_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class PTZSystemManager(QObject):
    """
    Gestor principal del sistema PTZ
    Coordina todos los componentes y maneja el ciclo de vida del sistema
    """
    
    # Señales
    system_status_changed = pyqtSignal(str)
    camera_status_changed = pyqtSignal(str, dict)
    emergency_stop_triggered = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        
        # Componentes principales
        self.ptz_integration: Optional[PTZSystemIntegration] = None
        self.control_interface: Optional[PTZControlInterface] = None
        self.grid_utils = GridUtils()
        
        # Estado del sistema
        self.is_running = False
        self.main_app = None
        self.config_widgets: Dict[str, PTZConfigWidget] = {}
        
        # Timer para monitoreo
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_system_status)
        
        # Configurar manejo de señales del sistema
        self.setup_signal_handlers()
        
    def initialize(self, main_app=None, config_file: str = "ptz_tracking_config.json") -> bool:
        """
        Inicializar el sistema PTZ
        
        Args:
            main_app: Aplicación principal (opcional)
            config_file: Archivo de configuración
            
        Returns:
            bool: True si se inicializó correctamente
        """
        try:
            logger.info("🚀 Inicializando Sistema PTZ Profesional...")
            
            self.main_app = main_app
            
            # Crear integración PTZ
            self.ptz_integration = PTZSystemIntegration(main_app)
            
            # Cargar configuración
            if os.path.exists(config_file):
                self.ptz_integration.load_configuration(config_file)
                logger.info(f"✅ Configuración cargada desde {config_file}")
            else:
                logger.warning(f"⚠️ Archivo de configuración {config_file} no encontrado, usando configuración por defecto")
                self.ptz_integration.create_default_config(config_file)
            
            # Crear interfaz de control
            self.control_interface = PTZControlInterface(self.ptz_integration)
            
            # Establecer integración global
            set_ptz_integration(self.ptz_integration)
            
            # Conectar señales
            self.connect_signals()
            
            # Inicializar grid utils
            self.grid_utils.initialize_grid(640, 480)  # Resolución por defecto
            
            # Iniciar monitoreo
            self.status_timer.start(5000)  # Cada 5 segundos
            
            self.is_running = True
            self.system_status_changed.emit("Sistema PTZ inicializado correctamente")
            
            logger.info("✅ Sistema PTZ inicializado exitosamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando sistema PTZ: {e}")
            return False
    
    def setup_signal_handlers(self):
        """Configurar manejadores de señales del sistema"""
        try:
            # Manejo de SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, self.signal_handler)
            
            # Manejo de SIGTERM
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Registrar función de limpieza al salir
            atexit.register(self.cleanup)
            
        except Exception as e:
            logger.error(f"Error configurando manejadores de señales: {e}")
    
    def signal_handler(self, signum, frame):
        """Manejar señales del sistema"""
        logger.info(f"Señal {signum} recibida, cerrando sistema...")
        self.shutdown()
    
    def connect_signals(self):
        """Conectar señales internas del sistema"""
        try:
            if self.ptz_integration:
                # Registrar callback para detecciones
                self.ptz_integration.register_detection_callback(self.on_detection_received)
                
        except Exception as e:
            logger.error(f"Error conectando señales: {e}")
    
    def start_tracking_system(self) -> bool:
        """
        Iniciar sistema de seguimiento para todas las cámaras habilitadas
        
        Returns:
            bool: True si se inició correctamente
        """
        try:
            if not self.ptz_integration:
                logger.error("Sistema PTZ no inicializado")
                return False
            
            logger.info("🎯 Iniciando sistema de seguimiento...")
            
            results = self.ptz_integration.start_all_tracking()
            
            success_count = sum(1 for success in results.values() if success)
            total_count = len(results)
            
            if success_count > 0:
                logger.info(f"✅ Seguimiento iniciado para {success_count}/{total_count} cámaras")
                self.system_status_changed.emit(f"Seguimiento activo en {success_count} cámaras")
                return True
            else:
                logger.warning("⚠️ No se pudo iniciar seguimiento en ninguna cámara")
                self.system_status_changed.emit("Sin cámaras de seguimiento activas")
                return False
                
        except Exception as e:
            logger.error(f"Error iniciando sistema de seguimiento: {e}")
            return False
    
    def stop_tracking_system(self):
        """Detener sistema de seguimiento"""
        try:
            if not self.ptz_integration:
                return
            
            logger.info("⏹️ Deteniendo sistema de seguimiento...")
            
            self.ptz_integration.stop_all_tracking()
            self.system_status_changed.emit("Sistema de seguimiento detenido")
            
        except Exception as e:
            logger.error(f"Error deteniendo sistema de seguimiento: {e}")
    
    def add_ptz_camera_interactive(self) -> bool:
        """
        Agregar cámara PTZ de forma interactiva
        
        Returns:
            bool: True si se agregó correctamente
        """
        try:
            # Crear widget de configuración temporal
            config_widget = PTZConfigWidget()
            
            # Mostrar como modal
            if config_widget.exec() == PTZConfigWidget.DialogCode.Accepted:
                config = config_widget.get_config_from_ui()
                
                if config:
                    camera_config = {
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
                    
                    success = self.ptz_integration.add_ptz_camera(camera_config)
                    
                    if success:
                        logger.info(f"✅ Cámara PTZ {config.ip} agregada exitosamente")
                        self.system_status_changed.emit(f"Cámara {config.ip} agregada")
                        return True
                    else:
                        logger.error(f"❌ Error agregando cámara PTZ {config.ip}")
                        return False
            
            return False
            
        except Exception as e:
            logger.error(f"Error agregando cámara PTZ: {e}")
            return False
    
    def open_camera_config(self, camera_ip: str) -> bool:
        """
        Abrir configuración para una cámara específica
        
        Args:
            camera_ip: IP de la cámara
            
        Returns:
            bool: True si se abrió correctamente
        """
        try:
            if not self.ptz_integration:
                logger.error("Sistema PTZ no inicializado")
                return False
            
            widget = self.ptz_integration.open_ptz_config_window(camera_ip)
            
            if widget:
                self.config_widgets[camera_ip] = widget
                logger.info(f"✅ Configuración abierta para cámara {camera_ip}")
                return True
            else:
                logger.error(f"❌ No se pudo abrir configuración para {camera_ip}")
                return False
                
        except Exception as e:
            logger.error(f"Error abriendo configuración: {e}")
            return False
    
    def emergency_stop(self):
        """Parada de emergencia del sistema"""
        try:
            logger.warning("🚨 PARADA DE EMERGENCIA ACTIVADA")
            
            if self.ptz_integration:
                self.ptz_integration.emergency_stop_all()
            
            self.emergency_stop_triggered.emit()
            self.system_status_changed.emit("PARADA DE EMERGENCIA ACTIVADA")
            
        except Exception as e:
            logger.error(f"Error en parada de emergencia: {e}")
    
    def get_system_status(self) -> Dict:
        """
        Obtener estado completo del sistema
        
        Returns:
            Dict: Estado del sistema
        """
        try:
            if not self.ptz_integration:
                return {"status": "not_initialized"}
            
            tracking_status = self.ptz_integration.get_tracking_status()
            
            status = {
                "system_running": self.is_running,
                "cameras_configured": len(self.ptz_integration.tracking_manager.camera_configs),
                "cameras_tracking": len([s for s in tracking_status.values() if s.get("running", False)]),
                "config_windows_open": len(self.config_widgets),
                "tracking_details": tracking_status,
                "grid_status": {
                    "dimensions": f"{self.grid_utils.grid_rows}x{self.grid_utils.grid_cols}",
                    "active_cells": len(self.grid_utils.get_active_cells()),
                    "zones": len(self.grid_utils.zones)
                }
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Error obteniendo estado del sistema: {e}")
            return {"status": "error", "error": str(e)}
    
    def update_system_status(self):
        """Actualizar estado del sistema periódicamente"""
        try:
            status = self.get_system_status()
            
            # Emitir cambios de estado
            for camera_ip, camera_status in status.get("tracking_details", {}).items():
                self.camera_status_changed.emit(camera_ip, camera_status)
            
        except Exception as e:
            logger.error(f"Error actualizando estado: {e}")
    
    def on_detection_received(self, detection_data: dict, source_camera_ip: str):
        """
        Callback para detecciones recibidas
        
        Args:
            detection_data: Datos de la detección
            source_camera_ip: IP de la cámara fuente
        """
        try:
            # Actualizar contador en grilla
            if "grid_cell" in detection_data:
                row, col = detection_data["grid_cell"]
                self.grid_utils.increment_detection_count(row, col)
            
            # Log de detección (opcional, puede generar mucho log)
            logger.debug(f"Detección recibida desde {source_camera_ip}: {detection_data.get('object_type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Error procesando detección: {e}")
    
    def save_system_configuration(self, filename: str = None) -> bool:
        """
        Guardar configuración completa del sistema
        
        Args:
            filename: Nombre del archivo (opcional)
            
        Returns:
            bool: True si se guardó correctamente
        """
        try:
            if not filename:
                filename = f"ptz_system_backup_{int(time.time())}.json"
            
            if self.ptz_integration:
                self.ptz_integration.save_configuration(filename)
            
            # Guardar configuración de grilla
            grid_filename = filename.replace('.json', '_grid.json')
            self.grid_utils.save_configuration(grid_filename)
            
            logger.info(f"✅ Configuración del sistema guardada en {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
            return False
    
    def load_system_configuration(self, filename: str) -> bool:
        """
        Cargar configuración completa del sistema
        
        Args:
            filename: Nombre del archivo
            
        Returns:
            bool: True si se cargó correctamente
        """
        try:
            if not os.path.exists(filename):
                logger.error(f"Archivo de configuración {filename} no encontrado")
                return False
            
            if self.ptz_integration:
                self.ptz_integration.load_configuration(filename)
            
            # Cargar configuración de grilla
            grid_filename = filename.replace('.json', '_grid.json')
            if os.path.exists(grid_filename):
                self.grid_utils.load_configuration(grid_filename)
            
            logger.info(f"✅ Configuración del sistema cargada desde {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return False
    
    def cleanup(self):
        """Limpieza al cerrar el sistema"""
        try:
            logger.info("🧹 Realizando limpieza del sistema...")
            
            # Cerrar ventanas de configuración
            for widget in list(self.config_widgets.values()):
                try:
                    widget.close()
                except:
                    pass
            
            # Guardar configuración automáticamente
            if self.ptz_integration:
                self.ptz_integration.save_configuration()
            
            # Exportar estadísticas
            self.grid_utils.export_statistics("ptz_session_stats.json")
            
        except Exception as e:
            logger.error(f"Error en limpieza: {e}")
    
    def shutdown(self):
        """Cerrar sistema completamente"""
        try:
            logger.info("🔄 Cerrando Sistema PTZ...")
            
            self.is_running = False
            
            # Detener monitoreo
            if self.status_timer.isActive():
                self.status_timer.stop()
            
            # Detener seguimiento
            self.stop_tracking_system()
            
            # Cerrar integración PTZ
            if self.ptz_integration:
                self.ptz_integration.shutdown()
            
            # Limpieza final
            self.cleanup()
            
            self.system_status_changed.emit("Sistema PTZ cerrado")
            logger.info("✅ Sistema PTZ cerrado correctamente")
            
        except Exception as e:
            logger.error(f"Error cerrando sistema: {e}")

# =============================================================================
# FUNCIONES PRINCIPALES
# =============================================================================

def create_system_tray(ptz_manager: PTZSystemManager) -> QSystemTrayIcon:
    """
    Crear icono de bandeja del sistema
    
    Args:
        ptz_manager: Gestor del sistema PTZ
        
    Returns:
        QSystemTrayIcon: Icono de bandeja
    """
    try:
        # Crear icono de bandeja
        tray_icon = QSystemTrayIcon()
        
        # Crear menú
        tray_menu = QMenu()
        
        # Acciones del menú
        status_action = QAction("Estado del Sistema", tray_menu)
        status_action.triggered.connect(lambda: show_system_status(ptz_manager))
        
        start_action = QAction("Iniciar Seguimiento", tray_menu)
        start_action.triggered.connect(ptz_manager.start_tracking_system)
        
        stop_action = QAction("Detener Seguimiento", tray_menu)
        stop_action.triggered.connect(ptz_manager.stop_tracking_system)
        
        emergency_action = QAction("🚨 PARADA DE EMERGENCIA", tray_menu)
        emergency_action.triggered.connect(ptz_manager.emergency_stop)
        
        separator1 = QAction(tray_menu)
        separator1.setSeparator(True)
        
        config_action = QAction("Agregar Cámara PTZ", tray_menu)
        config_action.triggered.connect(ptz_manager.add_ptz_camera_interactive)
        
        save_action = QAction("Guardar Configuración", tray_menu)
        save_action.triggered.connect(lambda: ptz_manager.save_system_configuration())
        
        separator2 = QAction(tray_menu)
        separator2.setSeparator(True)
        
        quit_action = QAction("Salir", tray_menu)
        quit_action.triggered.connect(ptz_manager.shutdown)
        
        # Agregar acciones al menú
        tray_menu.addAction(status_action)
        tray_menu.addAction(separator1)
        tray_menu.addAction(start_action)
        tray_menu.addAction(stop_action)
        tray_menu.addAction(emergency_action)
        tray_menu.addAction(separator2)
        tray_menu.addAction(config_action)
        tray_menu.addAction(save_action)
        tray_menu.addAction(separator2)
        tray_menu.addAction(quit_action)
        
        # Configurar bandeja
        tray_icon.setContextMenu(tray_menu)
        tray_icon.setToolTip("Sistema PTZ Profesional")
        
        # Conectar doble clic
        tray_icon.activated.connect(lambda reason: 
            show_system_status(ptz_manager) if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None
        )
        
        return tray_icon
        
    except Exception as e:
        logger.error(f"Error creando bandeja del sistema: {e}")
        return None

def show_system_status(ptz_manager: PTZSystemManager):
    """
    Mostrar estado del sistema en ventana
    
    Args:
        ptz_manager: Gestor del sistema PTZ
    """
    try:
        status = ptz_manager.get_system_status()
        
        status_text = f"""
🎯 Sistema PTZ Profesional - Estado Actual

📊 Estado General:
• Sistema en ejecución: {'✅ Sí' if status.get('system_running') else '❌ No'}
• Cámaras configuradas: {status.get('cameras_configured', 0)}
• Cámaras en seguimiento: {status.get('cameras_tracking', 0)}
• Ventanas de config abiertas: {status.get('config_windows_open', 0)}

🎮 Estado de Grilla:
• Dimensiones: {status.get('grid_status', {}).get('dimensions', 'N/A')}
• Celdas activas: {status.get('grid_status', {}).get('active_cells', 0)}
• Zonas configuradas: {status.get('grid_status', {}).get('zones', 0)}

📹 Detalles de Cámaras:
"""
        
        tracking_details = status.get('tracking_details', {})
        if tracking_details:
            for camera_ip, camera_status in tracking_details.items():
                running_status = "🟢 Activo" if camera_status.get('running') else "🔴 Inactivo"
                target = camera_status.get('current_target', 'Ninguno')
                queue_size = camera_status.get('queue_size', 0)
                
                status_text += f"""
• {camera_ip}: {running_status}
  - Objetivo actual: {target}
  - Cola de detecciones: {queue_size}
"""
        else:
            status_text += "\n• No hay cámaras PTZ configuradas"
        
        # Mostrar en mensaje
        msg = QMessageBox()
        msg.setWindowTitle("Estado del Sistema PTZ")
        msg.setText(status_text)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()
        
    except Exception as e:
        logger.error(f"Error mostrando estado: {e}")
        QMessageBox.critical(None, "Error", f"Error obteniendo estado del sistema:\n{e}")

def setup_cli_arguments() -> argparse.ArgumentParser:
    """
    Configurar argumentos de línea de comandos
    
    Returns:
        argparse.ArgumentParser: Parser de argumentos
    """
    parser = argparse.ArgumentParser(
        description="Sistema PTZ Profesional - Control y Seguimiento Automático",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python main_ptz_integration.py --config my_config.json
  python main_ptz_integration.py --gui --auto-start
  python main_ptz_integration.py --add-camera 192.168.1.100 admin password123
  python main_ptz_integration.py --emergency-stop
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="ptz_tracking_config.json",
        help="Archivo de configuración PTZ (default: ptz_tracking_config.json)"
    )
    
    parser.add_argument(
        "--gui", "-g",
        action="store_true",
        help="Mostrar interfaz gráfica con bandeja del sistema"
    )
    
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="Iniciar seguimiento automáticamente"
    )
    
    parser.add_argument(
        "--add-camera",
        nargs=3,
        metavar=("IP", "USER", "PASS"),
        help="Agregar cámara PTZ: IP usuario contraseña"
    )
    
    parser.add_argument(
        "--remove-camera",
        type=str,
        metavar="IP",
        help="Remover cámara PTZ por IP"
    )
    
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="Listar cámaras PTZ configuradas"
    )
    
    parser.add_argument(
        "--emergency-stop",
        action="store_true",
        help="Ejecutar parada de emergencia en todas las cámaras"
    )
    
    parser.add_argument(
        "--status",
        action="store_true",
        help="Mostrar estado del sistema y salir"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Nivel de logging (default: INFO)"
    )
    
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Ejecutar como servicio en segundo plano"
    )
    
    return parser

def handle_cli_commands(args, ptz_manager: PTZSystemManager) -> bool:
    """
    Manejar comandos de línea de comandos
    
    Args:
        args: Argumentos parseados
        ptz_manager: Gestor del sistema PTZ
        
    Returns:
        bool: True si se debe continuar con la ejecución normal
    """
    try:
        # Comando de parada de emergencia
        if args.emergency_stop:
            print("🚨 Ejecutando parada de emergencia...")
            ptz_manager.emergency_stop()
            print("✅ Parada de emergencia completada")
            return False
        
        # Comando de estado
        if args.status:
            status = ptz_manager.get_system_status()
            print("\n🎯 Estado del Sistema PTZ:")
            print(f"• Sistema activo: {'Sí' if status.get('system_running') else 'No'}")
            print(f"• Cámaras configuradas: {status.get('cameras_configured', 0)}")
            print(f"• Cámaras en seguimiento: {status.get('cameras_tracking', 0)}")
            
            tracking_details = status.get('tracking_details', {})
            if tracking_details:
                print("\n📹 Detalle de cámaras:")
                for ip, details in tracking_details.items():
                    status_icon = "🟢" if details.get('running') else "🔴"
                    print(f"  {status_icon} {ip} - Objetivo: {details.get('current_target', 'Ninguno')}")
            
            return False
        
        # Comando listar cámaras
        if args.list_cameras:
            cameras = ptz_manager.ptz_integration.tracking_manager.camera_configs
            print(f"\n📹 Cámaras PTZ configuradas ({len(cameras)}):")
            
            if cameras:
                for ip, config in cameras.items():
                    mode_icon = "🎯" if config.tracking_enabled else "📊"
                    print(f"  {mode_icon} {ip}:{config.port} ({config.username}) - {config.tracking_mode.value}")
            else:
                print("  No hay cámaras PTZ configuradas")
            
            return False
        
        # Comando agregar cámara
        if args.add_camera:
            ip, username, password = args.add_camera
            
            camera_config = {
                "ip": ip,
                "port": 80,
                "username": username,
                "password": password,
                "tracking_mode": "tracking",
                "tracking_enabled": False
            }
            
            print(f"🔧 Agregando cámara PTZ {ip}...")
            
            if ptz_manager.ptz_integration.add_ptz_camera(camera_config):
                print(f"✅ Cámara {ip} agregada exitosamente")
                ptz_manager.save_system_configuration()
            else:
                print(f"❌ Error agregando cámara {ip}")
            
            return False
        
        # Comando remover cámara
        if args.remove_camera:
            ip = args.remove_camera
            
            print(f"🗑️ Removiendo cámara PTZ {ip}...")
            
            if ptz_manager.ptz_integration.remove_ptz_camera(ip):
                print(f"✅ Cámara {ip} removida exitosamente")
                ptz_manager.save_system_configuration()
            else:
                print(f"❌ Error removiendo cámara {ip}")
            
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error manejando comando CLI: {e}")
        print(f"❌ Error: {e}")
        return False

def main():
    """Función principal del sistema PTZ"""
    try:
        # Configurar argumentos CLI
        parser = setup_cli_arguments()
        args = parser.parse_args()
        
        # Configurar nivel de logging
        logging.getLogger().setLevel(getattr(logging, args.log_level))
        
        print("🚀 Iniciando Sistema PTZ Profesional...")
        
        # Crear aplicación Qt si se requiere GUI
        app = None
        if args.gui or not any([args.emergency_stop, args.status, args.list_cameras, 
                               args.add_camera, args.remove_camera]):
            app = QApplication(sys.argv)
            app.setQuitOnLastWindowClosed(False)  # Mantener en bandeja
        
        # Crear gestor del sistema
        ptz_manager = PTZSystemManager()
        
        # Inicializar sistema
        if not ptz_manager.initialize(config_file=args.config):
            print("❌ Error inicializando sistema PTZ")
            return 1
        
        # Manejar comandos CLI
        if not handle_cli_commands(args, ptz_manager):
            return 0
        
        # Auto-iniciar seguimiento si se especifica
        if args.auto_start:
            print("🎯 Iniciando seguimiento automático...")
            ptz_manager.start_tracking_system()
        
        # Crear interfaz gráfica si se requiere
        if args.gui and app:
            print("🖥️ Inicializando interfaz gráfica...")
            
            # Verificar soporte de bandeja del sistema
            if not QSystemTrayIcon.isSystemTrayAvailable():
                QMessageBox.critical(None, "Error", 
                    "La bandeja del sistema no está disponible en este sistema.")
                return 1
            
            # Crear icono de bandeja
            tray_icon = create_system_tray(ptz_manager)
            
            if tray_icon:
                tray_icon.show()
                tray_icon.showMessage(
                    "Sistema PTZ",
                    "Sistema PTZ Profesional iniciado\nHaga clic derecho para opciones",
                    QSystemTrayIcon.MessageIcon.Information,
                    3000
                )
                
                print("✅ Sistema PTZ ejecutándose en bandeja del sistema")
                print("💡 Haga clic derecho en el icono de la bandeja para opciones")
            
            # Ejecutar aplicación
            return app.exec()
        
        elif args.daemon:
            print("🔄 Ejecutando como servicio...")
            
            # Modo daemon - mantener ejecutándose
            try:
                while ptz_manager.is_running:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Interrupción recibida, cerrando...")
            
            return 0
        
        else:
            # Modo CLI sin GUI
            print("✅ Sistema PTZ inicializado (modo CLI)")
            print("💡 Use --help para ver opciones disponibles")
            print("🛑 Presione Ctrl+C para salir")
            
            try:
                while ptz_manager.is_running:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Cerrando sistema...")
            
            return 0
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupción recibida")
        return 0
    
    except Exception as e:
        logger.error(f"Error fatal en main: {e}")
        print(f"❌ Error fatal: {e}")
        return 1
    
    finally:
        # Limpieza final
        try:
            if 'ptz_manager' in locals():
                ptz_manager.shutdown()
        except:
            pass

# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    # Importar time para timestamps
    import time
    
    # Ejecutar función principal
    exit_code = main()
    
    # Salir con código de estado
    sys.exit(exit_code)