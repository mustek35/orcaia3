from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QTextEdit, QMenuBar, QMenu, QGridLayout, QStackedWidget, QLabel,
    QScrollArea, QPushButton, QMessageBox
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt, QTimer
import importlib
from ui.camera_modal import CameraDialog
from gui.resumen_detecciones import ResumenDeteccionesWidget
from ui.config_modal import ConfiguracionDialog
from ui.fps_config_dialog import FPSConfigDialog
from ui.camera_manager import guardar_camaras, cargar_camaras_guardadas
from core.rtsp_builder import generar_rtsp
import os
import cProfile
import pstats
import io

# === IMPORTS SISTEMA PTZ ===
try:
    from core.ptz_integration import PTZSystemIntegration, PTZControlInterface
    from core.detection_ptz_bridge import detection_ptz_bridge, setup_ptz_integration_hooks
    PTZ_AVAILABLE = True
    print("✅ Sistema PTZ disponible")
except ImportError as e:
    PTZ_AVAILABLE = False
    print(f"⚠️ Sistema PTZ no disponible: {e}")
# === FIN IMPORTS PTZ ===

CONFIG_PATH = "config.json"

class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        print("INFO: Iniciando profiler para MainGUI...")
        self.profiler = cProfile.Profile()
        self.profiler.enable()

        self.setWindowTitle("Monitor PTZ Inteligente - Orca")
        self.setGeometry(100, 100, 1600, 900)

        # Configuración de FPS por defecto
        self.fps_config = {
            "visual_fps": 25,
            "detection_fps": 8, 
            "ui_update_fps": 15,
            "adaptive_fps": True
        }

        self.camera_data_list = []
        self.camera_widgets = []

        # === INICIALIZACIÓN SISTEMA PTZ (Variables) ===
        self.ptz_system = None
        self.ptz_control_interface = None
        self.ptz_config_widgets = {}
        # === FIN VARIABLES PTZ ===

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        self.setup_menu_bar()
        self.setup_ui()

        # === INICIALIZACIÓN SISTEMA PTZ (Después de UI) ===
        if PTZ_AVAILABLE:
            self.initialize_ptz_system()
        # === FIN INICIALIZACIÓN PTZ ===

        cargar_camaras_guardadas(self)

        # === TIMER ESTADO PTZ ===
        if PTZ_AVAILABLE and self.ptz_system:
            self.ptz_status_timer = QTimer()
            self.ptz_status_timer.timeout.connect(self.update_ptz_status)
            self.ptz_status_timer.start(3000)
        # === FIN TIMER PTZ ===

    def initialize_ptz_system(self):
        """Inicializar sistema PTZ integrado de manera segura"""
        try:
            print("🚀 Inicializando sistema PTZ...")
            
            # Crear integración PTZ
            self.ptz_system = PTZSystemIntegration(main_app=self)
            self.ptz_control_interface = PTZControlInterface(self.ptz_system)
            
            # Configurar hooks de detección automática
            setup_ptz_integration_hooks()
            
            # Registrar callback para envío de detecciones
            self.ptz_system.register_detection_callback(self.on_detection_for_ptz)
            
            print("✅ Sistema PTZ inicializado correctamente")
            self.append_debug("🎯 Sistema PTZ listo para seguimiento automático")
            
        except Exception as e:
            print(f"❌ Error inicializando sistema PTZ: {e}")
            self.append_debug(f"⚠️ Sistema PTZ no disponible: {e}")
            self.ptz_system = None
            self.ptz_control_interface = None

    def setup_menu_bar(self):
        """Configurar barra de menú"""
        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        self.menu_inicio = self.menu_bar.addMenu("Inicio")
        self.menu_config = self.menu_bar.addMenu("Configuración")

        self.action_agregar = QAction("➕ Agregar Cámara", self)
        self.action_agregar.triggered.connect(lambda: self.open_camera_dialog())
        self.menu_inicio.addAction(self.action_agregar)

        self.action_salir = QAction("🚪 Salir de la Aplicación", self)
        self.action_salir.triggered.connect(self.close) 
        self.menu_inicio.addAction(self.action_salir)

        # === MENÚ PTZ ===
        if PTZ_AVAILABLE:
            self.menu_ptz = self.menu_bar.addMenu("🎯 PTZ")
            
            self.action_ptz_panel = QAction("🎮 Panel de Control PTZ", self)
            self.action_ptz_panel.triggered.connect(self.abrir_panel_ptz_principal)
            self.menu_ptz.addAction(self.action_ptz_panel)
            
            self.action_ptz_emergency = QAction("🚨 PARADA DE EMERGENCIA", self)
            self.action_ptz_emergency.triggered.connect(self.emergency_stop_all_ptz)
            self.menu_ptz.addAction(self.action_ptz_emergency)
        # === FIN MENÚ PTZ ===

    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        self.init_tab = QWidget()
        self.init_tab_layout = QVBoxLayout()
        self.init_tab.setLayout(self.init_tab_layout)
        
        self.setup_inicio_ui() 

        self.stacked_widget.addWidget(self.init_tab)

    def setup_inicio_ui(self):
        """Configurar UI principal"""
        from PyQt6.QtWidgets import QSplitter

        # --- Parte superior: cámaras ---
        self.video_grid = QGridLayout()
        video_grid_container_widget = QWidget()
        video_grid_container_widget.setLayout(self.video_grid)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(video_grid_container_widget)

        # --- Parte inferior: lista + log + resumen ---
        bottom_layout = QHBoxLayout()
    
        self.camera_list = QListWidget()
        self.camera_list.setFixedWidth(250)
        bottom_layout.addWidget(self.camera_list)

        self.debug_console = QTextEdit()
        self.debug_console.setReadOnly(True)
        bottom_layout.addWidget(self.debug_console, 2)

        self.resumen_widget = ResumenDeteccionesWidget()
        self.resumen_widget.log_signal.connect(self.append_debug)
        bottom_layout.addWidget(self.resumen_widget, 1)

        bottom_widget = QWidget()
        bottom_widget.setLayout(bottom_layout)

        # --- Dividir con splitter vertical ---
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(scroll_area)
        splitter.addWidget(bottom_widget)
        splitter.setSizes([1, 1])

        self.init_tab_layout.addWidget(splitter)

    def append_debug(self, message: str):
        """Método seguro para agregar mensajes de debug"""
        try:
            # Filtrar mensajes de spam
            if any(substr in message for substr in ["hevc @", "VPS 0", "undecodable NALU", "Frame procesado"]):
                return
            
            # Verificar si debug_console existe
            if hasattr(self, 'debug_console') and self.debug_console:
                self.debug_console.append(message)
            else:
                # Fallback a print si no existe debug_console
                print(f"[DEBUG] {message}")
        except Exception as e:
            print(f"[DEBUG Error] {message} (Error: {e})")

    def open_camera_dialog(self, index=None):
        """Abrir diálogo de cámara"""
        print("🛠️ [DEBUG] Ejecutando open_camera_dialog")
        if index is not None and index >= len(self.camera_data_list):
            return
        existing = self.camera_data_list[index] if index is not None else None
        dialog = CameraDialog(self, existing_data=existing)
        if dialog.exec(): 
            if dialog.result() == 1: 
                new_data = dialog.get_camera_data()
                if index is not None:
                    self.camera_data_list[index] = new_data
                    self.camera_list.item(index).setText(f"{new_data['ip']} - {new_data['tipo']}")
                    self.append_debug(f"✏️ Cámara editada: {new_data}")
                else:
                    self.camera_data_list.append(new_data)
                    self.camera_list.addItem(f"{new_data['ip']} - {new_data['tipo']}")
                    self.append_debug(f"✅ Cámara agregada: {new_data}")
                
                # === INTEGRACIÓN PTZ AUTOMÁTICA ===
                if PTZ_AVAILABLE and new_data.get('tipo') == 'ptz':
                    self.agregar_camara_ptz_automatica(new_data)
                # === FIN INTEGRACIÓN PTZ ===
                
                guardar_camaras(self)

    def agregar_camara_ptz_automatica(self, camera_data):
        """Agregar cámara PTZ automáticamente al sistema de seguimiento"""
        try:
            if not self.ptz_system:
                return
            
            ptz_config = {
                "ip": camera_data["ip"],
                "port": camera_data.get("puerto", 80),
                "username": camera_data["usuario"],
                "password": camera_data["contrasena"],
                "tracking_mode": "analytics_only",
                "tracking_enabled": False,
            }
            
            success = self.ptz_system.add_ptz_camera(ptz_config)
            
            if success:
                self.append_debug(f"🎯 Cámara PTZ {camera_data['ip']} agregada al sistema")
            else:
                self.append_debug(f"⚠️ No se pudo agregar cámara PTZ {camera_data['ip']}")
                
        except Exception as e:
            self.append_debug(f"❌ Error agregando cámara PTZ: {e}")

    def abrir_panel_ptz_principal(self):
        """Abrir panel principal de control PTZ"""
        try:
            if not PTZ_AVAILABLE or not self.ptz_system:
                QMessageBox.warning(self, "PTZ No Disponible", 
                    "❌ Sistema PTZ no está disponible")
                return
            
            QMessageBox.information(self, "Panel PTZ", 
                "🎮 Panel PTZ disponible

Funcionalidad básica implementada")
                
        except Exception as e:
            QMessageBox.critical(self, "Error PTZ", f"Error: {str(e)}")

    def emergency_stop_all_ptz(self):
        """Parada de emergencia para todas las cámaras PTZ"""
        try:
            if not PTZ_AVAILABLE or not self.ptz_system:
                QMessageBox.warning(self, "PTZ No Disponible", "Sistema PTZ no disponible")
                return
            
            self.ptz_system.emergency_stop_all()
            QMessageBox.warning(self, "Parada de Emergencia", 
                "🚨 PARADA DE EMERGENCIA EJECUTADA")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error: {str(e)}")

    def update_ptz_status(self):
        """Actualizar estado PTZ"""
        pass

    def on_detection_for_ptz(self, detection_data, source_camera_ip):
        """Callback para detecciones PTZ"""
        pass

    def closeEvent(self, event):
        """Manejar cierre de aplicación"""
        try:
            print("INFO: Cerrando aplicación...")
            
            # === CERRAR SISTEMA PTZ ===
            if PTZ_AVAILABLE and self.ptz_system:
                try:
                    self.ptz_system.save_configuration()
                    self.ptz_system.shutdown()
                except Exception as e:
                    print(f"Error cerrando PTZ: {e}")
            # === FIN CIERRE PTZ ===
            
            event.accept()
            
        except Exception as e:
            print(f"Error en cierre: {e}")
            event.accept()
