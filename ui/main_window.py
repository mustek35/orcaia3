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
    from gui.ptz_config_widget import PTZConfigWidget
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

        # === INICIALIZACIÓN SISTEMA PTZ ===
        self.ptz_system = None
        self.ptz_control_interface = None
        self.ptz_config_widgets = {}
        if PTZ_AVAILABLE:
            pass
        # === FIN INICIALIZACIÓN PTZ ===

        # Configuración de FPS por defecto
        self.fps_config = {
            "visual_fps": 25,
            "detection_fps": 8, 
            "ui_update_fps": 15,
            "adaptive_fps": True
        }

        self.camera_data_list = []
        self.camera_widgets = [] 

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        self.setup_menu_bar()
        self.setup_ui()
            self.initialize_ptz_system()

        cargar_camaras_guardadas(self)

        # === TIMER ESTADO PTZ ===
        if PTZ_AVAILABLE and self.ptz_system:
            self.ptz_status_timer = QTimer()
            self.ptz_status_timer.timeout.connect(self.update_ptz_status)
            self.ptz_status_timer.start(3000)  # Cada 3 segundos
        # === FIN TIMER PTZ ===

    def initialize_ptz_system(self):
        pass
    """Inicializar sistema PTZ integrado de manera segura"""
    try:
        pass
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
        pass
    print(f"❌ Error inicializando sistema PTZ: {e}")
    self.append_debug(f"⚠️ Sistema PTZ no disponible: {e}")
    self.ptz_system = None
    self.ptz_control_interface = None
    def setup_menu_bar(self):
        """Configurar barra de menú con opciones PTZ"""
        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        # Menú Inicio
        self.menu_inicio = self.menu_bar.addMenu("Inicio")
        
        self.action_agregar = QAction("➕ Agregar Cámara", self)
        self.action_agregar.triggered.connect(lambda: self.open_camera_dialog())
        self.menu_inicio.addAction(self.action_agregar)

        self.action_salir = QAction("🚪 Salir de la Aplicación", self)
        self.action_salir.triggered.connect(self.close) 
        self.menu_inicio.addAction(self.action_salir)

        # Menú Configuración  
        self.menu_config = self.menu_bar.addMenu("Configuración")
        
        self.action_ver_config = QAction("⚙️ Ver Configuración", self)
        self.action_ver_config.triggered.connect(self.abrir_configuracion_modal)
        self.menu_config.addAction(self.action_ver_config)

        self.action_fps_config = QAction("🎯 Configurar FPS", self)
        self.action_fps_config.triggered.connect(self.abrir_fps_config)
        self.menu_config.addAction(self.action_fps_config)

        self.action_edit_line = QAction("🏁 Línea de Cruce", self)
        self.action_edit_line.triggered.connect(self.toggle_line_edit)
        self.menu_config.addAction(self.action_edit_line)

        # === MENÚ PTZ ===
        if PTZ_AVAILABLE:
            self.menu_ptz = self.menu_bar.addMenu("🎯 PTZ")
            
            self.action_ptz_panel = QAction("🎮 Panel de Control PTZ", self)
            self.action_ptz_panel.triggered.connect(self.abrir_panel_ptz_principal)
            self.menu_ptz.addAction(self.action_ptz_panel)
            
            self.action_ptz_emergency = QAction("🚨 PARADA DE EMERGENCIA", self)
            self.action_ptz_emergency.triggered.connect(self.emergency_stop_all_ptz)
            self.menu_ptz.addAction(self.action_ptz_emergency)
            
            self.action_ptz_status = QAction("📊 Estado del Sistema PTZ", self)
            self.action_ptz_status.triggered.connect(self.mostrar_estado_ptz)
            self.menu_ptz.addAction(self.action_ptz_status)
        # === FIN MENÚ PTZ ===

    def setup_ui(self):
        """Configurar interfaz de usuario"""
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        self.init_tab = QWidget()
        self.init_tab_layout = QVBoxLayout()
        self.init_tab.setLayout(self.init_tab_layout)
        
        # === PANEL DE CONTROL PTZ ===
        if PTZ_AVAILABLE:
            self.setup_ptz_control_panel()
        # === FIN PANEL PTZ ===
        
        self.setup_inicio_ui() 

        self.stacked_widget.addWidget(self.init_tab)

    def setup_ptz_control_panel(self):
        """Crear panel de control PTZ en la interfaz principal"""
        try:
            ptz_panel = QWidget()
            ptz_layout = QHBoxLayout()
            ptz_panel.setLayout(ptz_layout)
            
            # Título del panel PTZ
            ptz_title = QLabel("🎯 Control PTZ:")
            ptz_title.setStyleSheet("font-weight: bold; color: #2196F3;")
            ptz_layout.addWidget(ptz_title)
            
            # Botón Panel PTZ
            self.btn_panel_ptz = QPushButton("🎮 Panel PTZ")
            self.btn_panel_ptz.setToolTip("Abrir panel completo de configuración PTZ")
            self.btn_panel_ptz.clicked.connect(self.abrir_panel_ptz_principal)
            self.btn_panel_ptz.setStyleSheet("""
                QPushButton {
                    background-color: #2196F3;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #1976D2;
                }
            """)
            ptz_layout.addWidget(self.btn_panel_ptz)
            
            # Botón Parada de Emergencia
            self.btn_emergency_stop = QPushButton("🚨 STOP")
            self.btn_emergency_stop.setToolTip("Parada de emergencia para todas las cámaras PTZ")
            self.btn_emergency_stop.clicked.connect(self.emergency_stop_all_ptz)
            self.btn_emergency_stop.setStyleSheet("""
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    border: none;
                    padding: 8px 15px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #d32f2f;
                }
            """)
            ptz_layout.addWidget(self.btn_emergency_stop)
            
            # Indicador de estado PTZ
            self.lbl_ptz_status = QLabel("PTZ: Iniciando...")
            self.lbl_ptz_status.setStyleSheet("""
                QLabel {
                    border: 1px solid #ddd;
                    padding: 8px;
                    border-radius: 4px;
                    background-color: #f5f5f5;
                }
            """)
            ptz_layout.addWidget(self.lbl_ptz_status)
            
            # Botón configuración rápida
            self.btn_quick_config = QPushButton("⚙️ Config Rápida")
            self.btn_quick_config.setToolTip("Configuración rápida de cámara PTZ")
            self.btn_quick_config.clicked.connect(self.configuracion_rapida_ptz)
            ptz_layout.addWidget(self.btn_quick_config)
            
            ptz_layout.addStretch()  # Espacio flexible
            
            # Agregar panel al layout principal
            self.init_tab_layout.addWidget(ptz_panel)
            
        except Exception as e:
            print(f"Error creando panel PTZ: {e}")

    def abrir_fps_config(self):
        """Abrir diálogo de configuración de FPS"""
        dialog = FPSConfigDialog(self, self.fps_config)
        dialog.fps_config_changed.connect(self.update_fps_config)
        
        if dialog.exec():
            self.fps_config = dialog.get_config()
            self.apply_fps_to_all_cameras()
            self.append_debug(f"⚙️ Configuración de FPS aplicada: {self.fps_config}")
    
    def update_fps_config(self, config):
        """Actualizar configuración de FPS en tiempo real"""
        self.fps_config = config
        self.apply_fps_to_all_cameras()
        self.append_debug(f"🎯 FPS actualizado en tiempo real: Visual={config['visual_fps']}, "
                         f"Detección={config['detection_fps']}, UI={config['ui_update_fps']}")
    
    def apply_fps_to_all_cameras(self):
        """Aplicar configuración de FPS a todas las cámaras activas"""
        for widget in self.camera_widgets:
            try:
                # Actualizar GrillaWidget
                if hasattr(widget, 'set_fps_config'):
                    widget.set_fps_config(
                        visual_fps=self.fps_config['visual_fps'],
                        detection_fps=self.fps_config['detection_fps'],
                        ui_update_fps=self.fps_config['ui_update_fps']
                    )
                
                # Actualizar VisualizadorDetector
                if hasattr(widget, 'visualizador') and widget.visualizador:
                    if hasattr(widget.visualizador, 'update_fps_config'):
                        widget.visualizador.update_fps_config(
                            visual_fps=self.fps_config['visual_fps'],
                            detection_fps=self.fps_config['detection_fps']
                        )
                        
            except Exception as e:
                self.append_debug(f"❌ Error aplicando FPS a cámara: {e}")

    def get_optimized_fps_for_camera(self, camera_data):
        """Obtener configuración de FPS optimizada según el tipo de cámara"""
        base_config = self.fps_config.copy()
        
        # Ajustar según el tipo de cámara
        camera_type = camera_data.get('tipo', 'fija')
        models = camera_data.get('modelos', [camera_data.get('modelo', 'Personas')])
        
        if camera_type == 'ptz':
            # PTZ necesita más FPS para seguimiento fluido
            base_config['visual_fps'] = min(30, base_config['visual_fps'] + 5)
            base_config['detection_fps'] = min(15, base_config['detection_fps'] + 2)
        
        if 'Embarcaciones' in models or 'Barcos' in models:
            # Detección marítima puede necesitar menos FPS
            base_config['detection_fps'] = max(3, base_config['detection_fps'] - 2)
        
        return base_config

    def append_debug(self, message: str):
        pass
    """Método seguro para agregar mensajes de debug"""
    try:
        pass
    # Filtrar mensajes de spam
    if any(substr in message for substr in ["hevc @", "VPS 0", "undecodable NALU", "Frame procesado"]):
        pass
    return

    # Verificar si debug_console existe
    if hasattr(self, 'debug_console') and self.debug_console:
        pass
    self.debug_console.append(message)
    else:
        pass
    # Fallback a print si no existe debug_console
    print(f"[DEBUG] {message}")
    except Exception as e:
        pass
    print(f"[DEBUG Error] {message} (Error: {e})")
    def setup_inicio_ui(self):
        from PyQt6.QtWidgets import QSplitter

        # --- Parte superior: cámaras ---
        self.video_grid = QGridLayout()
        video_grid_container_widget = QWidget()
        video_grid_container_widget.setLayout(self.video_grid)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setWidget(video_grid_container_widget)

        # --- Parte inferior: lista + log + resumen ---
        bottom_layout = QHBoxLayout()
    
        self.camera_list = QListWidget()
        self.camera_list.setFixedWidth(250)
        self.camera_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.camera_list.customContextMenuRequested.connect(self.show_camera_menu)
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
        splitter.setSizes([1, 1])  # 50% y 50%

        self.init_tab_layout.addWidget(splitter)

    def open_camera_dialog(self, index=None):
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
                    self.start_camera_stream(new_data) 
                else:
                    self.camera_data_list.append(new_data)
                    self.camera_list.addItem(f"{new_data['ip']} - {new_data['tipo']}")
                    self.append_debug(f"✅ Cámara agregada: {new_data}")
                    self.start_camera_stream(new_data)
                
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
                "tracking_mode": "analytics_only",  # Modo seguro por defecto
                "tracking_enabled": False,  # Deshabilitado hasta configurar
                "tracking_sensitivity": 0.005,
                "max_pt_speed": 0.5,
                "deadzone_x": 0.03,
                "deadzone_y": 0.03,
                "confirmation_frames": 3
            }
            
            success = self.ptz_system.add_ptz_camera(ptz_config)
            
            if success:
                self.append_debug(f"🎯 Cámara PTZ {camera_data['ip']} agregada al sistema de seguimiento")
                self.append_debug("💡 Use 'Panel PTZ' para configurar seguimiento automático")
            else:
                self.append_debug(f"⚠️ No se pudo agregar cámara PTZ {camera_data['ip']} al seguimiento")
                
        except Exception as e:
            self.append_debug(f"❌ Error agregando cámara PTZ: {e}")

    def abrir_configuracion_modal(self):
        dialog = ConfiguracionDialog(self, camera_list=self.camera_data_list)
        if dialog.exec():
            guardar_camaras(self)
            self.append_debug(f"⚙️ Configuración del sistema guardada.")
        else:
            self.append_debug(f"⚙️ Cambios en configuración del sistema cancelados.")

    def toggle_line_edit(self):
        items = self.camera_list.selectedItems()
        if not items:
            return
        index = self.camera_list.row(items[0])
        if index >= len(self.camera_widgets):
            return
        widget = self.camera_widgets[index]
        if widget.cross_line_edit_mode:
            widget.finish_line_edit()
        else:
            widget.start_line_edit()

    def start_camera_stream(self, camera_data):
        # Agregar configuración de FPS optimizada a los datos de la cámara
        optimized_fps = self.get_optimized_fps_for_camera(camera_data)
        camera_data['fps_config'] = optimized_fps

        for i, widget in enumerate(self.camera_widgets):
            if hasattr(widget, 'cam_data') and widget.cam_data.get('ip') == camera_data.get('ip'):
                print(f"INFO: Reemplazando widget para cámara IP: {camera_data.get('ip')}")
                widget.detener()
                self.video_grid.removeWidget(widget) 
                widget.deleteLater()
                self.camera_widgets.pop(i)
                break
        
        video_grid_container_widget = None
        # Buscar el video_grid_container_widget que es el widget del scroll_area
        for i in range(self.init_tab_layout.count()):
            item = self.init_tab_layout.itemAt(i)
            if isinstance(item.widget(), QScrollArea):
                video_grid_container_widget = item.widget().widget()
                break

        try:
            grilla_widget_module = importlib.import_module("gui.grilla_widget")
            GrillaWidget_class = grilla_widget_module.GrillaWidget
        except ImportError as e:
            print(f"ERROR: No se pudo importar GrillaWidget: {e}")
            self.append_debug(f"ERROR: No se pudo importar GrillaWidget: {e}")
            return

        parent_widget = video_grid_container_widget if video_grid_container_widget else self
        video_widget = GrillaWidget_class(parent=parent_widget, fps_config=optimized_fps) 
        
        video_widget.cam_data = camera_data 
        video_widget.log_signal.connect(self.append_debug)
        
        # === AGREGAR CALLBACK PTZ ===
        if PTZ_AVAILABLE and camera_data.get('tipo') == 'ptz':
            self.setup_ptz_callbacks_for_widget(video_widget, camera_data['ip'])
        # === FIN CALLBACK PTZ ===
        
        row = 0
        col = len(self.camera_widgets) 
        
        self.video_grid.addWidget(video_widget, row, col)
        self.camera_widgets.append(video_widget) 
        
        video_widget.mostrar_vista(camera_data) 
        video_widget.show()
        self.append_debug(f"🎥 Reproduciendo: {camera_data.get('ip', 'IP Desconocida')} con FPS optimizado")

    def setup_ptz_callbacks_for_widget(self, widget, camera_ip):
        """Configurar callbacks PTZ para widget de cámara"""
        try:
            # Si el widget tiene procesamiento de detecciones, interceptarlo
            if hasattr(widget, 'on_detection_signal'):
                original_detection_handler = widget.on_detection_signal
                
                def enhanced_detection_handler(detection_data):
                    # Llamar handler original
                    result = original_detection_handler(detection_data)
                    
                    # Enviar al sistema PTZ
                    if detection_data and isinstance(detection_data, dict):
                        self.send_detection_to_ptz(detection_data, camera_ip)
                    
                    return result
                
                widget.on_detection_signal = enhanced_detection_handler
                
            self.append_debug(f"🔗 Callbacks PTZ configurados para {camera_ip}")
            
        except Exception as e:
            self.append_debug(f"⚠️ Error configurando callbacks PTZ: {e}")

    def send_detection_to_ptz(self, detection_data, camera_ip):
        """Enviar detección al sistema PTZ"""
        try:
            # Usar el bridge de detección para procesar y enviar
            if hasattr(detection_ptz_bridge, 'process_detection_from_gui'):
                detection_ptz_bridge.process_detection_from_gui(detection_data, camera_ip)
            
        except Exception as e:
            print(f"Error enviando detección a PTZ: {e}")

    def on_detection_for_ptz(self, detection_data, source_camera_ip):
        """Callback para detecciones enviadas al sistema PTZ"""
        try:
            # Log opcional de seguimiento PTZ (no spam)
            if detection_data.get('confidence', 0) > 0.7:  # Solo detecciones de alta confianza
                obj_type = detection_data.get('object_type', 'objeto')
                self.append_debug(f"🎯 PTZ: Siguiendo {obj_type} desde {source_camera_ip}")
                
        except Exception as e:
            print(f"Error en callback PTZ: {e}")

    def show_camera_menu(self, position):
        item = self.camera_list.itemAt(position)
        if item:
            index = self.camera_list.row(item)
            menu = QMenu()
            edit_action = menu.addAction("✏️ Editar Cámara")
            delete_action = menu.addAction("🗑️ Eliminar Cámara")
            stop_action = menu.addAction("⛔ Detener Visual") 
            fps_action = menu.addAction("🎯 Configurar FPS Individual")
            
            # === OPCIONES PTZ ===
            if PTZ_AVAILABLE and index < len(self.camera_data_list):
                camera_data = self.camera_data_list[index]
                if camera_data.get('tipo') == 'ptz':
                    ptz_config_action = menu.addAction("🎮 Configurar PTZ")
                    ptz_test_action = menu.addAction("🔧 Probar Conexión PTZ")
            # === FIN OPCIONES PTZ ===
            
            action = menu.exec(self.camera_list.mapToGlobal(position))

            if action == edit_action:
                self.open_camera_dialog(index=index) 
            elif action == delete_action:
                cam_to_delete_data = self.camera_data_list.pop(index)
                self.camera_list.takeItem(index) 
                
                # === REMOVER DE SISTEMA PTZ ===
                if PTZ_AVAILABLE and cam_to_delete_data.get('tipo') == 'ptz':
                    self.remover_camara_ptz(cam_to_delete_data['ip'])
                # === FIN REMOVER PTZ ===
                
                for i, widget in enumerate(self.camera_widgets):
                    if hasattr(widget, 'cam_data') and widget.cam_data.get('ip') == cam_to_delete_data.get('ip'):
                        widget.detener()
                        self.video_grid.removeWidget(widget)
                        widget.deleteLater()
                        self.camera_widgets.pop(i)
                        self.append_debug(f"🗑️ Cámara {cam_to_delete_data.get('ip')} y su widget eliminados.")
                        break
                guardar_camaras(self) 
            elif action == stop_action:
                cam_ip_to_stop = self.camera_data_list[index].get('ip')
                for i, widget in enumerate(self.camera_widgets):
                     if hasattr(widget, 'cam_data') and widget.cam_data.get('ip') == cam_ip_to_stop:
                         pass
                        widget.detener()
                        self.append_debug(f"⛔ Visual detenida para: {cam_ip_to_stop}")
                        break
            elif action == fps_action:
                self.configure_individual_fps(index)
            # === MANEJAR ACCIONES PTZ ===
            elif PTZ_AVAILABLE and 'action' in locals():
                if hasattr(action, 'text'):
                    if action.text() == "🎮 Configurar PTZ":
                        self.abrir_configuracion_ptz_individual(index)
                    elif action.text() == "🔧 Probar Conexión PTZ":
                        self.probar_conexion_ptz_individual(index)
            # === FIN ACCIONES PTZ ===

    def remover_camara_ptz(self, camera_ip):
        """Remover cámara del sistema PTZ"""
        try:
            if self.ptz_system:
                self.ptz_system.remove_ptz_camera(camera_ip)
                self.append_debug(f"🎯 Cámara PTZ {camera_ip} removida del sistema de seguimiento")
        except Exception as e:
            self.append_debug(f"❌ Error removiendo cámara PTZ: {e}")

    def abrir_configuracion_ptz_individual(self, camera_index):
        """Abrir configuración PTZ para cámara específica"""
        try:
            if camera_index >= len(self.camera_data_list):
                return
                
            camera_data = self.camera_data_list[camera_index]
            camera_ip = camera_data['ip']
            
            if self.ptz_system:
                config_widget = self.ptz_system.open_ptz_config_window(camera_ip)
                if config_widget:
                    self.append_debug(f"🎮 Panel PTZ abierto para {camera_ip}")
                else:
                    QMessageBox.warning(self, "Error", f"No se pudo abrir configuración PTZ para {camera_ip}")
            
        except Exception as e:
            self.append_debug(f"❌ Error abriendo configuración PTZ: {e}")

    def probar_conexion_ptz_individual(self, camera_index):
        """Probar conexión PTZ para cámara específica"""
        try:
            if camera_index >= len(self.camera_data_list):
                return
                
            camera_data = self.camera_data_list[camera_index]
            
            from core.light_api import LightAPI
            
            light_api = LightAPI(
                ip=camera_data['ip'],
                port=camera_data.get('puerto', 80),
                username=camera_data['usuario'],
                password=camera_data['contrasena']
            )
            
            if light_api.test_connection():
                device_info = light_api.get_device_info()
                model = device_info.get("DeviceModel", "Desconocido") if device_info else "Desconocido"
                
                QMessageBox.information(self, "PTZ Conexión OK", 
                    f"✅ Conexión PTZ exitosa\n\n"
                    f"🏷️ Modelo: {model}\n"
                    f"🌐 IP: {camera_data['ip']}\n"
                    f"👤 Usuario: {camera_data['usuario']}")
                
                self.append_debug(f"✅ Test PTZ exitoso: {camera_data['ip']} - {model}")
            else:
                QMessageBox.warning(self, "PTZ Error", 
                    f"❌ No se pudo conectar a PTZ\n\n"
                    f"🌐 IP: {camera_data['ip']}\n"
                    f"👤 Usuario: {camera_data['usuario']}\n\n"
                    f"Verifique:\n"
                    f"• Conectividad de red\n"
                    f"• Credenciales correctas\n"
                    f"• Cámara compatible con LightAPI")
                
                self.append_debug(f"❌ Test PTZ fallido: {camera_data['ip']}")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error PTZ", f"Error probando conexión PTZ:\n\n{str(e)}")
            self.append_debug(f"❌ Error test PTZ: {e}")

    def configure_individual_fps(self, camera_index):
        """Configurar FPS individual para una cámara específica"""
        if camera_index >= len(self.camera_widgets):
            return
            
        widget = self.camera_widgets[camera_index]
        current_fps = widget.fps_config if hasattr(widget, 'fps_config') else self.fps_config
        
        dialog = FPSConfigDialog(self, current_fps)
        dialog.setWindowTitle(f"🎯 FPS para {self.camera_data_list[camera_index].get('ip', 'Cámara')}")
        
        def apply_individual_fps(config):
            widget.set_fps_config(
                visual_fps=config['visual_fps'],
                detection_fps=config['detection_fps'],
                ui_update_fps=config['ui_update_fps']
            )
            self.append_debug(f"🎯 FPS individual aplicado a {widget.cam_data.get('ip', 'Cámara')}")
        
        dialog.fps_config_changed.connect(apply_individual_fps)
        dialog.exec()

    # ===== MÉTODOS PTZ PRINCIPALES =====
    
    def abrir_panel_ptz_principal(self):
        """Abrir panel principal de control PTZ"""
        try:
            if not PTZ_AVAILABLE or not self.ptz_system:
                QMessageBox.warning(self, "PTZ No Disponible", 
                    "❌ Sistema PTZ no está disponible\n\n"
                    "Verifique que los archivos PTZ estén instalados correctamente.")
                return
            
            # Buscar cámaras PTZ configuradas
            ptz_cameras = [(ip, data) for ip, data in zip(
                [cam['ip'] for cam in self.camera_data_list if cam.get('tipo') == 'ptz'],
                [cam for cam in self.camera_data_list if cam.get('tipo') == 'ptz']
            )]
            
            if not ptz_cameras:
                QMessageBox.information(self, "Sin Cámaras PTZ", 
                    "ℹ️ No hay cámaras PTZ configuradas\n\n"
                    "Para agregar una cámara PTZ:\n"
                    "1. Use 'Agregar Cámara' del menú\n"
                    "2. Seleccione tipo 'ptz'\n"
                    "3. Configure IP, usuario y contraseña")
                return
            
            # Si hay una sola cámara PTZ, abrir directamente
            if len(ptz_cameras) == 1:
                camera_ip = ptz_cameras[0][0]
                self.ptz_system.open_ptz_config_window(camera_ip)
                self.append_debug(f"🎮 Panel PTZ abierto para {camera_ip}")
            else:
                # Mostrar selector de cámaras PTZ
                self.mostrar_selector_ptz(ptz_cameras)
                
        except Exception as e:
            QMessageBox.critical(self, "Error PTZ", f"Error abriendo panel PTZ:\n\n{str(e)}")
            self.append_debug(f"❌ Error abriendo panel PTZ: {e}")

    def mostrar_selector_ptz(self, ptz_cameras):
        """Mostrar selector de cámaras PTZ"""
        try:
            from PyQt6.QtWidgets import QDialog, QListWidget, QVBoxLayout, QPushButton, QLabel
            
            dialog = QDialog(self)
            dialog.setWindowTitle("🎮 Seleccionar Cámara PTZ")
            dialog.setFixedSize(400, 300)
            
            layout = QVBoxLayout()
            
            # Título
            title = QLabel("Seleccione la cámara PTZ a configurar:")
            title.setStyleSheet("font-weight: bold; margin: 10px;")
            layout.addWidget(title)
            
            # Lista de cámaras
            camera_list = QListWidget()
            for ip, cam_data in ptz_cameras:
                item_text = f"🎯 {ip} ({cam_data.get('usuario', 'N/A')})"
                camera_list.addItem(item_text)
            
            layout.addWidget(camera_list)
            
            # Botones
            button_layout = QHBoxLayout()
            
            open_btn = QPushButton("🎮 Abrir Panel PTZ")
            open_btn.clicked.connect(lambda: self.abrir_ptz_seleccionado(dialog, camera_list, ptz_cameras))
            
            cancel_btn = QPushButton("❌ Cancelar")
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addWidget(cancel_btn)
            button_layout.addWidget(open_btn)
            
            layout.addLayout(button_layout)
            dialog.setLayout(layout)
            
            dialog.exec()
            
        except Exception as e:
            self.append_debug(f"❌ Error en selector PTZ: {e}")

    def abrir_ptz_seleccionado(self, dialog, camera_list, ptz_cameras):
        """Abrir panel PTZ para cámara seleccionada"""
        try:
            current_row = camera_list.currentRow()
            if current_row >= 0 and current_row < len(ptz_cameras):
                camera_ip = ptz_cameras[current_row][0]
                self.ptz_system.open_ptz_config_window(camera_ip)
                self.append_debug(f"🎮 Panel PTZ abierto para {camera_ip}")
                dialog.accept()
            else:
                QMessageBox.warning(dialog, "Selección", "Por favor seleccione una cámara PTZ")
                
        except Exception as e:
            self.append_debug(f"❌ Error abriendo PTZ seleccionado: {e}")

    def emergency_stop_all_ptz(self):
        """Parada de emergencia para todas las cámaras PTZ"""
        try:
            if not PTZ_AVAILABLE or not self.ptz_system:
                QMessageBox.warning(self, "PTZ No Disponible", "Sistema PTZ no está disponible")
                return
            
            # Ejecutar parada de emergencia
            self.ptz_system.emergency_stop_all()
            
            # Mostrar confirmación
            QMessageBox.warning(self, "Parada de Emergencia", 
                "🚨 PARADA DE EMERGENCIA EJECUTADA\n\n"
                "✅ Todos los movimientos PTZ han sido detenidos\n"
                "✅ Seguimiento automático deshabilitado\n\n"
                "Use el Panel PTZ para reactivar el seguimiento")
            
            self.append_debug("🚨 PARADA DE EMERGENCIA PTZ EJECUTADA")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en parada de emergencia:\n\n{str(e)}")
            self.append_debug(f"❌ Error en parada de emergencia: {e}")

    def mostrar_estado_ptz(self):
        """Mostrar estado completo del sistema PTZ"""
        try:
            if not PTZ_AVAILABLE or not self.ptz_system:
                QMessageBox.information(self, "Estado PTZ", "❌ Sistema PTZ no disponible")
                return
            
            status = self.ptz_system.get_tracking_status()
            
            # Construir mensaje de estado
            estado_msg = "📊 ESTADO DEL SISTEMA PTZ\n"
            estado_msg += "=" * 40 + "\n\n"
            
            if not status:
                estado_msg += "ℹ️ No hay cámaras PTZ configuradas\n\n"
                estado_msg += "Para configurar cámaras PTZ:\n"
                estado_msg += "1. Agregue una cámara tipo 'ptz'\n"
                estado_msg += "2. Use 'Panel PTZ' para configurar seguimiento"
            else:
                # Estadísticas generales
                total_cameras = len(status)
                active_cameras = sum(1 for s in status.values() if s.get("running", False))
                
                estado_msg += f"📹 Cámaras PTZ configuradas: {total_cameras}\n"
                estado_msg += f"🎯 Cámaras en seguimiento activo: {active_cameras}\n"
                estado_msg += f"📊 Estado general: {'🟢 Operativo' if active_cameras > 0 else '🔴 Inactivo'}\n\n"
                
                # Detalle por cámara
                estado_msg += "DETALLE POR CÁMARA:\n"
                estado_msg += "-" * 30 + "\n"
                
                for camera_ip, camera_status in status.items():
                    running = camera_status.get("running", False)
                    target = camera_status.get("current_target", "Ninguno")
                    queue_size = camera_status.get("queue_size", 0)
                    
                    estado_msg += f"\n🎯 {camera_ip}:\n"
                    estado_msg += f"   Estado: {'🟢 Activo' if running else '🔴 Inactivo'}\n"
                    estado_msg += f"   Objetivo actual: {target}\n"
                    estado_msg += f"   Cola de detecciones: {queue_size}\n"
            
            # Mostrar en mensaje
            QMessageBox.information(self, "Estado del Sistema PTZ", estado_msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error obteniendo estado PTZ:\n\n{str(e)}")
            self.append_debug(f"❌ Error obteniendo estado PTZ: {e}")

    def configuracion_rapida_ptz(self):
        """Configuración rápida para nueva cámara PTZ"""
        try:
            from PyQt6.QtWidgets import QDialog, QFormLayout, QLineEdit, QPushButton, QComboBox
            
            dialog = QDialog(self)
            dialog.setWindowTitle("⚙️ Configuración Rápida PTZ")
            dialog.setFixedSize(400, 250)
            
            layout = QFormLayout()
            
            # Campos básicos
            ip_input = QLineEdit()
            ip_input.setPlaceholderText("192.168.1.100")
            
            user_input = QLineEdit()
            user_input.setPlaceholderText("admin")
            
            pass_input = QLineEdit()
            pass_input.setEchoMode(QLineEdit.EchoMode.Password)
            pass_input.setPlaceholderText("contraseña")
            
            tracking_combo = QComboBox()
            tracking_combo.addItems(["Solo Analíticas", "Seguimiento Activo"])
            
            layout.addRow("🌐 IP:", ip_input)
            layout.addRow("👤 Usuario:", user_input)
            layout.addRow("🔒 Contraseña:", pass_input)
            layout.addRow("🎯 Modo:", tracking_combo)
            
            # Botones
            button_layout = QHBoxLayout()
            
            test_btn = QPushButton("🔧 Probar")
            test_btn.clicked.connect(lambda: self.probar_config_rapida(ip_input.text(), user_input.text(), pass_input.text()))
            
            save_btn = QPushButton("💾 Guardar")
            save_btn.clicked.connect(lambda: self.guardar_config_rapida(
                dialog, ip_input.text(), user_input.text(), pass_input.text(), 
                tracking_combo.currentIndex() == 1
            ))
            
            cancel_btn = QPushButton("❌ Cancelar")
            cancel_btn.clicked.connect(dialog.reject)
            
            button_layout.addWidget(test_btn)
            button_layout.addWidget(cancel_btn)
            button_layout.addWidget(save_btn)
            
            layout.addRow(button_layout)
            dialog.setLayout(layout)
            
            dialog.exec()
            
        except Exception as e:
            self.append_debug(f"❌ Error en configuración rápida: {e}")

    def probar_config_rapida(self, ip, user, password):
        """Probar configuración rápida PTZ"""
        try:
            if not all([ip, user, password]):
                QMessageBox.warning(self, "Campos Vacíos", "Complete todos los campos para probar")
                return
            
            from core.light_api import LightAPI
            
            light_api = LightAPI(ip, 80, user, password)
            
            if light_api.test_connection():
                device_info = light_api.get_device_info()
                model = device_info.get("DeviceModel", "Desconocido") if device_info else "Desconocido"
                
                QMessageBox.information(self, "Test Exitoso", 
                    f"✅ Conexión PTZ exitosa\n\n🏷️ Modelo: {model}")
            else:
                QMessageBox.warning(self, "Test Fallido", 
                    "❌ No se pudo conectar\n\nVerifique IP y credenciales")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error en test: {str(e)}")

    def guardar_config_rapida(self, dialog, ip, user, password, tracking_enabled):
        """Guardar configuración rápida PTZ"""
        try:
            if not all([ip, user, password]):
                QMessageBox.warning(self, "Campos Vacíos", "Complete todos los campos")
                return
            
            # Crear datos de cámara
            camera_data = {
                "ip": ip,
                "puerto": 80,
                "usuario": user,
                "contrasena": password,
                "tipo": "ptz",
                "canal": "0",
                "modelo": "Personas",
                "modelos": ["Personas"],
                "confianza": 0.4,
                "intervalo": 40,
                "imgsz": 640,
                "device": "cuda",
                "resolucion": "main",
                "umbral": 0.5,
                "guardar_capturas": True,
                "modo_centinela": False
            }
            
            # Agregar a lista de cámaras
            self.camera_data_list.append(camera_data)
            self.camera_list.addItem(f"{ip} - ptz")
            
            # Iniciar stream
            self.start_camera_stream(camera_data)
            
            # Agregar al sistema PTZ
            if PTZ_AVAILABLE:
                ptz_config = {
                    "ip": ip,
                    "port": 80,
                    "username": user,
                    "password": password,
                    "tracking_mode": "tracking" if tracking_enabled else "analytics_only",
                    "tracking_enabled": tracking_enabled
                }
                
                self.ptz_system.add_ptz_camera(ptz_config)
            
            # Guardar configuración
            guardar_camaras(self)
            
            self.append_debug(f"⚙️ Cámara PTZ {ip} configurada rápidamente")
            
            dialog.accept()
            
            # Mostrar mensaje de éxito
            QMessageBox.information(self, "Configuración Guardada", 
                f"✅ Cámara PTZ {ip} configurada\n\n"
                f"🎯 Seguimiento: {'Activo' if tracking_enabled else 'Solo Analíticas'}\n\n"
                f"Use 'Panel PTZ' para configuración avanzada")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error guardando configuración:\n\n{str(e)}")
            self.append_debug(f"❌ Error en configuración rápida: {e}")

    def update_ptz_status(self):
        """Actualizar indicador de estado PTZ"""
        try:
            if not PTZ_AVAILABLE or not self.ptz_system:
                self.lbl_ptz_status.setText("PTZ: No disponible")
                self.lbl_ptz_status.setStyleSheet("""
                    QLabel {
                        border: 1px solid #f44336;
                        padding: 8px;
                        border-radius: 4px;
                        background-color: #ffebee;
                        color: #f44336;
                    }
                """)
                return
            
            status = self.ptz_system.get_tracking_status()
            
            if not status:
                self.lbl_ptz_status.setText("PTZ: Sin cámaras")
                self.lbl_ptz_status.setStyleSheet("""
                    QLabel {
                        border: 1px solid #ff9800;
                        padding: 8px;
                        border-radius: 4px;
                        background-color: #fff3e0;
                        color: #ff9800;
                    }
                """)
                return
            
            active_cameras = sum(1 for s in status.values() if s.get("running", False))
            total_cameras = len(status)
            
            if active_cameras > 0:
                self.lbl_ptz_status.setText(f"PTZ: {active_cameras}/{total_cameras} activas")
                self.lbl_ptz_status.setStyleSheet("""
                    QLabel {
                        border: 1px solid #4caf50;
                        padding: 8px;
                        border-radius: 4px;
                        background-color: #e8f5e8;
                        color: #4caf50;
                    }
                """)
            else:
                self.lbl_ptz_status.setText(f"PTZ: {total_cameras} configuradas")
                self.lbl_ptz_status.setStyleSheet("""
                    QLabel {
                        border: 1px solid #2196f3;
                        padding: 8px;
                        border-radius: 4px;
                        background-color: #e3f2fd;
                        color: #2196f3;
                    }
                """)
                
        except Exception as e:
            self.lbl_ptz_status.setText("PTZ: Error")
            self.lbl_ptz_status.setStyleSheet("""
                QLabel {
                    border: 1px solid #f44336;
                    padding: 8px;
                    border-radius: 4px;
                    background-color: #ffebee;
                    color: #f44336;
                }
            """)

    # ===== FIN MÉTODOS PTZ =====

    def restart_all_cameras(self):
        for widget in list(self.camera_widgets):
            try:
                if hasattr(widget, 'detener') and callable(widget.detener):
                    widget.detener()
                self.video_grid.removeWidget(widget)
                widget.deleteLater()
            except Exception as e:
                print(f"ERROR al detener cámara: {e}")
        self.camera_widgets.clear()
        for cam in self.camera_data_list:
            self.start_camera_stream(cam)
        self.append_debug("🔄 Cámaras reiniciadas con nueva configuración")

    def closeEvent(self, event):
        print("INFO: Iniciando proceso de cierre de MainGUI...")
        
        # === CERRAR SISTEMA PTZ ===
        if PTZ_AVAILABLE and self.ptz_system:
            try:
                print("INFO: Cerrando sistema PTZ...")
                self.ptz_system.save_configuration()
                self.ptz_system.shutdown()
                print("INFO: Sistema PTZ cerrado correctamente")
            except Exception as e:
                print(f"ERROR: Error cerrando sistema PTZ: {e}")
        # === FIN CIERRE PTZ ===
        
        print(f"INFO: Deteniendo {len(self.camera_widgets)} widgets de cámara activos...")
        for widget in self.camera_widgets:
            try:
                if hasattr(widget, 'detener') and callable(widget.detener):
                    cam_ip = "N/A"
                    if hasattr(widget, 'cam_data') and widget.cam_data:
                        cam_ip = widget.cam_data.get('ip', 'N/A')
                    print(f"INFO: Llamando a detener() para el widget de la cámara IP: {cam_ip}")
                    widget.detener()
                else:
                    cam_ip_info = "N/A"
                    if hasattr(widget, 'cam_data') and widget.cam_data:
                         cam_ip_info = widget.cam_data.get('ip', 'N/A')
                    print(f"WARN: El widget para IP {cam_ip_info} no tiene el método detener() o no es llamable.")
            except Exception as e:
                cam_ip_err = "N/A"
                if hasattr(widget, 'cam_data') and widget.cam_data:
                    cam_ip_err = widget.cam_data.get('ip', 'N/A')
                print(f"ERROR: Excepción al detener widget para IP {cam_ip_err}: {e}")
        
        if hasattr(self, 'resumen_widget') and self.resumen_widget: 
            if hasattr(self.resumen_widget, 'stop_threads') and callable(self.resumen_widget.stop_threads):
                print("INFO: Llamando a stop_threads() para resumen_widget...")
                try:
                    self.resumen_widget.stop_threads()
                except Exception as e:
                    print(f"ERROR: Excepción al llamar a stop_threads() en resumen_widget: {e}")
            else:
                print("WARN: resumen_widget no tiene el método stop_threads() o no es llamable.")
        else:
            print("WARN: self.resumen_widget no existe, no se pueden detener sus hilos.")

        # Profiling logic
        print("INFO: Deteniendo profiler y guardando estadísticas...")
        self.profiler.disable()
        stats_filename = "main_gui_profile.prof"
        self.profiler.dump_stats(stats_filename)
        print(f"INFO: Resultados del profiler guardados en {stats_filename}")

        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative', 'tottime')
        ps.print_stats(30)
        print("\n--- Resumen del Profiler (Top 30 por tiempo acumulado) ---")
        print(s.getvalue())
        print("--- Fin del Resumen del Profiler ---\n")

        print("INFO: Proceso de cierre de MainGUI completado. Aceptando evento.")
        event.accept()