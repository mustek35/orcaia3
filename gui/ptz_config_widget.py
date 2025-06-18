"""
Widget de Configuración PTZ Profesional
Interfaz completa para configurar y controlar cámaras PTZ
"""

import sys
import json
import time
from typing import Dict, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QCheckBox, QGroupBox, QTabWidget, QSlider,
    QTextEdit, QListWidget, QListWidgetItem, QProgressBar,
    QSplitter, QFrame, QScrollArea, QMessageBox, QFileDialog
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor

from core.light_api import LightAPI, PTZDirection, ZoomDirection, PresetInfo
from core.ptz_tracking_system import PTZTrackingMode, PTZCameraConfig
from gui.grilla_widget import GrillaWidget

class PTZControlThread(QThread):
    """Hilo para operaciones PTZ que pueden tomar tiempo"""
    
    operation_completed = pyqtSignal(bool, str)  # success, message
    image_captured = pyqtSignal(bytes)  # image_data
    
    def __init__(self, light_api: LightAPI, operation: str, **kwargs):
        super().__init__()
        self.light_api = light_api
        self.operation = operation
        self.kwargs = kwargs
    
    def run(self):
        try:
            if self.operation == "capture_image":
                image_data = self.light_api.capture_snapshot()
                if image_data:
                    self.image_captured.emit(image_data)
                    self.operation_completed.emit(True, "Imagen capturada exitosamente")
                else:
                    self.operation_completed.emit(False, "Error capturando imagen")
            
            elif self.operation == "create_preset":
                success = self.light_api.create_preset(
                    self.kwargs.get("preset_id"),
                    self.kwargs.get("preset_name")
                )
                message = "Preset creado exitosamente" if success else "Error creando preset"
                self.operation_completed.emit(success, message)
            
            elif self.operation == "goto_preset":
                success = self.light_api.goto_preset(self.kwargs.get("preset_id"))
                message = "Movido a preset exitosamente" if success else "Error moviendo a preset"
                self.operation_completed.emit(success, message)
            
            elif self.operation == "test_connection":
                success = self.light_api.test_connection()
                message = "Conexión exitosa" if success else "Error de conexión"
                self.operation_completed.emit(success, message)
                
        except Exception as e:
            self.operation_completed.emit(False, f"Error: {str(e)}")

class PTZConfigWidget(QWidget):
    """Widget principal de configuración PTZ"""
    
    # Señales
    configuration_changed = pyqtSignal(dict)
    tracking_toggled = pyqtSignal(str, bool)  # camera_ip, enabled
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.camera_config: Optional[PTZCameraConfig] = None
        self.light_api: Optional[LightAPI] = None
        self.movement_timer = QTimer()
        self.movement_timer.timeout.connect(self.stop_movement)
        
        self.setup_ui()
        self.setup_connections()
        
    def setup_ui(self):
        """Configurar la interfaz de usuario"""
        self.setWindowTitle("Configuración PTZ Profesional")
        self.setMinimumSize(1200, 800)
        
        # Layout principal
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Splitter principal
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Panel izquierdo - Configuración
        config_widget = self.create_config_panel()
        splitter.addWidget(config_widget)
        
        # Panel derecho - Control y monitoreo
        control_widget = self.create_control_panel()
        splitter.addWidget(control_widget)
        
        # Proporciones del splitter
        splitter.setSizes([500, 700])
    
    def create_config_panel(self) -> QWidget:
        """Crear panel de configuración"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Título
        title = QLabel("Configuración PTZ")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Scroll area para configuración
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(480)
        
        config_content = QWidget()
        config_layout = QVBoxLayout()
        config_content.setLayout(config_layout)
        
        # Sección de conexión
        connection_group = self.create_connection_group()
        config_layout.addWidget(connection_group)
        
        # Sección de seguimiento
        tracking_group = self.create_tracking_group()
        config_layout.addWidget(tracking_group)
        
        # Sección de zona de seguimiento
        zone_group = self.create_tracking_zone_group()
        config_layout.addWidget(zone_group)
        
        # Botones de acción
        action_buttons = self.create_action_buttons()
        config_layout.addWidget(action_buttons)
        
        config_layout.addStretch()
        
        scroll.setWidget(config_content)
        layout.addWidget(scroll)
        
        return widget
    
    def create_connection_group(self) -> QGroupBox:
        """Crear grupo de configuración de conexión"""
        group = QGroupBox("Configuración de Conexión")
        layout = QFormLayout()
        
        # Campos de conexión
        self.ip_input = QLineEdit()
        self.ip_input.setPlaceholderText("192.168.1.100")
        
        self.port_input = QSpinBox()
        self.port_input.setRange(1, 65535)
        self.port_input.setValue(80)
        
        self.username_input = QLineEdit()
        self.username_input.setPlaceholderText("admin")
        
        self.password_input = QLineEdit()
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        
        # Botón de prueba de conexión
        self.test_connection_btn = QPushButton("Probar Conexión")
        self.test_connection_btn.clicked.connect(self.test_connection)
        
        # Estado de conexión
        self.connection_status = QLabel("No conectado")
        self.connection_status.setStyleSheet("color: red;")
        
        layout.addRow("IP:", self.ip_input)
        layout.addRow("Puerto:", self.port_input)
        layout.addRow("Usuario:", self.username_input)
        layout.addRow("Contraseña:", self.password_input)
        layout.addRow("", self.test_connection_btn)
        layout.addRow("Estado:", self.connection_status)
        
        group.setLayout(layout)
        return group
    
    def create_tracking_group(self) -> QGroupBox:
        """Crear grupo de configuración de seguimiento"""
        group = QGroupBox("Configuración de Seguimiento")
        layout = QFormLayout()
        
        # Modo de seguimiento
        self.tracking_mode_combo = QComboBox()
        self.tracking_mode_combo.addItems([
            "Deshabilitado",
            "Seguimiento Activo", 
            "Solo Analíticas"
        ])
        
        # Habilitar seguimiento
        self.tracking_enabled_checkbox = QCheckBox("Habilitar Seguimiento")
        
        # Parámetros de seguimiento
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 100)
        self.sensitivity_slider.setValue(50)
        self.sensitivity_label = QLabel("0.005")
        
        self.max_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_speed_slider.setRange(1, 100)
        self.max_speed_slider.setValue(50)
        self.max_speed_label = QLabel("0.5")
        
        self.deadzone_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.deadzone_x_slider.setRange(1, 50)
        self.deadzone_x_slider.setValue(3)
        self.deadzone_x_label = QLabel("0.03")
        
        self.deadzone_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.deadzone_y_slider.setRange(1, 50)
        self.deadzone_y_slider.setValue(3)
        self.deadzone_y_label = QLabel("0.03")
        
        self.confirmation_frames_spin = QSpinBox()
        self.confirmation_frames_spin.setRange(1, 10)
        self.confirmation_frames_spin.setValue(3)
        
        # Conectar sliders a labels
        self.sensitivity_slider.valueChanged.connect(
            lambda v: self.sensitivity_label.setText(f"{v * 0.0001:.4f}")
        )
        self.max_speed_slider.valueChanged.connect(
            lambda v: self.max_speed_label.setText(f"{v * 0.01:.2f}")
        )
        self.deadzone_x_slider.valueChanged.connect(
            lambda v: self.deadzone_x_label.setText(f"{v * 0.01:.2f}")
        )
        self.deadzone_y_slider.valueChanged.connect(
            lambda v: self.deadzone_y_label.setText(f"{v * 0.01:.2f}")
        )
        
        # Layouts para sliders con labels
        sens_layout = QHBoxLayout()
        sens_layout.addWidget(self.sensitivity_slider)
        sens_layout.addWidget(self.sensitivity_label)
        
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(self.max_speed_slider)
        speed_layout.addWidget(self.max_speed_label)
        
        deadzone_x_layout = QHBoxLayout()
        deadzone_x_layout.addWidget(self.deadzone_x_slider)
        deadzone_x_layout.addWidget(self.deadzone_x_label)
        
        deadzone_y_layout = QHBoxLayout()
        deadzone_y_layout.addWidget(self.deadzone_y_slider)
        deadzone_y_layout.addWidget(self.deadzone_y_label)
        
        layout.addRow("Modo:", self.tracking_mode_combo)
        layout.addRow("", self.tracking_enabled_checkbox)
        layout.addRow("Sensibilidad:", sens_layout)
        layout.addRow("Velocidad Máx:", speed_layout)
        layout.addRow("Zona Muerta X:", deadzone_x_layout)
        layout.addRow("Zona Muerta Y:", deadzone_y_layout)
        layout.addRow("Frames Confirmación:", self.confirmation_frames_spin)
        
        group.setLayout(layout)
        return group
    
    def create_tracking_zone_group(self) -> QGroupBox:
        """Crear grupo de zona de seguimiento"""
        group = QGroupBox("Zona de Seguimiento")
        layout = QVBoxLayout()
        
        # Información
        info_label = QLabel("Seleccione las celdas de la grilla donde se activará el seguimiento:")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Grilla de seguimiento (placeholder - se conectará con la grilla real)
        self.tracking_grid = GrillaWidget()
        self.tracking_grid.setFixedSize(320, 240)
        self.tracking_grid.setStyleSheet("border: 1px solid gray;")
        layout.addWidget(self.tracking_grid)
        
        # Botones de grilla
        grid_buttons_layout = QHBoxLayout()
        
        self.select_all_btn = QPushButton("Seleccionar Todo")
        self.select_all_btn.clicked.connect(self.select_all_grid)
        
        self.clear_all_btn = QPushButton("Limpiar Todo")
        self.clear_all_btn.clicked.connect(self.clear_all_grid)
        
        grid_buttons_layout.addWidget(self.select_all_btn)
        grid_buttons_layout.addWidget(self.clear_all_btn)
        
        layout.addLayout(grid_buttons_layout)
        
        group.setLayout(layout)
        return group
    
    def create_action_buttons(self) -> QWidget:
        """Crear botones de acción"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Botones principales
        self.save_config_btn = QPushButton("Guardar Configuración")
        self.save_config_btn.clicked.connect(self.save_configuration)
        self.save_config_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.load_config_btn = QPushButton("Cargar Configuración")
        self.load_config_btn.clicked.connect(self.load_configuration)
        
        layout.addWidget(self.save_config_btn)
        layout.addWidget(self.load_config_btn)
        
        widget.setLayout(layout)
        return widget
    
    def create_control_panel(self) -> QWidget:
        """Crear panel de control"""
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        # Título
        title = QLabel("Control PTZ")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Tabs de control
        tabs = QTabWidget()
        
        # Tab de movimiento
        movement_tab = self.create_movement_tab()
        tabs.addTab(movement_tab, "Movimiento")
        
        # Tab de presets
        presets_tab = self.create_presets_tab()
        tabs.addTab(presets_tab, "Presets")
        
        # Tab de funciones especiales
        special_tab = self.create_special_functions_tab()
        tabs.addTab(special_tab, "Funciones")
        
        # Tab de monitoreo
        monitoring_tab = self.create_monitoring_tab()
        tabs.addTab(monitoring_tab, "Monitoreo")
        
        layout.addWidget(tabs)
        
        return widget
    
    def create_movement_tab(self) -> QWidget:
        """Crear tab de control de movimiento"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Control direccional
        direction_group = QGroupBox("Control Direccional")
        direction_layout = QGridLayout()
        
        # Botones direccionales
        self.up_btn = QPushButton("↑")
        self.down_btn = QPushButton("↓")
        self.left_btn = QPushButton("←")
        self.right_btn = QPushButton("→")
        self.up_left_btn = QPushButton("↖")
        self.up_right_btn = QPushButton("↗")
        self.down_left_btn = QPushButton("↙")
        self.down_right_btn = QPushButton("↘")
        self.stop_btn = QPushButton("STOP")
        
        # Estilo para botones direccionales
        btn_style = """
            QPushButton {
                font-size: 18px;
                font-weight: bold;
                min-width: 50px;
                min-height: 50px;
                border: 2px solid #ddd;
                border-radius: 25px;
                background-color: #f0f0f0;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
                border-color: #bbb;
            }
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
        """
        
        for btn in [self.up_btn, self.down_btn, self.left_btn, self.right_btn,
                   self.up_left_btn, self.up_right_btn, self.down_left_btn, 
                   self.down_right_btn]:
            btn.setStyleSheet(btn_style)
        
        self.stop_btn.setStyleSheet(btn_style + """
            QPushButton {
                background-color: #ff4444;
                color: white;
                border-color: #cc0000;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
        """)
        
        # Disposición de botones
        direction_layout.addWidget(self.up_left_btn, 0, 0)
        direction_layout.addWidget(self.up_btn, 0, 1)
        direction_layout.addWidget(self.up_right_btn, 0, 2)
        direction_layout.addWidget(self.left_btn, 1, 0)
        direction_layout.addWidget(self.stop_btn, 1, 1)
        direction_layout.addWidget(self.right_btn, 1, 2)
        direction_layout.addWidget(self.down_left_btn, 2, 0)
        direction_layout.addWidget(self.down_btn, 2, 1)
        direction_layout.addWidget(self.down_right_btn, 2, 2)
        
        direction_group.setLayout(direction_layout)
        layout.addWidget(direction_group)
        
        # Control de zoom
        zoom_group = QGroupBox("Control de Zoom")
        zoom_layout = QHBoxLayout()
        
        self.zoom_in_btn = QPushButton("Zoom +")
        self.zoom_out_btn = QPushButton("Zoom -")
        self.zoom_stop_btn = QPushButton("Stop Zoom")
        
        zoom_layout.addWidget(self.zoom_in_btn)
        zoom_layout.addWidget(self.zoom_out_btn)
        zoom_layout.addWidget(self.zoom_stop_btn)
        
        zoom_group.setLayout(zoom_layout)
        layout.addWidget(zoom_group)
        
        # Control de velocidad
        speed_group = QGroupBox("Velocidad de Movimiento")
        speed_layout = QFormLayout()
        
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(1, 100)
        self.speed_slider.setValue(50)
        self.speed_value_label = QLabel("0.5")
        
        speed_with_label = QHBoxLayout()
        speed_with_label.addWidget(self.speed_slider)
        speed_with_label.addWidget(self.speed_value_label)
        
        self.speed_slider.valueChanged.connect(
            lambda v: self.speed_value_label.setText(f"{v * 0.01:.2f}")
        )
        
        speed_layout.addRow("Velocidad:", speed_with_label)
        speed_group.setLayout(speed_layout)
        layout.addWidget(speed_group)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_presets_tab(self) -> QWidget:
        """Crear tab de presets"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Lista de presets
        presets_group = QGroupBox("Presets Disponibles")
        presets_layout = QVBoxLayout()
        
        self.presets_list = QListWidget()
        self.presets_list.setMinimumHeight(200)
        presets_layout.addWidget(self.presets_list)
        
        # Botones de preset
        preset_buttons_layout = QHBoxLayout()
        
        self.refresh_presets_btn = QPushButton("Actualizar")
        self.goto_preset_btn = QPushButton("Ir a Preset")
        self.delete_preset_btn = QPushButton("Eliminar")
        
        preset_buttons_layout.addWidget(self.refresh_presets_btn)
        preset_buttons_layout.addWidget(self.goto_preset_btn)
        preset_buttons_layout.addWidget(self.delete_preset_btn)
        
        presets_layout.addLayout(preset_buttons_layout)
        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)
        
        # Crear nuevo preset
        create_preset_group = QGroupBox("Crear Nuevo Preset")
        create_layout = QFormLayout()
        
        self.preset_id_spin = QSpinBox()
        self.preset_id_spin.setRange(1, 255)
        
        self.preset_name_input = QLineEdit()
        self.preset_name_input.setPlaceholderText("Nombre del preset")
        
        self.create_preset_btn = QPushButton("Crear Preset")
        
        create_layout.addRow("ID:", self.preset_id_spin)
        create_layout.addRow("Nombre:", self.preset_name_input)
        create_layout.addRow("", self.create_preset_btn)
        
        create_preset_group.setLayout(create_layout)
        layout.addWidget(create_preset_group)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_special_functions_tab(self) -> QWidget:
        """Crear tab de funciones especiales"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Captura de imagen
        capture_group = QGroupBox("Captura de Imagen")
        capture_layout = QVBoxLayout()
        
        self.capture_btn = QPushButton("Capturar Imagen")
        self.capture_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        
        # Preview de imagen
        self.image_preview = QLabel("No hay imagen")
        self.image_preview.setMinimumSize(300, 200)
        self.image_preview.setStyleSheet("border: 1px solid gray;")
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_preview.setScaledContents(True)
        
        capture_layout.addWidget(self.capture_btn)
        capture_layout.addWidget(self.image_preview)
        
        capture_group.setLayout(capture_layout)
        layout.addWidget(capture_group)
        
        # Funciones especiales
        special_group = QGroupBox("Funciones Especiales")
        special_layout = QGridLayout()
        
        self.wiper_btn = QPushButton("Activar Limpiaparabrisas")
        self.defog_on_btn = QPushButton("Defog ON")
        self.defog_off_btn = QPushButton("Defog OFF")
        
        special_layout.addWidget(self.wiper_btn, 0, 0)
        special_layout.addWidget(self.defog_on_btn, 0, 1)
        special_layout.addWidget(self.defog_off_btn, 1, 0)
        
        special_group.setLayout(special_layout)
        layout.addWidget(special_group)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def create_monitoring_tab(self) -> QWidget:
        """Crear tab de monitoreo"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Estado actual
        status_group = QGroupBox("Estado Actual")
        status_layout = QFormLayout()
        
        self.current_position_label = QLabel("Desconocida")
        self.tracking_status_label = QLabel("Deshabilitado")
        self.api_status_label = QLabel("Desconectado")
        
        status_layout.addRow("Posición Actual:", self.current_position_label)
        status_layout.addRow("Estado Seguimiento:", self.tracking_status_label)
        status_layout.addRow("Estado API:", self.api_status_label)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Log de actividad
        log_group = QGroupBox("Log de Actividad")
        log_layout = QVBoxLayout()
        
        self.activity_log = QTextEdit()
        self.activity_log.setReadOnly(True)
        self.activity_log.setMaximumHeight(200)
        
        log_buttons_layout = QHBoxLayout()
        self.clear_log_btn = QPushButton("Limpiar Log")
        self.save_log_btn = QPushButton("Guardar Log")
        
        log_buttons_layout.addWidget(self.clear_log_btn)
        log_buttons_layout.addWidget(self.save_log_btn)
        
        log_layout.addWidget(self.activity_log)
        log_layout.addLayout(log_buttons_layout)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    def setup_connections(self):
        """Configurar conexiones de señales"""
        
        # Botones de movimiento
        self.up_btn.pressed.connect(lambda: self.start_movement(PTZDirection.UP))
        self.up_btn.released.connect(self.stop_movement)
        
        self.down_btn.pressed.connect(lambda: self.start_movement(PTZDirection.DOWN))
        self.down_btn.released.connect(self.stop_movement)
        
        self.left_btn.pressed.connect(lambda: self.start_movement(PTZDirection.LEFT))
        self.left_btn.released.connect(self.stop_movement)
        
        self.right_btn.pressed.connect(lambda: self.start_movement(PTZDirection.RIGHT))
        self.right_btn.released.connect(self.stop_movement)
        
        self.up_left_btn.pressed.connect(lambda: self.start_movement(PTZDirection.UP_LEFT))
        self.up_left_btn.released.connect(self.stop_movement)
        
        self.up_right_btn.pressed.connect(lambda: self.start_movement(PTZDirection.UP_RIGHT))
        self.up_right_btn.released.connect(self.stop_movement)
        
        self.down_left_btn.pressed.connect(lambda: self.start_movement(PTZDirection.DOWN_LEFT))
        self.down_left_btn.released.connect(self.stop_movement)
        
        self.down_right_btn.pressed.connect(lambda: self.start_movement(PTZDirection.DOWN_RIGHT))
        self.down_right_btn.released.connect(self.stop_movement)
        
        self.stop_btn.clicked.connect(self.stop_movement)
        
        # Botones de zoom
        self.zoom_in_btn.pressed.connect(lambda: self.start_zoom(ZoomDirection.IN))
        self.zoom_in_btn.released.connect(self.stop_zoom)
        
        self.zoom_out_btn.pressed.connect(lambda: self.start_zoom(ZoomDirection.OUT))
        self.zoom_out_btn.released.connect(self.stop_zoom)
        
        self.zoom_stop_btn.clicked.connect(self.stop_zoom)
        
        # Presets
        self.refresh_presets_btn.clicked.connect(self.refresh_presets)
        self.goto_preset_btn.clicked.connect(self.goto_selected_preset)
        self.create_preset_btn.clicked.connect(self.create_new_preset)
        self.delete_preset_btn.clicked.connect(self.delete_selected_preset)
        
        # Funciones especiales
        self.capture_btn.clicked.connect(self.capture_image)
        self.wiper_btn.clicked.connect(self.activate_wiper)
        self.defog_on_btn.clicked.connect(lambda: self.set_defog(True))
        self.defog_off_btn.clicked.connect(lambda: self.set_defog(False))
        
        # Log
        self.clear_log_btn.clicked.connect(self.clear_activity_log)
        self.save_log_btn.clicked.connect(self.save_activity_log)
        
        # Configuración
        self.tracking_enabled_checkbox.stateChanged.connect(self.on_tracking_enabled_changed)
        
        # Timer para actualizar estado
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(2000)  # Actualizar cada 2 segundos
    
    def set_camera_config(self, config: PTZCameraConfig):
        """Establecer configuración de cámara"""
        self.camera_config = config
        
        # Inicializar API Light
        self.light_api = LightAPI(
            ip=config.ip,
            port=config.port,
            username=config.username,
            password=config.password
        )
        
        # Cargar configuración en la UI
        self.load_config_to_ui(config)
        
        # Actualizar estado
        self.update_status()
        
        self.log_activity(f"Configuración cargada para cámara {config.ip}")
    
    def load_config_to_ui(self, config: PTZCameraConfig):
        """Cargar configuración en la interfaz"""
        
        # Configuración de conexión
        self.ip_input.setText(config.ip)
        self.port_input.setValue(config.port)
        self.username_input.setText(config.username)
        self.password_input.setText(config.password)
        
        # Configuración de seguimiento
        mode_index = {
            PTZTrackingMode.DISABLED: 0,
            PTZTrackingMode.TRACKING: 1,
            PTZTrackingMode.ANALYTICS_ONLY: 2
        }.get(config.tracking_mode, 0)
        
        self.tracking_mode_combo.setCurrentIndex(mode_index)
        self.tracking_enabled_checkbox.setChecked(config.tracking_enabled)
        
        # Parámetros de seguimiento
        self.sensitivity_slider.setValue(int(config.tracking_sensitivity * 10000))
        self.max_speed_slider.setValue(int(config.max_pt_speed * 100))
        self.deadzone_x_slider.setValue(int(config.deadzone_x * 100))
        self.deadzone_y_slider.setValue(int(config.deadzone_y * 100))
        self.confirmation_frames_spin.setValue(config.confirmation_frames)
        
        # Zona de seguimiento
        if config.tracking_grid_cells:
            self.tracking_grid.set_selected_cells(config.tracking_grid_cells)
    
    def get_config_from_ui(self) -> PTZCameraConfig:
        """Obtener configuración desde la interfaz"""
        
        # Modo de seguimiento
        mode_map = {
            0: PTZTrackingMode.DISABLED,
            1: PTZTrackingMode.TRACKING,
            2: PTZTrackingMode.ANALYTICS_ONLY
        }
        
        config = PTZCameraConfig(
            ip=self.ip_input.text(),
            port=self.port_input.value(),
            username=self.username_input.text(),
            password=self.password_input.text(),
            tracking_mode=mode_map.get(self.tracking_mode_combo.currentIndex(), PTZTrackingMode.DISABLED),
            tracking_enabled=self.tracking_enabled_checkbox.isChecked(),
            tracking_sensitivity=self.sensitivity_slider.value() * 0.0001,
            max_pt_speed=self.max_speed_slider.value() * 0.01,
            deadzone_x=self.deadzone_x_slider.value() * 0.01,
            deadzone_y=self.deadzone_y_slider.value() * 0.01,
            confirmation_frames=self.confirmation_frames_spin.value(),
            tracking_grid_cells=self.tracking_grid.get_selected_cells()
        )
        
        return config
    
    # =============================================================================
    # MÉTODOS DE CONTROL PTZ
    # =============================================================================
    
    def start_movement(self, direction: PTZDirection):
        """Iniciar movimiento en una dirección"""
        if not self.light_api:
            self.log_activity("Error: No hay conexión API configurada")
            return
        
        speed = self.speed_slider.value() * 0.01
        success = self.light_api.move_direction(direction, speed)
        
        if success:
            self.log_activity(f"Movimiento iniciado: {direction.value} (velocidad: {speed:.2f})")
        else:
            self.log_activity(f"Error iniciando movimiento: {direction.value}")
    
    def stop_movement(self):
        """Detener movimiento"""
        if not self.light_api:
            return
        
        success = self.light_api.stop_movement()
        if success:
            self.log_activity("Movimiento detenido")
        else:
            self.log_activity("Error deteniendo movimiento")
    
    def start_zoom(self, direction: ZoomDirection):
        """Iniciar zoom"""
        if not self.light_api:
            self.log_activity("Error: No hay conexión API configurada")
            return
        
        speed = self.speed_slider.value() * 0.01
        
        if direction == ZoomDirection.IN:
            success = self.light_api.zoom_in(speed)
        elif direction == ZoomDirection.OUT:
            success = self.light_api.zoom_out(speed)
        else:
            success = self.light_api.zoom_stop()
        
        if success:
            self.log_activity(f"Zoom {direction.value} iniciado")
        else:
            self.log_activity(f"Error en zoom {direction.value}")
    
    def stop_zoom(self):
        """Detener zoom"""
        if not self.light_api:
            return
        
        success = self.light_api.zoom_stop()
        if success:
            self.log_activity("Zoom detenido")
        else:
            self.log_activity("Error deteniendo zoom")
    
    # =============================================================================
    # MÉTODOS DE PRESETS
    # =============================================================================
    
    def refresh_presets(self):
        """Actualizar lista de presets"""
        if not self.light_api:
            self.log_activity("Error: No hay conexión API configurada")
            return
        
        self.log_activity("Actualizando lista de presets...")
        
        presets = self.light_api.get_presets(force_refresh=True)
        
        self.presets_list.clear()
        
        for preset in presets:
            item = QListWidgetItem(f"{preset.id}: {preset.name}")
            item.setData(Qt.ItemDataRole.UserRole, preset.id)
            self.presets_list.addItem(item)
        
        self.log_activity(f"Se encontraron {len(presets)} presets")
    
    def goto_selected_preset(self):
        """Ir al preset seleccionado"""
        current_item = self.presets_list.currentItem()
        if not current_item:
            self.log_activity("Error: No hay preset seleccionado")
            return
        
        preset_id = current_item.data(Qt.ItemDataRole.UserRole)
        
        self.log_activity(f"Moviendo a preset {preset_id}...")
        
        # Usar hilo para operación
        self.ptz_thread = PTZControlThread(self.light_api, "goto_preset", preset_id=preset_id)
        self.ptz_thread.operation_completed.connect(self.on_operation_completed)
        self.ptz_thread.start()
    
    def create_new_preset(self):
        """Crear nuevo preset"""
        if not self.light_api:
            self.log_activity("Error: No hay conexión API configurada")
            return
        
        preset_id = self.preset_id_spin.value()
        preset_name = self.preset_name_input.text().strip()
        
        if not preset_name:
            self.log_activity("Error: Ingrese un nombre para el preset")
            return
        
        self.log_activity(f"Creando preset {preset_id}: {preset_name}...")
        
        # Usar hilo para operación
        self.ptz_thread = PTZControlThread(
            self.light_api, 
            "create_preset", 
            preset_id=preset_id, 
            preset_name=preset_name
        )
        self.ptz_thread.operation_completed.connect(self.on_preset_created)
        self.ptz_thread.start()
    
    def delete_selected_preset(self):
        """Eliminar preset seleccionado"""
        current_item = self.presets_list.currentItem()
        if not current_item:
            self.log_activity("Error: No hay preset seleccionado")
            return
        
        preset_id = current_item.data(Qt.ItemDataRole.UserRole)
        
        # Confirmar eliminación
        reply = QMessageBox.question(
            self, 
            "Confirmar Eliminación",
            f"¿Está seguro de eliminar el preset {preset_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            success = self.light_api.delete_preset(preset_id)
            
            if success:
                self.log_activity(f"Preset {preset_id} eliminado")
                self.refresh_presets()
            else:
                self.log_activity(f"Error eliminando preset {preset_id}")
    
    # =============================================================================
    # FUNCIONES ESPECIALES
    # =============================================================================
    
    def capture_image(self):
        """Capturar imagen"""
        if not self.light_api:
            self.log_activity("Error: No hay conexión API configurada")
            return
        
        self.log_activity("Capturando imagen...")
        self.capture_btn.setEnabled(False)
        
        # Usar hilo para operación
        self.ptz_thread = PTZControlThread(self.light_api, "capture_image")
        self.ptz_thread.operation_completed.connect(self.on_image_capture_completed)
        self.ptz_thread.image_captured.connect(self.on_image_captured)
        self.ptz_thread.start()
    
    def activate_wiper(self):
        """Activar limpiaparabrisas"""
        if not self.light_api:
            self.log_activity("Error: No hay conexión API configurada")
            return
        
        success = self.light_api.enable_wiper()
        
        if success:
            self.log_activity("Limpiaparabrisas activado")
        else:
            self.log_activity("Error activando limpiaparabrisas")
    
    def set_defog(self, enable: bool):
        """Configurar modo defog"""
        if not self.light_api:
            self.log_activity("Error: No hay conexión API configurada")
            return
        
        success = self.light_api.set_defog_mode(enable)
        
        action = "activado" if enable else "desactivado"
        if success:
            self.log_activity(f"Defog {action}")
        else:
            self.log_activity(f"Error {'activando' if enable else 'desactivando'} defog")
    
    # =============================================================================
    # MÉTODOS DE CONFIGURACIÓN
    # =============================================================================
    
    def test_connection(self):
        """Probar conexión con la cámara"""
        ip = self.ip_input.text()
        port = self.port_input.value()
        username = self.username_input.text()
        password = self.password_input.text()
        
        if not ip:
            self.log_activity("Error: Ingrese una IP")
            return
        
        self.log_activity(f"Probando conexión con {ip}:{port}...")
        self.test_connection_btn.setEnabled(False)
        
        # Crear API temporal para prueba
        test_api = LightAPI(ip, port, username, password)
        
        # Usar hilo para operación
        self.ptz_thread = PTZControlThread(test_api, "test_connection")
        self.ptz_thread.operation_completed.connect(self.on_connection_tested)
        self.ptz_thread.start()
    
    def save_configuration(self):
        """Guardar configuración"""
        try:
            config = self.get_config_from_ui()
            
            # Guardar en archivo
            config_data = {
                "ptz_cameras": [{
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
                }]
            }
            
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Configuración PTZ",
                "ptz_config.json",
                "JSON Files (*.json)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    json.dump(config_data, f, indent=4)
                
                self.log_activity(f"Configuración guardada en: {filename}")
                self.configuration_changed.emit(config_data)
                
        except Exception as e:
            self.log_activity(f"Error guardando configuración: {e}")
    
    def load_configuration(self):
        """Cargar configuración"""
        try:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Cargar Configuración PTZ",
                "",
                "JSON Files (*.json)"
            )
            
            if filename:
                with open(filename, 'r') as f:
                    config_data = json.load(f)
                
                # Cargar primera cámara PTZ
                ptz_cameras = config_data.get("ptz_cameras", [])
                if ptz_cameras:
                    cam_data = ptz_cameras[0]
                    
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
                    
                    self.set_camera_config(config)
                    self.log_activity(f"Configuración cargada desde: {filename}")
                else:
                    self.log_activity("No se encontraron cámaras PTZ en el archivo")
                    
        except Exception as e:
            self.log_activity(f"Error cargando configuración: {e}")
    
    # =============================================================================
    # MÉTODOS DE ZONA DE SEGUIMIENTO
    # =============================================================================
    
    def select_all_grid(self):
        """Seleccionar todas las celdas de la grilla"""
        self.tracking_grid.select_all_cells()
        self.log_activity("Todas las celdas seleccionadas para seguimiento")
    
    def clear_all_grid(self):
        """Limpiar selección de la grilla"""
        self.tracking_grid.clear_selection()
        self.log_activity("Selección de celdas limpiada")
    
    # =============================================================================
    # SLOTS DE SEÑALES
    # =============================================================================
    
    @pyqtSlot(bool, str)
    def on_operation_completed(self, success: bool, message: str):
        """Manejar operación completada"""
        self.log_activity(message)
        
        # Rehabilitar botones
        self.capture_btn.setEnabled(True)
        self.test_connection_btn.setEnabled(True)
    
    @pyqtSlot(bool, str)
    def on_connection_tested(self, success: bool, message: str):
        """Manejar resultado de prueba de conexión"""
        self.test_connection_btn.setEnabled(True)
        
        if success:
            self.connection_status.setText("Conectado")
            self.connection_status.setStyleSheet("color: green;")
            
            # Crear API Light para esta configuración
            self.light_api = LightAPI(
                ip=self.ip_input.text(),
                port=self.port_input.value(),
                username=self.username_input.text(),
                password=self.password_input.text()
            )
            
            # Actualizar presets automáticamente
            self.refresh_presets()
        else:
            self.connection_status.setText("Error de conexión")
            self.connection_status.setStyleSheet("color: red;")
        
        self.log_activity(message)
    
    @pyqtSlot(bool, str)
    def on_preset_created(self, success: bool, message: str):
        """Manejar creación de preset"""
        self.log_activity(message)
        
        if success:
            self.preset_name_input.clear()
            self.preset_id_spin.setValue(self.preset_id_spin.value() + 1)
            self.refresh_presets()
    
    @pyqtSlot(bool, str)
    def on_image_capture_completed(self, success: bool, message: str):
        """Manejar finalización de captura"""
        self.capture_btn.setEnabled(True)
        self.log_activity(message)
    
    @pyqtSlot(bytes)
    def on_image_captured(self, image_data: bytes):
        """Mostrar imagen capturada"""
        try:
            pixmap = QPixmap()
            pixmap.loadFromData(image_data)
            
            # Escalar imagen para preview
            scaled_pixmap = pixmap.scaled(
                self.image_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            
            self.image_preview.setPixmap(scaled_pixmap)
            
            # Opcional: Guardar imagen automáticamente
            timestamp = int(time.time())
            filename = f"ptz_capture_{timestamp}.jpg"
            pixmap.save(filename)
            self.log_activity(f"Imagen guardada como: {filename}")
            
        except Exception as e:
            self.log_activity(f"Error procesando imagen: {e}")
    
    def on_tracking_enabled_changed(self, state):
        """Manejar cambio en habilitación de seguimiento"""
        enabled = state == Qt.CheckState.Checked.value
        
        if self.camera_config:
            self.tracking_toggled.emit(self.camera_config.ip, enabled)
            
        status = "habilitado" if enabled else "deshabilitado"
        self.log_activity(f"Seguimiento {status}")
    
    # =============================================================================
    # MÉTODOS DE ACTUALIZACIÓN
    # =============================================================================
    
    def update_status(self):
        """Actualizar estado de la interfaz"""
        if not self.light_api:
            return
        
        try:
            # Actualizar posición actual
            current_pos = self.light_api.get_current_position()
            if current_pos:
                pos_text = f"Pan: {current_pos.pan:.3f}, Tilt: {current_pos.tilt:.3f}, Zoom: {current_pos.zoom:.3f}"
                self.current_position_label.setText(pos_text)
            
            # Actualizar estado de seguimiento
            if self.camera_config and self.camera_config.tracking_enabled:
                self.tracking_status_label.setText("Activo")
                self.tracking_status_label.setStyleSheet("color: green;")
            else:
                self.tracking_status_label.setText("Deshabilitado")
                self.tracking_status_label.setStyleSheet("color: red;")
            
            # Actualizar estado de API
            if self.light_api.test_connection():
                self.api_status_label.setText("Conectado")
                self.api_status_label.setStyleSheet("color: green;")
            else:
                self.api_status_label.setText("Desconectado")
                self.api_status_label.setStyleSheet("color: red;")
                
        except Exception as e:
            self.log_activity(f"Error actualizando estado: {e}")
    
    def log_activity(self, message: str):
        """Agregar mensaje al log de actividad"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        
        self.activity_log.append(formatted_message)
        
        # Mantener solo las últimas 100 líneas
        lines = self.activity_log.toPlainText().split('\n')
        if len(lines) > 100:
            self.activity_log.setPlainText('\n'.join(lines[-100:]))
    
    def clear_activity_log(self):
        """Limpiar log de actividad"""
        self.activity_log.clear()
        self.log_activity("Log limpiado")
    
    def save_activity_log(self):
        """Guardar log de actividad"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Guardar Log de Actividad",
                f"ptz_log_{int(time.time())}.txt",
                "Text Files (*.txt)"
            )
            
            if filename:
                with open(filename, 'w') as f:
                    f.write(self.activity_log.toPlainText())
                
                self.log_activity(f"Log guardado en: {filename}")
                
        except Exception as e:
            self.log_activity(f"Error guardando log: {e}")

# Función principal para testing
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Crear widget de configuración
    widget = PTZConfigWidget()
    widget.show()
    
    # Configuración de ejemplo
    example_config = PTZCameraConfig(
        ip="192.168.1.100",
        port=80,
        username="admin",
        password="admin123",
        tracking_mode=PTZTrackingMode.TRACKING,
        tracking_enabled=True
    )
    
    widget.set_camera_config(example_config)
    
    sys.exit(app.exec())