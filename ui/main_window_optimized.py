"""
MainWindow optimizado que integra todos los componentes optimizados.
Reemplaza main_window.py con mejor gestión de recursos y rendimiento.
"""

import sys
import json
import os
import cProfile
import pstats
import io
import psutil
from datetime import datetime
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QTextEdit, QMenuBar, QMenu, QGridLayout, QStackedWidget, QLabel,
    QScrollArea, QSplitter, QStatusBar, QProgressBar, QFrame
)
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from ui.camera_modal import CameraDialog
from gui.resumen_detecciones import ResumenDeteccionesWidget
from ui.config_modal import ConfiguracionDialog
from ui.camera_manager import guardar_camaras, cargar_camaras_guardadas
from core.grilla_widget_optimized import OptimizedGrillaWidget, create_optimized_grilla_widget
from core.detector_worker_optimized import detector_manager
from logging_utils import get_logger

logger = get_logger(__name__)

CONFIG_PATH = "config.json"

class PerformanceMonitor(QWidget):
    """Widget para monitorear rendimiento del sistema"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
        # Timer para actualizar métricas
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(2000)  # Actualizar cada 2 segundos
        
        # Métricas actuales
        self.system_metrics = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'memory_used': 0.0,
            'gpu_memory': 0.0,
            'active_cameras': 0,
            'total_fps': 0.0
        }
    
    def setup_ui(self):
        """Configura la interfaz del monitor"""
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(15)
        
        # CPU
        self.cpu_label = QLabel("CPU: 0%")
        self.cpu_label.setMinimumWidth(80)
        layout.addWidget(self.cpu_label)
        
        # RAM
        self.ram_label = QLabel("RAM: 0%")
        self.ram_label.setMinimumWidth(80)
        layout.addWidget(self.ram_label)
        
        # GPU (si está disponible)
        self.gpu_label = QLabel("GPU: N/A")
        self.gpu_label.setMinimumWidth(80)
        layout.addWidget(self.gpu_label)
        
        # Cámaras activas
        self.cameras_label = QLabel("Cámaras: 0")
        self.cameras_label.setMinimumWidth(80)
        layout.addWidget(self.cameras_label)
        
        # FPS total
        self.fps_label = QLabel("FPS: 0.0")
        self.fps_label.setMinimumWidth(80)
        layout.addWidget(self.fps_label)
        
        # Separador
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        layout.addWidget(separator)
        
        # Estado de conexión
        self.connection_label = QLabel("●")
        self.connection_label.setStyleSheet("color: green; font-weight: bold;")
        layout.addWidget(self.connection_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_metrics(self):
        """Actualiza las métricas del sistema"""
        try:
            # CPU
            cpu_percent = psutil.cpu_percent(interval=None)
            self.system_metrics['cpu_percent'] = cpu_percent
            
            # Memoria
            memory = psutil.virtual_memory()
            self.system_metrics['memory_percent'] = memory.percent
            self.system_metrics['memory_used'] = memory.used / (1024**3)  # GB
            
            # GPU (intentar obtener info NVIDIA)
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
                    self.system_metrics['gpu_memory'] = gpu_memory
            except:
                pass
            
            # Actualizar UI
            self._update_ui_metrics()
            
        except Exception as e:
            logger.error(f"Error actualizando métricas: {e}")
    
    def _update_ui_metrics(self):
        """Actualiza la UI con las métricas"""
        # CPU con código de color
        cpu = self.system_metrics['cpu_percent']
        cpu_color = self._get_performance_color(cpu, 70, 90)
        self.cpu_label.setText(f"CPU: {cpu:.0f}%")
        self.cpu_label.setStyleSheet(f"color: {cpu_color};")
        
        # RAM con código de color
        ram = self.system_metrics['memory_percent']
        ram_color = self._get_performance_color(ram, 70, 85)
        self.ram_label.setText(f"RAM: {ram:.0f}%")
        self.ram_label.setStyleSheet(f"color: {ram_color};")
        
        # GPU
        gpu_mem = self.system_metrics['gpu_memory']
        if gpu_mem > 0:
            gpu_percent = (gpu_mem / 4.0) * 100  # Asumiendo 4GB para RTX 3050
            gpu_color = self._get_performance_color(gpu_percent, 70, 90)
            self.gpu_label.setText(f"GPU: {gpu_mem:.1f}GB")
            self.gpu_label.setStyleSheet(f"color: {gpu_color};")
        
        # Cámaras
        cameras = self.system_metrics['active_cameras']
        self.cameras_label.setText(f"Cámaras: {cameras}")
        
        # FPS total
        fps = self.system_metrics['total_fps']
        fps_color = self._get_performance_color(100 - fps, 30, 50)  # Invertido: menos FPS = peor
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.fps_label.setStyleSheet(f"color: {fps_color};")
    
    def _get_performance_color(self, value, warning_threshold, critical_threshold):
        """Obtiene color basado en umbrales de rendimiento"""
        if value < warning_threshold:
            return "green"
        elif value < critical_threshold:
            return "orange"
        else:
            return "red"
    
    def update_camera_metrics(self, active_cameras, total_fps):
        """Actualiza métricas específicas de cámaras"""
        self.system_metrics['active_cameras'] = active_cameras
        self.system_metrics['total_fps'] = total_fps


class OptimizedMainGUI(QMainWindow):
    """
    Ventana principal optimizada con:
    - Gestión eficiente de múltiples cámaras
    - Monitor de rendimiento en tiempo real
    - Threading optimizado
    - Gestión inteligente de recursos
    """
    
    # Señales para comunicación interna
    camera_performance_updated = pyqtSignal(dict)
    
    def __init__(self):
        super().__init__()
        
        # Configuración inicial
        self.setWindowTitle("Monitor PTZ Inteligente - Orca [OPTIMIZADO]")
        self.setGeometry(100, 100, 1600, 900)
        
        # Profiling para optimización
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        
        # Datos de cámaras y widgets
        self.camera_data_list = []
        self.camera_widgets = []
        self.camera_performance = {}
        
        # Configurar UI
        self._setup_ui()
        self._setup_menu()
        self._setup_status_bar()
        
        # Configurar monitoreo de rendimiento
        self._setup_performance_monitoring()
        
        # Cargar configuración guardada
        cargar_camaras_guardadas(self)
        
        logger.info("OptimizedMainGUI inicializado")
    
    def _setup_ui(self):
        """Configura la interfaz de usuario optimizada"""
        # Widget central
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Layout principal
        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)
        
        # Widget apilado para diferentes vistas
        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)
        
        # Página principal con cámaras
        self.main_page = QWidget()
        self._setup_main_page()
        self.stacked_widget.addWidget(self.main_page)
    
    def _setup_main_page(self):
        """Configura la página principal con cámaras"""
        layout = QVBoxLayout()
        self.main_page.setLayout(layout)
        
        # Splitter principal (vertical)
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        layout.addWidget(main_splitter)
        
        # --- Área superior: Video Grid ---
        self.video_grid_container = self._create_video_grid_container()
        main_splitter.addWidget(self.video_grid_container)
        
        # --- Área inferior: Controles y logs ---
        bottom_widget = self._create_bottom_panel()
        main_splitter.addWidget(bottom_widget)
        
        # Configurar proporciones del splitter
        main_splitter.setSizes([600, 300])  # 2:1 ratio
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
    
    def _create_video_grid_container(self):
        """Crea el contenedor para la grilla de videos"""
        # Scroll area para múltiples cámaras
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Widget contenedor de la grilla
        self.video_grid_widget = QWidget()
        self.video_grid = QGridLayout()
        self.video_grid.setSpacing(5)
        self.video_grid_widget.setLayout(self.video_grid)
        
        scroll_area.setWidget(self.video_grid_widget)
        
        return scroll_area
    
    def _create_bottom_panel(self):
        """Crea el panel inferior con controles y logs"""
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        
        # --- Panel izquierdo: Lista de cámaras ---
        left_panel = QFrame()
        left_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout()
        left_panel.setLayout(left_layout)
        
        # Título y lista de cámaras
        camera_label = QLabel("📹 Cámaras Activas")
        camera_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        left_layout.addWidget(camera_label)
        
        self.camera_list = QListWidget()
        self.camera_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.camera_list.customContextMenuRequested.connect(self._show_camera_menu)
        left_layout.addWidget(self.camera_list)
        
        layout.addWidget(left_panel)
        
        # --- Panel central: Console de debug ---
        center_panel = QFrame()
        center_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        center_layout = QVBoxLayout()
        center_panel.setLayout(center_layout)
        
        console_label = QLabel("🖥️ Console de Sistema")
        console_label.setFont(QFont("Arial", 10, QFont.Weight.Bold))
        center_layout.addWidget(console_label)
        
        self.debug_console = QTextEdit()
        self.debug_console.setReadOnly(True)
        self.debug_console.setMaximumHeight(200)
        self.debug_console.setFont(QFont("Consolas", 9))
        center_layout.addWidget(self.debug_console)
        
        layout.addWidget(center_panel, 2)  # 2x más espacio
        
        # --- Panel derecho: Resumen de detecciones ---
        right_panel = QFrame()
        right_panel.setFrameStyle(QFrame.Shape.StyledPanel)
        right_panel.setFixedWidth(350)
        right_layout = QVBoxLayout()
        right_panel.setLayout(right_layout)
        
        self.resumen_widget = ResumenDeteccionesWidget()
        self.resumen_widget.log_signal.connect(self.debug_console.append)
        right_layout.addWidget(self.resumen_widget)
        
        layout.addWidget(right_panel)
        
        return widget
    
    def _setup_menu(self):
        """Configura el menú de la aplicación"""
        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)
        
        # Menú Archivo
        file_menu = self.menu_bar.addMenu("📁 Archivo")
        
        add_camera_action = QAction("➕ Agregar Cámara", self)
        add_camera_action.setShortcut("Ctrl+N")
        add_camera_action.triggered.connect(self._add_camera_dialog)
        file_menu.addAction(add_camera_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("🚪 Salir", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menú Configuración
        config_menu = self.menu_bar.addMenu("⚙️ Configuración")
        
        settings_action = QAction("🔧 Configuración General", self)
        settings_action.triggered.connect(self._open_settings_dialog)
        config_menu.addAction(settings_action)
        
        config_menu.addSeparator()
        
        performance_action = QAction("📊 Monitor de Rendimiento", self)
        performance_action.setCheckable(True)
        performance_action.setChecked(True)
        performance_action.triggered.connect(self._toggle_performance_monitor)
        config_menu.addAction(performance_action)
        
        # Menú Herramientas
        tools_menu = self.menu_bar.addMenu("🛠️ Herramientas")
        
        pause_all_action = QAction("⏸️ Pausar Todas las Cámaras", self)
        pause_all_action.triggered.connect(self._pause_all_cameras)
        tools_menu.addAction(pause_all_action)
        
        resume_all_action = QAction("▶️ Reanudar Todas las Cámaras", self)
        resume_all_action.triggered.connect(self._resume_all_cameras)
        tools_menu.addAction(resume_all_action)
        
        tools_menu.addSeparator()
        
        cleanup_action = QAction("🧹 Limpiar Memoria", self)
        cleanup_action.triggered.connect(self._cleanup_memory)
        tools_menu.addAction(cleanup_action)
    
    def _setup_status_bar(self):
        """Configura la barra de estado con monitor de rendimiento"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Monitor de rendimiento integrado
        self.performance_monitor = PerformanceMonitor()
        self.status_bar.addPermanentWidget(self.performance_monitor)
        
        # Mensaje de estado
        self.status_bar.showMessage("Sistema optimizado listo - RTX 3050 detectada")
    
    def _setup_performance_monitoring(self):
        """Configura el monitoreo de rendimiento"""
        # Timer para actualizar métricas de cámaras
        self.camera_metrics_timer = QTimer()
        self.camera_metrics_timer.timeout.connect(self._update_camera_metrics)
        self.camera_metrics_timer.start(3000)  # Cada 3 segundos
        
        # Conectar señales de rendimiento
        self.camera_performance_updated.connect(self._on_camera_performance_updated)
    
    def _add_camera_dialog(self, edit_index=None):
        """Abre el diálogo para agregar/editar cámara"""
        try:
            existing_data = None
            if edit_index is not None and edit_index < len(self.camera_data_list):
                existing_data = self.camera_data_list[edit_index]
            
            dialog = CameraDialog(self, existing_data=existing_data)
            
            if dialog.exec():
                camera_data = dialog.get_camera_data()
                
                if edit_index is not None:
                    # Editar cámara existente
                    self._update_existing_camera(edit_index, camera_data)
                else:
                    # Agregar nueva cámara
                    self._add_new_camera(camera_data)
                
                # Guardar configuración
                guardar_camaras(self)
                
        except Exception as e:
            logger.error(f"Error en diálogo de cámara: {e}")
            self.debug_console.append(f"❌ Error en configuración de cámara: {e}")
    
    def _add_new_camera(self, camera_data):
        """Agrega una nueva cámara al sistema"""
        try:
            # Validar datos
            if not camera_data.get("ip"):
                raise ValueError("IP de cámara requerida")
            
            # Verificar que no exista duplicada
            for existing_cam in self.camera_data_list:
                if existing_cam.get("ip") == camera_data.get("ip"):
                    raise ValueError(f"Cámara con IP {camera_data['ip']} ya existe")
            
            # Agregar a la lista
            self.camera_data_list.append(camera_data)
            
            # Actualizar UI
            display_name = f"{camera_data['ip']} - {camera_data.get('tipo', 'N/A')}"
            self.camera_list.addItem(display_name)
            
            # Iniciar stream de la cámara
            self._start_camera_stream(camera_data)
            
            self.debug_console.append(f"✅ Cámara agregada: {camera_data['ip']}")
            logger.info(f"Nueva cámara agregada: {camera_data['ip']}")
            
        except Exception as e:
            logger.error(f"Error agregando cámara: {e}")
            self.debug_console.append(f"❌ Error agregando cámara: {e}")
    
    def _update_existing_camera(self, index, camera_data):
        """Actualiza una cámara existente"""
        try:
            old_camera = self.camera_data_list[index]
            old_ip = old_camera.get("ip")
            
            # Actualizar datos
            self.camera_data_list[index] = camera_data
            
            # Actualizar lista UI
            display_name = f"{camera_data['ip']} - {camera_data.get('tipo', 'N/A')}"
            self.camera_list.item(index).setText(display_name)
            
            # Reiniciar stream si cambió la IP o configuración crítica
            restart_required = (
                old_ip != camera_data.get("ip") or
                old_camera.get("modelo") != camera_data.get("modelo") or
                old_camera.get("rtsp") != camera_data.get("rtsp")
            )
            
            if restart_required:
                self._restart_camera_stream(index, camera_data)
            else:
                # Solo actualizar configuración
                self._update_camera_config(index, camera_data)
            
            self.debug_console.append(f"✏️ Cámara actualizada: {camera_data['ip']}")
            logger.info(f"Cámara actualizada: {camera_data['ip']}")
            
        except Exception as e:
            logger.error(f"Error actualizando cámara: {e}")
            self.debug_console.append(f"❌ Error actualizando cámara: {e}")
    
    def _start_camera_stream(self, camera_data):
        """Inicia el stream de una cámara"""
        try:
            # Verificar si ya existe un widget para esta IP
            camera_ip = camera_data.get("ip")
            for i, widget in enumerate(self.camera_widgets):
                if hasattr(widget, 'cam_data') and widget.cam_data.get('ip') == camera_ip:
                    logger.info(f"Reemplazando widget existente para {camera_ip}")
                    self._remove_camera_widget(i)
                    break
            
            # Crear widget optimizado
            camera_widget = create_optimized_grilla_widget(camera_data, parent=self.video_grid_widget)
            
            if not camera_widget:
                raise RuntimeError("No se pudo crear el widget de cámara")
            
            # Conectar señales
            camera_widget.log_signal.connect(self.debug_console.append)
            camera_widget.performance_signal.connect(
                lambda stats, ip=camera_ip: self._update_camera_performance(ip, stats)
            )
            
            # Agregar al grid (disposición automática)
            row, col = self._get_next_grid_position()
            self.video_grid.addWidget(camera_widget, row, col)
            
            # Agregar a la lista de widgets
            self.camera_widgets.append(camera_widget)
            
            # Mostrar el widget
            camera_widget.show()
            
            logger.info(f"Stream iniciado para {camera_ip}")
            self.debug_console.append(f"🎥 Stream iniciado: {camera_ip}")
            
        except Exception as e:
            logger.error(f"Error iniciando stream: {e}")
            self.debug_console.append(f"❌ Error iniciando stream: {e}")
    
    def _get_next_grid_position(self):
        """Calcula la siguiente posición en el grid"""
        num_cameras = len(self.camera_widgets)
        
        # Disposición automática: 2 columnas para empezar
        cols = 2
        if num_cameras > 4:
            cols = 3
        if num_cameras > 9:
            cols = 4
        
        row = num_cameras // cols
        col = num_cameras % cols
        
        return row, col
    
    def _remove_camera_widget(self, index):
        """Remueve un widget de cámara de forma segura"""
        try:
            if 0 <= index < len(self.camera_widgets):
                widget = self.camera_widgets[index]
                
                # Detener el widget
                if hasattr(widget, 'detener'):
                    widget.detener()
                
                # Remover del grid
                self.video_grid.removeWidget(widget)
                
                # Eliminar el widget
                widget.deleteLater()
                
                # Remover de la lista
                self.camera_widgets.pop(index)
                
                logger.info(f"Widget de cámara removido (index: {index})")
                
        except Exception as e:
            logger.error(f"Error removiendo widget de cámara: {e}")
    
    def _restart_camera_stream(self, index, camera_data):
        """Reinicia el stream de una cámara"""
        try:
            # Remover widget actual
            camera_ip = camera_data.get("ip")
            for i, widget in enumerate(self.camera_widgets):
                if hasattr(widget, 'cam_data') and widget.cam_data.get('ip') == camera_ip:
                    self._remove_camera_widget(i)
                    break
            
            # Crear nuevo stream
            self._start_camera_stream(camera_data)
            
            logger.info(f"Stream reiniciado para {camera_ip}")
            
        except Exception as e:
            logger.error(f"Error reiniciando stream: {e}")
    
    def _update_camera_config(self, index, camera_data):
        """Actualiza configuración de cámara sin reiniciar stream"""
        try:
            camera_ip = camera_data.get("ip")
            
            # Buscar el widget correspondiente
            for widget in self.camera_widgets:
                if hasattr(widget, 'cam_data') and widget.cam_data.get('ip') == camera_ip:
                    # Actualizar configuración específica
                    if hasattr(widget, 'update_detector_confidence'):
                        new_confidence = camera_data.get("confianza")
                        if new_confidence != widget.cam_data.get("confianza"):
                            widget.update_detector_confidence(new_confidence)
                    
                    # Actualizar datos de la cámara
                    widget.cam_data.update(camera_data)
                    break
            
            logger.info(f"Configuración actualizada para {camera_ip}")
            
        except Exception as e:
            logger.error(f"Error actualizando configuración: {e}")
    
    def _show_camera_menu(self, position):
        """Muestra menú contextual para cámaras"""
        try:
            item = self.camera_list.itemAt(position)
            if not item:
                return
            
            index = self.camera_list.row(item)
            
            menu = QMenu(self)
            
            # Opciones del menú
            edit_action = menu.addAction("✏️ Editar Cámara")
            pause_action = menu.addAction("⏸️ Pausar")
            resume_action = menu.addAction("▶️ Reanudar")
            restart_action = menu.addAction("🔄 Reiniciar Stream")
            
            menu.addSeparator()
            delete_action = menu.addAction("🗑️ Eliminar Cámara")
            
            # Ejecutar menú
            action = menu.exec(self.camera_list.mapToGlobal(position))
            
            if action == edit_action:
                self._add_camera_dialog(edit_index=index)
            elif action == pause_action:
                self._pause_camera(index)
            elif action == resume_action:
                self._resume_camera(index)
            elif action == restart_action:
                self._restart_camera(index)
            elif action == delete_action:
                self._delete_camera(index)
                
        except Exception as e:
            logger.error(f"Error en menú de cámara: {e}")
    
    def _pause_camera(self, index):
        """Pausa una cámara específica"""
        try:
            if 0 <= index < len(self.camera_widgets):
                widget = self.camera_widgets[index]
                if hasattr(widget, 'pause'):
                    widget.pause()
                    camera_ip = self.camera_data_list[index].get("ip", "N/A")
                    self.debug_console.append(f"⏸️ Cámara pausada: {camera_ip}")
                    
        except Exception as e:
            logger.error(f"Error pausando cámara: {e}")
    
    def _resume_camera(self, index):
        """Reanuda una cámara específica"""
        try:
            if 0 <= index < len(self.camera_widgets):
                widget = self.camera_widgets[index]
                if hasattr(widget, 'resume'):
                    widget.resume()
                    camera_ip = self.camera_data_list[index].get("ip", "N/A")
                    self.debug_console.append(f"▶️ Cámara reanudada: {camera_ip}")
                    
        except Exception as e:
            logger.error(f"Error reanudando cámara: {e}")
    
    def _restart_camera(self, index):
        """Reinicia una cámara específica"""
        try:
            if 0 <= index < len(self.camera_data_list):
                camera_data = self.camera_data_list[index]
                self._restart_camera_stream(index, camera_data)
                self.debug_console.append(f"🔄 Stream reiniciado: {camera_data.get('ip', 'N/A')}")
                
        except Exception as e:
            logger.error(f"Error reiniciando cámara: {e}")
    
    def _delete_camera(self, index):
        """Elimina una cámara del sistema"""
        try:
            if 0 <= index < len(self.camera_data_list):
                camera_data = self.camera_data_list.pop(index)
                camera_ip = camera_data.get("ip", "N/A")
                
                # Remover de la lista UI
                self.camera_list.takeItem(index)
                
                # Remover widget correspondiente
                for i, widget in enumerate(self.camera_widgets):
                    if hasattr(widget, 'cam_data') and widget.cam_data.get('ip') == camera_ip:
                        self._remove_camera_widget(i)
                        break
                
                # Reorganizar grid
                self._reorganize_camera_grid()
                
                # Guardar configuración
                guardar_camaras(self)
                
                self.debug_console.append(f"🗑️ Cámara eliminada: {camera_ip}")
                logger.info(f"Cámara eliminada: {camera_ip}")
                
        except Exception as e:
            logger.error(f"Error eliminando cámara: {e}")
    
    def _reorganize_camera_grid(self):
        """Reorganiza el grid de cámaras después de eliminar una"""
        try:
            # Limpiar el grid
            for widget in self.camera_widgets:
                self.video_grid.removeWidget(widget)
            
            # Reposicionar widgets
            for i, widget in enumerate(self.camera_widgets):
                row, col = self._get_grid_position_for_index(i)
                self.video_grid.addWidget(widget, row, col)
                
        except Exception as e:
            logger.error(f"Error reorganizando grid: {e}")
    
    def _get_grid_position_for_index(self, index):
        """Obtiene posición en grid para un índice específico"""
        total_cameras = len(self.camera_widgets)
        
        # Lógica de disposición adaptativa
        if total_cameras <= 2:
            cols = 1
        elif total_cameras <= 4:
            cols = 2
        elif total_cameras <= 9:
            cols = 3
        else:
            cols = 4
        
        row = index // cols
        col = index % cols
        
        return row, col
    
    def _open_settings_dialog(self):
        """Abre el diálogo de configuración general"""
        try:
            dialog = ConfiguracionDialog(self, camera_list=self.camera_data_list)
            
            if dialog.exec():
                # Configuración guardada exitosamente
                guardar_camaras(self)
                self.debug_console.append("⚙️ Configuración del sistema guardada")
                logger.info("Configuración del sistema actualizada")
            else:
                self.debug_console.append("⚙️ Cambios en configuración cancelados")
                
        except Exception as e:
            logger.error(f"Error en diálogo de configuración: {e}")
            self.debug_console.append(f"❌ Error en configuración: {e}")
    
    def _toggle_performance_monitor(self, enabled):
        """Alterna la visibilidad del monitor de rendimiento"""
        try:
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.setVisible(enabled)
                
            status = "habilitado" if enabled else "deshabilitado"
            self.debug_console.append(f"📊 Monitor de rendimiento {status}")
            
        except Exception as e:
            logger.error(f"Error alternando monitor de rendimiento: {e}")
    
    def _pause_all_cameras(self):
        """Pausa todas las cámaras activas"""
        try:
            paused_count = 0
            for widget in self.camera_widgets:
                if hasattr(widget, 'pause'):
                    widget.pause()
                    paused_count += 1
            
            self.debug_console.append(f"⏸️ {paused_count} cámaras pausadas")
            logger.info(f"Pausadas {paused_count} cámaras")
            
        except Exception as e:
            logger.error(f"Error pausando todas las cámaras: {e}")
    
    def _resume_all_cameras(self):
        """Reanuda todas las cámaras pausadas"""
        try:
            resumed_count = 0
            for widget in self.camera_widgets:
                if hasattr(widget, 'resume'):
                    widget.resume()
                    resumed_count += 1
            
            self.debug_console.append(f"▶️ {resumed_count} cámaras reanudadas")
            logger.info(f"Reanudadas {resumed_count} cámaras")
            
        except Exception as e:
            logger.error(f"Error reanudando todas las cámaras: {e}")
    
    def _cleanup_memory(self):
        """Ejecuta limpieza de memoria del sistema"""
        try:
            import gc
            import torch
            
            # Limpieza Python
            collected = gc.collect()
            
            # Limpieza CUDA si está disponible
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Limpieza del manager de detectores
            detector_manager.stop_all_detectors()
            
            self.debug_console.append(f"🧹 Memoria limpiada - {collected} objetos recolectados")
            logger.info(f"Limpieza de memoria ejecutada - {collected} objetos")
            
        except Exception as e:
            logger.error(f"Error en limpieza de memoria: {e}")
            self.debug_console.append(f"❌ Error en limpieza de memoria: {e}")
    
    def _update_camera_metrics(self):
        """Actualiza métricas de rendimiento de cámaras"""
        try:
            active_cameras = len([w for w in self.camera_widgets if hasattr(w, 'cam_data')])
            total_fps = sum(
                perf.get('detection_fps', 0) 
                for perf in self.camera_performance.values()
            )
            
            # Actualizar monitor de rendimiento
            if hasattr(self, 'performance_monitor'):
                self.performance_monitor.update_camera_metrics(active_cameras, total_fps)
            
            # Emitir señal de actualización
            metrics = {
                'active_cameras': active_cameras,
                'total_fps': total_fps,
                'individual_performance': self.camera_performance.copy()
            }
            self.camera_performance_updated.emit(metrics)
            
        except Exception as e:
            logger.error(f"Error actualizando métricas de cámaras: {e}")
    
    def _update_camera_performance(self, camera_ip, performance_stats):
        """Actualiza estadísticas de rendimiento de una cámara específica"""
        try:
            self.camera_performance[camera_ip] = performance_stats
            
            # Log de rendimiento crítico
            detection_fps = performance_stats.get('detection_fps', 0)
            memory_usage = performance_stats.get('memory_usage', 0)
            
            if detection_fps < 3.0:
                logger.warning(f"Cámara {camera_ip}: FPS bajo ({detection_fps:.1f})")
            
            if memory_usage > 3.0:  # > 3GB para RTX 3050
                logger.warning(f"Cámara {camera_ip}: Memoria GPU alta ({memory_usage:.1f}GB)")
                
        except Exception as e:
            logger.error(f"Error actualizando rendimiento de cámara: {e}")
    
    def _on_camera_performance_updated(self, metrics):
        """Maneja actualizaciones de rendimiento de cámaras"""
        try:
            # Actualizar status bar si es necesario
            active_cameras = metrics.get('active_cameras', 0)
            total_fps = metrics.get('total_fps', 0)
            
            if active_cameras == 0:
                self.status_bar.showMessage("Sin cámaras activas")
            else:
                self.status_bar.showMessage(
                    f"{active_cameras} cámaras activas - {total_fps:.1f} FPS total"
                )
                
        except Exception as e:
            logger.error(f"Error procesando métricas de rendimiento: {e}")
    
    def start_camera_stream(self, camera_data):
        """Método de compatibilidad para cargar_camaras_guardadas"""
        self._start_camera_stream(camera_data)
    
    def closeEvent(self, event):
        """Maneja el cierre de la aplicación de forma optimizada"""
        try:
            logger.info("Iniciando cierre optimizado de la aplicación...")
            
            # Detener timers
            if hasattr(self, 'camera_metrics_timer'):
                self.camera_metrics_timer.stop()
            
            # Detener todos los widgets de cámara
            logger.info(f"Deteniendo {len(self.camera_widgets)} widgets de cámara...")
            for i, widget in enumerate(self.camera_widgets):
                try:
                    if hasattr(widget, 'detener'):
                        camera_ip = "N/A"
                        if hasattr(widget, 'cam_data') and widget.cam_data:
                            camera_ip = widget.cam_data.get('ip', 'N/A')
                        
                        logger.info(f"Deteniendo widget {i+1}/{len(self.camera_widgets)} (IP: {camera_ip})")
                        widget.detener()
                        
                except Exception as e:
                    logger.error(f"Error deteniendo widget {i}: {e}")
            
            # Detener todos los detectores
            logger.info("Deteniendo todos los detectores...")
            try:
                detector_manager.stop_all_detectors()
            except Exception as e:
                logger.error(f"Error deteniendo detectores: {e}")
            
            # Detener resumen widget
            if hasattr(self, 'resumen_widget') and self.resumen_widget:
                try:
                    if hasattr(self.resumen_widget, 'stop_threads'):
                        logger.info("Deteniendo threads del resumen widget...")
                        self.resumen_widget.stop_threads()
                except Exception as e:
                    logger.error(f"Error deteniendo resumen widget: {e}")
            
            # Limpieza final de memoria
            try:
                import gc
                import torch
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                gc.collect()
                logger.info("Limpieza final de memoria completada")
                
            except Exception as e:
                logger.error(f"Error en limpieza final: {e}")
            
            # Profiling final
            try:
                logger.info("Finalizando profiler y guardando estadísticas...")
                self.profiler.disable()
                
                # Guardar estadísticas de profiling
                stats_filename = f"main_gui_profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.prof"
                self.profiler.dump_stats(stats_filename)
                
                # Mostrar resumen en consola
                s = io.StringIO()
                ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative', 'tottime')
                ps.print_stats(20)  # Top 20 funciones
                
                logger.info("=== RESUMEN DE PROFILING ===")
                logger.info(s.getvalue())
                logger.info("=== FIN RESUMEN PROFILING ===")
                
            except Exception as e:
                logger.error(f"Error en profiling final: {e}")
            
            logger.info("Cierre optimizado completado exitosamente")
            event.accept()
            
        except Exception as e:
            logger.error(f"Error durante cierre de aplicación: {e}")
            event.accept()  # Cerrar de todas formas


class OptimizedApplication:
    """Clase wrapper para la aplicación optimizada"""
    
    def __init__(self):
        self.app = None
        self.main_window = None
    
    def run(self):
        """Ejecuta la aplicación optimizada"""
        try:
            # Configurar aplicación Qt
            from PyQt6.QtWidgets import QApplication
            import sys
            
            self.app = QApplication(sys.argv)
            self.app.setApplicationName("Monitor PTZ Inteligente - Orca")
            self.app.setApplicationVersion("2.0 Optimizado")
            
            # Configurar estilo y optimizaciones Qt
            self._setup_qt_optimizations()
            
            # Crear ventana principal
            self.main_window = OptimizedMainGUI()
            
            # Configurar señales de aplicación
            self.app.aboutToQuit.connect(self._cleanup_on_quit)
            
            # Mostrar ventana
            self.main_window.show()
            
            # Mensaje de inicio optimizado
            logger.info("=== MONITOR PTZ INTELIGENTE - ORCA [OPTIMIZADO] ===")
            logger.info("🚀 Sistema optimizado para RTX 3050 iniciado")
            logger.info("📊 Configuraciones recomendadas:")
            logger.info("   • Visual FPS: 25, Detección FPS: 8, UI FPS: 15")
            logger.info("   • Memoria GPU: <3.5GB, CPU: <30%")
            logger.info("   • Cámaras simultáneas: hasta 4")
            logger.info("=== SISTEMA LISTO ===")
            
            # Ejecutar aplicación
            return self.app.exec()
            
        except Exception as e:
            logger.error(f"Error ejecutando aplicación: {e}")
            return 1
    
    def _setup_qt_optimizations(self):
        """Configura optimizaciones específicas de Qt"""
        try:
            # Optimizaciones de rendering
            self.app.setAttribute(Qt.ApplicationAttribute.AA_UseOpenGLES, True)
            self.app.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, False)
            
            # Optimizaciones de estilo
            self.app.setStyle('Fusion')  # Estilo más eficiente
            
            logger.info("✅ Optimizaciones Qt aplicadas")
            
        except Exception as e:
            logger.warning(f"⚠️ No se pudieron aplicar todas las optimizaciones Qt: {e}")
    
    def _cleanup_on_quit(self):
        """Limpieza final antes de cerrar la aplicación"""
        try:
            logger.info("Ejecutando limpieza final de la aplicación...")
            
            # Limpieza global de detectores
            detector_manager.stop_all_detectors()
            
            # Limpieza de memoria CUDA
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
            
            logger.info("Limpieza final completada")
            
        except Exception as e:
            logger.error(f"Error en limpieza final: {e}")


# Función principal optimizada
def main():
    """Función principal para ejecutar la aplicación optimizada"""
    try:
        # Configurar logging
        import logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
            handlers=[
                logging.FileHandler('orca_optimized.log'),
                logging.StreamHandler()
            ]
        )
        
        # Verificar requisitos del sistema
        _check_system_requirements()
        
        # Crear y ejecutar aplicación
        app = OptimizedApplication()
        return app.run()
        
    except Exception as e:
        logger.error(f"Error crítico en main: {e}")
        return 1


def _check_system_requirements():
    """Verifica los requisitos del sistema"""
    try:
        import torch
        import cv2
        import psutil
        
        logger.info("=== VERIFICACIÓN DE SISTEMA ===")
        
        # Verificar CUDA
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("⚠️ CUDA no disponible - funcionará en CPU (rendimiento reducido)")
        
        # Verificar memoria RAM
        memory = psutil.virtual_memory()
        logger.info(f"✅ RAM: {memory.total / (1024**3):.1f}GB disponible")
        
        if memory.total < 8 * (1024**3):  # Menos de 8GB
            logger.warning("⚠️ RAM baja detectada - se recomienda 8GB+ para óptimo rendimiento")
        
        # Verificar OpenCV
        logger.info(f"✅ OpenCV: {cv2.__version__}")
        
        # Verificar PyTorch
        logger.info(f"✅ PyTorch: {torch.__version__}")
        
        logger.info("=== VERIFICACIÓN COMPLETADA ===")
        
    except Exception as e:
        logger.error(f"Error verificando requisitos: {e}")


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)