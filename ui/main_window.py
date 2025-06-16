from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QTextEdit, QMenuBar, QMenu, QGridLayout, QStackedWidget, QLabel,
    QScrollArea 
)
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
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

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.main_layout = QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

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

        self.action_ver_config = QAction("⚙️ Ver Configuración", self)
        self.action_ver_config.triggered.connect(self.abrir_configuracion_modal)
        self.menu_config.addAction(self.action_ver_config)

        # Agregar acción de FPS al menú
        self.action_fps_config = QAction("🎯 Configurar FPS", self)
        self.action_fps_config.triggered.connect(self.abrir_fps_config)
        self.menu_config.addAction(self.action_fps_config)

        self.action_edit_line = QAction("🏁 Línea de Cruce", self)
        self.action_edit_line.triggered.connect(self.toggle_line_edit)
        self.menu_config.addAction(self.action_edit_line)

        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        self.init_tab = QWidget()
        self.init_tab_layout = QVBoxLayout()
        self.init_tab.setLayout(self.init_tab_layout)
        self.setup_inicio_ui() 

        self.stacked_widget.addWidget(self.init_tab)

        cargar_camaras_guardadas(self)

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
        if any(substr in message for substr in ["hevc @", "VPS 0", "undecodable NALU", "Frame procesado"]):
            return
        self.debug_console.append(message)

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
                guardar_camaras(self)

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
        
        row = 0
        col = len(self.camera_widgets) 
        
        self.video_grid.addWidget(video_widget, row, col)
        self.camera_widgets.append(video_widget) 
        
        video_widget.mostrar_vista(camera_data) 
        video_widget.show()
        self.append_debug(f"🎥 Reproduciendo: {camera_data.get('ip', 'IP Desconocida')} con FPS optimizado")

    def show_camera_menu(self, position):
        item = self.camera_list.itemAt(position)
        if item:
            index = self.camera_list.row(item)
            menu = QMenu()
            edit_action = menu.addAction("✏️ Editar Cámara")
            delete_action = menu.addAction("🗑️ Eliminar Cámara")
            stop_action = menu.addAction("⛔ Detener Visual") 
            fps_action = menu.addAction("🎯 Configurar FPS Individual")
            action = menu.exec(self.camera_list.mapToGlobal(position))

            if action == edit_action:
                self.open_camera_dialog(index=index) 
            elif action == delete_action:
                cam_to_delete_data = self.camera_data_list.pop(index)
                self.camera_list.takeItem(index) 
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
                        widget.detener()
                        self.append_debug(f"⛔ Visual detenida para: {cam_ip_to_stop}")
                        break
            elif action == fps_action:
                self.configure_individual_fps(index)

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
