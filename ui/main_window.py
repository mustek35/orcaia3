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
from ui.camera_manager import guardar_camaras, cargar_camaras_guardadas
from core.rtsp_builder import generar_rtsp
import os
import cProfile # Añadido para profiling
import pstats   # Añadido para profiling
import io       # Añadido para profiling

CONFIG_PATH = "config.json"

class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        
        print("INFO: Iniciando profiler para MainGUI...")
        self.profiler = cProfile.Profile()
        self.profiler.enable()

        self.setWindowTitle("Monitor PTZ Inteligente - Orca")
        self.setGeometry(100, 100, 1600, 900)

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

        self.stacked_widget = QStackedWidget()
        self.main_layout.addWidget(self.stacked_widget)

        self.init_tab = QWidget()
        self.init_tab_layout = QVBoxLayout()
        self.init_tab.setLayout(self.init_tab_layout)
        self.setup_inicio_ui() 

        self.stacked_widget.addWidget(self.init_tab)

        cargar_camaras_guardadas(self)

    def setup_inicio_ui(self):
        self.video_grid = QGridLayout()
        video_grid_container_widget = QWidget() 
        video_grid_container_widget.setLayout(self.video_grid)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True) 
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff) 
        scroll_area.setWidget(video_grid_container_widget)
        self.init_tab_layout.addWidget(scroll_area, 3) 

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
        self.resumen_widget.log_signal.connect(self.debug_console.append)
        bottom_layout.addWidget(self.resumen_widget, 1)

        self.init_tab_layout.addLayout(bottom_layout, 1) 

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
                    self.debug_console.append(f"✏️ Cámara editada: {new_data}")
                    self.start_camera_stream(new_data) 
                else:
                    self.camera_data_list.append(new_data)
                    self.camera_list.addItem(f"{new_data['ip']} - {new_data['tipo']}")
                    self.debug_console.append(f"✅ Cámara agregada: {new_data}")
                    self.start_camera_stream(new_data)
                guardar_camaras(self)

    def abrir_configuracion_modal(self):
        dialog = ConfiguracionDialog(self, camera_list=self.camera_data_list)
        if dialog.exec():
            self.debug_console.append(f"⚙️ Configuración general guardada (si aplica).")

    def start_camera_stream(self, camera_data):
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
        # Esto es un poco indirecto, sería mejor tener una referencia directa si es posible
        for i in range(self.init_tab_layout.count()):
            item = self.init_tab_layout.itemAt(i)
            if isinstance(item.widget(), QScrollArea):
                video_grid_container_widget = item.widget().widget() # El widget dentro del QScrollArea
                break

        try:
            grilla_widget_module = importlib.import_module("gui.grilla_widget")
            GrillaWidget_class = grilla_widget_module.GrillaWidget
        except ImportError as e:
            print(f"ERROR: No se pudo importar GrillaWidget: {e}")
            self.debug_console.append(f"ERROR: No se pudo importar GrillaWidget: {e}")
            return

        # Usar video_grid_container_widget como parent si se encontró, sino self.
        # El parent debe ser un QWidget. self.video_grid es un QLayout.
        parent_widget = video_grid_container_widget if video_grid_container_widget else self
        video_widget = GrillaWidget_class(parent=parent_widget) 
        
        video_widget.cam_data = camera_data 
        video_widget.log_signal.connect(self.debug_console.append) 
        
        row = 0
        col = len(self.camera_widgets) 
        
        self.video_grid.addWidget(video_widget, row, col)
        self.camera_widgets.append(video_widget) 
        
        video_widget.mostrar_vista(camera_data) 
        video_widget.show()
        self.debug_console.append(f"🎥 Reproduciendo: {camera_data.get('ip', 'IP Desconocida')}")

    def show_camera_menu(self, position):
        item = self.camera_list.itemAt(position)
        if item:
            index = self.camera_list.row(item)
            menu = QMenu()
            edit_action = menu.addAction("✏️ Editar Cámara")
            delete_action = menu.addAction("🗑️ Eliminar Cámara")
            stop_action = menu.addAction("⛔ Detener Visual") 
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
                        self.debug_console.append(f"🗑️ Cámara {cam_to_delete_data.get('ip')} y su widget eliminados.")
                        break
                guardar_camaras(self) 
            elif action == stop_action:
                cam_ip_to_stop = self.camera_data_list[index].get('ip')
                for i, widget in enumerate(self.camera_widgets):
                     if hasattr(widget, 'cam_data') and widget.cam_data.get('ip') == cam_ip_to_stop:
                        widget.detener()
                        self.debug_console.append(f"⛔ Visual detenida para: {cam_ip_to_stop}")
                        break

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
        # Ordenar por 'cumulative' (tiempo acumulado) y luego por 'tottime' (tiempo total interno)
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative', 'tottime')
        ps.print_stats(30) # Imprime las 30 funciones más costosas
        print("\n--- Resumen del Profiler (Top 30 por tiempo acumulado) ---")
        print(s.getvalue())
        print("--- Fin del Resumen del Profiler ---\n")

        print("INFO: Proceso de cierre de MainGUI completado. Aceptando evento.")
        event.accept()
