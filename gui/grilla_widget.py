from PyQt6.QtWidgets import QWidget, QSizePolicy, QMenu
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QFont, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QSizeF, QSize, QPointF, QTimer
from PyQt6.QtMultimedia import QVideoFrame, QVideoFrameFormat
from gui.visualizador_detector import VisualizadorDetector
from core.gestor_alertas import GestorAlertas
from core.rtsp_builder import generar_rtsp
from core.analytics_processor import AnalyticsProcessor
from gui.image_saver import ImageSaverThread
from gui.video_saver import VideoSaverThread
from core.cross_line_counter import CrossLineCounter
from collections import defaultdict, deque
import numpy as np
from datetime import datetime
import uuid
import json
import os

DEBUG_LOGS = False

CONFIG_FILE_PATH = "config.json"

class GrillaWidget(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self, filas=18, columnas=22, area=None, parent=None):
        super().__init__(parent)
        self.filas = filas
        self.columnas = columnas
        self.area = area if area else [0] * (filas * columnas)
        self.temporal = set()
        self.pixmap = None
        self.last_frame = None 
        self.original_frame_size = None 
        self.latest_tracked_boxes = [] 
        self.selected_cells = set()
        self.discarded_cells = set()

        self.cam_data = None
        self.alertas = None
        self.objetos_previos = {}
        self.umbral_movimiento = 20
        self.detectors = None 
        self.analytics_processor = AnalyticsProcessor(self) # Pass self as parent for Qt object management
        # You might want to connect its signals if you were using it actively
        # self.analytics_processor.log_signal.connect(self.registrar_log)
        # self.analytics_processor.processing_finished.connect(self.handle_analytics_results)

        self.cross_counter = CrossLineCounter()
        self.cross_counter.counts_updated.connect(self._update_cross_counts)
        self.cross_counter.log_signal.connect(self.registrar_log)
        self.cross_counter.cross_event.connect(self._handle_cross_event)
        self.cross_counter.start()
        self.cross_counter.active = False
        self.cross_counts = {}
        self.cross_line_enabled = False
        self.cross_line_edit_mode = False
        self._temp_line_start = None
        self._dragging_line = None
        self._last_mouse_pos = None

        # Mantiene aprox. 5 segundos previos (10 FPS) para clips de cruce
        self.frame_buffer = deque(maxlen=50)
        # Pendientes a completar con 5 segundos posteriores al evento
        self.pending_videos = []  # list of dicts {frames, frames_left, path}
        self.active_video_threads = []

        self.setFixedSize(640, 480)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Pixmap con las líneas de la grilla pre-renderizadas
        self._grid_lines_pixmap = None
        self._generate_grid_lines_pixmap()

        # Atributos para la optimización de paintEvent
        self.PAINT_UPDATE_INTERVAL = 80  # ms, para aprox. 12 FPS de UI
        self.paint_update_timer = QTimer(self)
        self.paint_update_timer.setSingleShot(True)
        self.paint_update_timer.timeout.connect(self.perform_paint_update)
        self.paint_scheduled = False

        # Atributos para limitar la frecuencia de actualización de la UI por frames de video
        self.ui_frame_counter = 1
        self.UI_UPDATE_INTERVAL = 5  # Procesar 1 de cada N frames para la UI

    def enable_cross_line(self):
        self.cross_line_enabled = True
        self.cross_counter.active = True
        self.cross_counts.clear()
        self.cross_counter.prev_sides.clear()
        self.cross_counter.counts = {"Entrada": defaultdict(int), "Salida": defaultdict(int)}
        self.request_paint_update()

    def disable_cross_line(self):
        self.cross_line_enabled = False
        self.cross_counter.active = False
        self.cross_counts.clear()
        self.cross_counter.prev_sides.clear()
        self.cross_counter.counts = {"Entrada": defaultdict(int), "Salida": defaultdict(int)}
        self.request_paint_update()

    def start_line_edit(self):
        self.enable_cross_line()
        self.cross_line_edit_mode = True
        self._temp_line_start = None
        self._dragging_line = None
        self._last_mouse_pos = None

    def finish_line_edit(self):
        self.cross_line_edit_mode = False
        self._temp_line_start = None
        self._dragging_line = None
        self._last_mouse_pos = None

    def _update_cross_counts(self, counts):
        self.cross_counts = counts
        self.request_paint_update()

    def _handle_cross_event(self, info):
        if self.last_frame is None:
            return
        now = datetime.now()
        fecha = now.strftime("%Y-%m-%d")
        hora = now.strftime("%H-%M-%S")
        ruta = os.path.join("capturas", "videos", fecha)
        os.makedirs(ruta, exist_ok=True)
        nombre = f"{fecha}_{hora}_{uuid.uuid4().hex[:6]}.mp4"
        path_final = os.path.join(ruta, nombre)
        frames_copy = list(self.frame_buffer)
        self.pending_videos.append({
            "frames": frames_copy,
            "frames_left": 50,
            "path": path_final,
        })
        self.registrar_log(f"🎥 Grabación iniciada: {nombre}")

    def perform_paint_update(self):
        self.paint_scheduled = False
        self.update() 

    def request_paint_update(self):
        if not self.paint_scheduled:
            self.paint_scheduled = True
            self.paint_update_timer.start(self.PAINT_UPDATE_INTERVAL)

    def _point_to_segment_distance(self, p, a, b):
        """Return the perpendicular distance from point p to line segment a-b."""
        ax, ay = a.x(), a.y()
        bx, by = b.x(), b.y()
        px, py = p.x(), p.y()
        dx = bx - ax
        dy = by - ay
        if dx == 0 and dy == 0:
            return (p - a).manhattanLength()
        t = ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy)
        t = max(0, min(1, t))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._generate_grid_lines_pixmap()

    def _generate_grid_lines_pixmap(self):
        if self.width() <= 0 or self.height() <= 0:
            self._grid_lines_pixmap = None
            return
        pixmap = QPixmap(self.size())
        pixmap.fill(Qt.GlobalColor.transparent)
        qp = QPainter(pixmap)
        qp.setPen(QColor(100, 100, 100, 100))
        cell_w = self.width() / self.columnas
        cell_h = self.height() / self.filas
        for row in range(self.filas + 1):
            y = row * cell_h
            qp.drawLine(0, int(y), self.width(), int(y))
        for col in range(self.columnas + 1):
            x = col * cell_w
            qp.drawLine(int(x), 0, int(x), self.height())
        qp.end()
        self._grid_lines_pixmap = pixmap

    def mostrar_vista(self, cam_data):
        if hasattr(self, 'visualizador') and self.visualizador: 
            self.visualizador.detener()

        if "rtsp" not in cam_data:
            cam_data["rtsp"] = generar_rtsp(cam_data)

        self.cam_data = cam_data
        self.discarded_cells = set() # Limpiar antes de cargar

        current_cam_ip = self.cam_data.get("ip")
        if current_cam_ip:
            try:
                with open(CONFIG_FILE_PATH, 'r') as f: # CONFIG_FILE_PATH will be defined at module level
                    config_data = json.load(f)
                
                camaras_config = config_data.get("camaras", [])
                for cam_config in camaras_config:
                    if cam_config.get("ip") == current_cam_ip:
                        discarded_list = cam_config.get("discarded_grid_cells")
                        if isinstance(discarded_list, list):
                            for cell_coords in discarded_list:
                                if isinstance(cell_coords, list) and len(cell_coords) == 2:
                                    self.discarded_cells.add(tuple(cell_coords))
                            self.registrar_log(f"Cargadas {len(self.discarded_cells)} celdas descartadas para {current_cam_ip}")
                        break 
            except FileNotFoundError:
                self.registrar_log(f"Archivo de configuración '{CONFIG_FILE_PATH}' no encontrado. Iniciando sin celdas descartadas predefinidas.")
            except json.JSONDecodeError:
                self.registrar_log(f"Error al decodificar '{CONFIG_FILE_PATH}'. Verifique el formato del archivo.")
            except Exception as e: # Captura genérica para otros posibles errores
                self.registrar_log(f"Error inesperado al cargar config: {e}")
        else:
            self.registrar_log("No se pudo obtener la IP de la cámara para cargar celdas descartadas.")
        
        self.alertas = GestorAlertas(cam_id=str(uuid.uuid4())[:8], filas=self.filas, columnas=self.columnas)

        self.visualizador  = VisualizadorDetector(cam_data)
        if self.visualizador:
            self.detector = getattr(self.visualizador, "detectors", [])

        self.visualizador.result_ready.connect(self.actualizar_boxes)
        self.visualizador.log_signal.connect(self.registrar_log)
        self.visualizador.iniciar()
        if self.visualizador and self.visualizador.video_sink:
             self.visualizador.video_sink.videoFrameChanged.connect(self.actualizar_pixmap_y_frame)

    def actualizar_boxes(self, boxes):
        self.latest_tracked_boxes = boxes
        if self.cross_line_enabled and self.original_frame_size:
            size = (
                self.original_frame_size.width(),
                self.original_frame_size.height(),
            )
            self.cross_counter.update_boxes(boxes, size)
        nuevas_detecciones_para_alertas = []
        for box_data in boxes:
            if not isinstance(box_data, dict):
                self.registrar_log(f"⚠️ formato de 'box' inesperado: {box_data}")
                continue

            x1, y1, x2, y2 = box_data.get('bbox', (0, 0, 0, 0))
            tracker_id = box_data.get('id')
            cls = box_data.get('cls')
            conf = box_data.get('conf', 0)
            
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            current_cls_positions = self.objetos_previos.get(cls, [])
            se_ha_movido = all(
                abs(cx - prev_cx) > self.umbral_movimiento or abs(cy - prev_cy) > self.umbral_movimiento
                for prev_cx, prev_cy in current_cls_positions
            )

            if se_ha_movido:
                nuevas_detecciones_para_alertas.append((x1, y1, x2, y2, cls)) 
                current_cls_positions.append((cx, cy))
                modelos_cam = []
                if self.cam_data:
                    modelos_cam = self.cam_data.get("modelos") or [self.cam_data.get("modelo")]
                if "Embarcaciones" in modelos_cam:
                    clase_nombre = "Embarcación"
                else:
                    clase_nombre = {0: "Persona", 2: "Auto", 9: "Barco"}.get(cls, f"Clase {cls}")
                conf_val = conf if isinstance(conf, (int, float)) else 0.0
                self.registrar_log(
                    f"🟢 Movimiento (ID: {tracker_id}) - {clase_nombre} ({conf_val:.2f}) en ({cx}, {cy})"
                )
            self.objetos_previos[cls] = current_cls_positions[-10:]

        if self.alertas and self.last_frame is not None:
            detecciones_filtradas = []
            if self.original_frame_size and self.original_frame_size.width() > 0 and self.original_frame_size.height() > 0:
                cell_w_video = self.original_frame_size.width() / self.columnas
                cell_h_video = self.original_frame_size.height() / self.filas

                if cell_w_video > 0 and cell_h_video > 0: # Avoid division by zero
                    if DEBUG_LOGS:
                        self.registrar_log(f"DEBUG: Celdas descartadas actualmente: {self.discarded_cells}")
                    for detection_data in nuevas_detecciones_para_alertas: # Assuming items are (x1, y1, x2, y2, cls)
                        x1_orig, y1_orig, x2_orig, y2_orig, cls_orig = detection_data
                        
                        # Calculate center of the detection in original video coordinates
                        cx_orig = (x1_orig + x2_orig) / 2
                        cy_orig = (y1_orig + y2_orig) / 2

                        # Determine the cell this detection falls into
                        # Ensure cx_orig and cy_orig are within bounds before division
                        if not (0 <= cx_orig < self.original_frame_size.width() and \
                                0 <= cy_orig < self.original_frame_size.height()):
                            detecciones_filtradas.append(detection_data) # Keep if center is out of bounds
                            continue

                        col_video = int(cx_orig / cell_w_video)
                        row_video = int(cy_orig / cell_h_video)
                        
                        # Clamp col_video and row_video to be within grid bounds
                        col_video = max(0, min(col_video, self.columnas - 1))
                        row_video = max(0, min(row_video, self.filas - 1))

                        if DEBUG_LOGS:
                            self.registrar_log(
                                f"DEBUG: Detección Original: {detection_data[:4]} Centro: ({cx_orig:.2f}, {cy_orig:.2f}) -> Celda: ({row_video}, {col_video})"
                            )
                        if (row_video, col_video) not in self.discarded_cells:
                            detecciones_filtradas.append(detection_data)
                            if DEBUG_LOGS:
                                self.registrar_log(
                                    f"DEBUG: Detección AÑADIDA a filtradas. Celda ({row_video}, {col_video}) NO está en descartadas."
                                )
                        else:
                            if DEBUG_LOGS:
                                self.registrar_log(
                                    f"DEBUG: Detección IGNORADA. Celda ({row_video}, {col_video}) ESTÁ en descartadas. Coords: ({x1_orig:.0f},{y1_orig:.0f})-({x2_orig:.0f},{y2_orig:.0f})"
                                )
                            pass 
                else: 
                    # Fallback if cell dimensions are zero (should not happen if width/height > 0)
                    detecciones_filtradas = list(nuevas_detecciones_para_alertas) 
            else: 
                # Fallback if original_frame_size is not available
                detecciones_filtradas = list(nuevas_detecciones_para_alertas)

            self.alertas.procesar_detecciones(
                detecciones_filtradas, 
                self.last_frame,
                self.registrar_log,
                self.cam_data
            )
            self.temporal = self.alertas.temporal
        
        self.request_paint_update() 

    def actualizar_pixmap_y_frame(self, frame):  # frame es QVideoFrame
        if not frame.isValid():
            return

        self.ui_frame_counter += 1

        if self.ui_frame_counter % self.UI_UPDATE_INTERVAL != 0:
            return

        image = None
        numpy_frame = None
        img_converted = None

        # Intentar leer los datos mapeando el frame directamente. Esto es
        # más eficiente cuando el formato de píxeles ya está en RGB o BGR.
        if frame.map(QVideoFrame.MapMode.ReadOnly):
            try:
                pf = frame.pixelFormat()
                rgb_formats = set()
                for name in [
                    "Format_RGB24",
                    "Format_RGB32",
                    "Format_BGR24",
                    "Format_BGR32",
                    "Format_RGBX8888",
                    "Format_RGBA8888",
                    "Format_BGRX8888",
                    "Format_BGRA8888",
                    "Format_ARGB32",
                ]:
                    fmt = getattr(QVideoFrameFormat.PixelFormat, name, None)
                    if fmt is not None:
                        rgb_formats.add(fmt)

                if pf in rgb_formats:
                    img_format = QVideoFrameFormat.imageFormatFromPixelFormat(pf)
                    if img_format != QImage.Format.Format_Invalid:
                        qimg = QImage(
                            frame.bits(),
                            frame.width(),
                            frame.height(),
                            frame.bytesPerLine(),
                            img_format,
                        ).copy()
                        image = qimg
                        img_converted = qimg.convertToFormat(QImage.Format.Format_RGB888)
                        ptr = img_converted.constBits()
                        ptr.setsize(img_converted.width() * img_converted.height() * 3)
                        numpy_frame = (
                            np.frombuffer(ptr, dtype=np.uint8)
                            .reshape((img_converted.height(), img_converted.width(), 3))
                            .copy()
                        )
            finally:
                frame.unmap()

        # Si la lectura mapeada no fue posible, usar el método tradicional.
        if image is None:
            image = frame.toImage()
            if image.isNull():
                return
            img_converted = image.convertToFormat(QImage.Format.Format_RGB888)
            ptr = img_converted.constBits()
            ptr.setsize(img_converted.width() * img_converted.height() * 3)
            numpy_frame = (
                np.frombuffer(ptr, dtype=np.uint8)
                .reshape((img_converted.height(), img_converted.width(), 3))
                .copy()
            )

        current_frame_width = img_converted.width()
        current_frame_height = img_converted.height()
        if (
            self.original_frame_size is None
            or self.original_frame_size.width() != current_frame_width
            or self.original_frame_size.height() != current_frame_height
        ):
            self.original_frame_size = QSize(current_frame_width, current_frame_height)

        self.last_frame = numpy_frame
        self.frame_buffer.append(numpy_frame)
        for rec in list(self.pending_videos):
            if rec["frames_left"] > 0:
                rec["frames"].append(numpy_frame)
                rec["frames_left"] -= 1
            if rec["frames_left"] <= 0:
                thread = VideoSaverThread(rec["frames"], rec["path"], fps=10)
                thread.finished.connect(lambda r=thread: self._remove_video_thread(r))
                self.active_video_threads.append(thread)
                thread.start()
                self.pending_videos.remove(rec)
                self.registrar_log(f"🎥 Video guardado: {os.path.basename(rec['path'])}")

        self.pixmap = QPixmap.fromImage(img_converted)
        self.request_paint_update()

    def registrar_log(self, mensaje):
        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ip = self.cam_data.get("ip", "IP-desconocida") if self.cam_data else "IP-indefinida"
        mensaje_completo = f"[{fecha_hora}] Cámara {ip}: {mensaje}"
        self.log_signal.emit(mensaje_completo)
        with open("eventos_detectados.txt", "a", encoding="utf-8") as f:
            f.write(mensaje_completo + "\n")

    def _remove_video_thread(self, thread):
        if thread in self.active_video_threads:
            self.active_video_threads.remove(thread)

    def detener(self):
        if hasattr(self, 'visualizador') and self.visualizador: 
            self.visualizador.detener()
            # self.visualizador = None # Keep this for later, stop processor first
        
        if hasattr(self, 'analytics_processor') and self.analytics_processor:
            self.analytics_processor.stop_processing()
            # self.analytics_processor = None # Optionally, also set to None

        if hasattr(self, 'visualizador') and self.visualizador: # Re-check as visualizador might be used by processor
            self.visualizador = None
        if self.detector:
             self.detector = None
        self.paint_update_timer.stop()
        if hasattr(self, 'cross_counter') and self.cross_counter:
            self.cross_counter.stop()
        for th in list(self.active_video_threads):
            if th.isRunning():
                th.wait(1000)

    def mousePressEvent(self, event):
        if self.cross_line_edit_mode:
            if event.button() == Qt.MouseButton.LeftButton:
                pos = event.position()
                x_rel = pos.x() / self.width()
                y_rel = pos.y() / self.height()
                x1_rel, y1_rel = self.cross_counter.line[0]
                x2_rel, y2_rel = self.cross_counter.line[1]
                p1 = QPointF(x1_rel * self.width(), y1_rel * self.height())
                p2 = QPointF(x2_rel * self.width(), y2_rel * self.height())
                thresh = 10.0
                if (pos - p1).manhattanLength() <= thresh:
                    self._dragging_line = 'p1'
                elif (pos - p2).manhattanLength() <= thresh:
                    self._dragging_line = 'p2'
                elif self._point_to_segment_distance(pos, p1, p2) <= thresh:
                    self._dragging_line = 'line'
                    self._last_mouse_pos = pos
                else:
                    self._dragging_line = 'new'
                    self._temp_line_start = pos
                    self.cross_counter.set_line(((x_rel, y_rel), (x_rel, y_rel)))
                self.request_paint_update()
            elif event.button() == Qt.MouseButton.RightButton:
                self.finish_line_edit()
            return
        pos = event.pos()
        cell_w = self.width() / self.columnas
        cell_h = self.height() / self.filas

        if cell_w == 0 or cell_h == 0:
            return

        col = int(pos.x() / cell_w)
        row = int(pos.y() / cell_h)

        if not (0 <= row < self.filas and 0 <= col < self.columnas):
            return

        clicked_cell = (row, col)

        if event.button() == Qt.MouseButton.LeftButton:
            if clicked_cell in self.selected_cells:
                self.selected_cells.remove(clicked_cell)
            else:
                self.selected_cells.add(clicked_cell)
            self.request_paint_update()
        elif event.button() == Qt.MouseButton.RightButton:
            menu = QMenu(self)
            if self.selected_cells:
                discard_action = menu.addAction("Descartar celdas para analíticas")
                discard_action.triggered.connect(self.handle_discard_cells)

                enable_action = menu.addAction("Habilitar celdas para analíticas")
                enable_action.triggered.connect(self.handle_enable_discarded_cells)

            if self.cross_line_enabled:
                disable_line = menu.addAction("Desactivar línea de conteo")
                disable_line.triggered.connect(self.disable_cross_line)
            else:
                enable_line = menu.addAction("Activar línea de conteo")
                enable_line.triggered.connect(self.start_line_edit)

            menu.exec(event.globalPosition().toPoint())


    def handle_discard_cells(self):
        if not self.selected_cells: # Guard clause, though menu shouldn't appear if empty
            return

        self.discarded_cells.update(self.selected_cells)
        # Guardar la configuración
        self._save_discarded_cells_to_config() 
        self.selected_cells.clear()
        self.request_paint_update() # Request repaint to reflect changes

    def handle_enable_discarded_cells(self):
        if not self.selected_cells:
            return

        cells_to_enable = self.selected_cells.intersection(self.discarded_cells)
        if not cells_to_enable:
            # Si ninguna de las celdas seleccionadas está actualmente descartada,
            # podríamos opcionalmente limpiar la selección o no hacer nada.
            # Por ahora, limpiamos la selección para consistencia con handle_discard_cells.
            self.selected_cells.clear()
            self.request_paint_update()
            return

        for cell in cells_to_enable:
            self.discarded_cells.remove(cell)
        
        self.log_signal.emit(f"Celdas habilitadas para analíticas: {len(cells_to_enable)}") # Opcional: registrar log
        # Guardar la configuración
        self._save_discarded_cells_to_config()
        self.selected_cells.clear()
        self.request_paint_update()

    def mouseMoveEvent(self, event):
        if self.cross_line_edit_mode and self._dragging_line:
            pos = event.position()
            x_rel = pos.x() / self.width()
            y_rel = pos.y() / self.height()
            if self._dragging_line == 'new':
                rel_start = (
                    self._temp_line_start.x() / self.width(),
                    self._temp_line_start.y() / self.height(),
                )
                self.cross_counter.set_line((rel_start, (x_rel, y_rel)))
            elif self._dragging_line == 'p1':
                _, p2 = self.cross_counter.line
                self.cross_counter.set_line(((x_rel, y_rel), p2))
            elif self._dragging_line == 'p2':
                p1, _ = self.cross_counter.line
                self.cross_counter.set_line((p1, (x_rel, y_rel)))
            elif self._dragging_line == 'line' and self._last_mouse_pos is not None:
                dx = (pos.x() - self._last_mouse_pos.x()) / self.width()
                dy = (pos.y() - self._last_mouse_pos.y()) / self.height()
                x1_rel, y1_rel = self.cross_counter.line[0]
                x2_rel, y2_rel = self.cross_counter.line[1]
                self.cross_counter.set_line(
                    (
                        (x1_rel + dx, y1_rel + dy),
                        (x2_rel + dx, y2_rel + dy),
                    )
                )
                self._last_mouse_pos = pos
            self.request_paint_update()

    def mouseReleaseEvent(self, event):
        if self.cross_line_edit_mode and self._dragging_line:
            if self._dragging_line == 'new':
                pos = event.position()
                rel_start = (
                    self._temp_line_start.x() / self.width(),
                    self._temp_line_start.y() / self.height(),
                )
                rel_end = (
                    pos.x() / self.width(),
                    pos.y() / self.height(),
                )
                self.cross_counter.set_line((rel_start, rel_end))
            self._dragging_line = None
            self._temp_line_start = None
            self._last_mouse_pos = None
            self.request_paint_update()
        else:
            super().mouseReleaseEvent(event)

    def _save_discarded_cells_to_config(self):
        if not self.cam_data or not self.cam_data.get("ip"):
            self.registrar_log("Error: No se pudo obtener la IP de la cámara para guardar la configuración de celdas descartadas.")
            return

        current_cam_ip = self.cam_data.get("ip")
        # Convertir set de tuplas a lista de listas para JSON
        discarded_list_for_json = sorted([list(cell) for cell in self.discarded_cells]) # Ordenar para consistencia

        config_data = None
        try:
            with open(CONFIG_FILE_PATH, 'r') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            self.registrar_log(f"Archivo '{CONFIG_FILE_PATH}' no encontrado. Se intentará crear uno nuevo o se asumirá una lista de cámaras vacía si no se puede cargar.")
            config_data = {"camaras": [], "configuracion": {}} # Estructura base
        except json.JSONDecodeError:
            self.registrar_log(f"Error crítico: El archivo de configuración '{CONFIG_FILE_PATH}' está corrupto. No se guardarán los cambios para evitar mayor pérdida de datos.")
            return # No continuar si el JSON está corrupto
        except Exception as e:
            self.registrar_log(f"Error inesperado al leer '{CONFIG_FILE_PATH}': {e}")
            return


        camara_encontrada = False
        if "camaras" not in config_data: # Asegurar que la clave "camaras" exista
            config_data["camaras"] = []

        for cam_config in config_data["camaras"]:
            if cam_config.get("ip") == current_cam_ip:
                cam_config["discarded_grid_cells"] = discarded_list_for_json
                camara_encontrada = True
                break
        
        if not camara_encontrada:
            new_cam_entry = self.cam_data.copy() 
            new_cam_entry["discarded_grid_cells"] = discarded_list_for_json
            config_data["camaras"].append(new_cam_entry)
            self.registrar_log(f"Cámara {current_cam_ip} no encontrada en config.json. Se añadió una nueva entrada.")


        try:
            with open(CONFIG_FILE_PATH, 'w') as f:
                json.dump(config_data, f, indent=4)
            self.registrar_log(f"Configuración de celdas descartadas guardada para la cámara {current_cam_ip} en '{CONFIG_FILE_PATH}'.")
        except Exception as e:
            self.registrar_log(f"Error al escribir la configuración en '{CONFIG_FILE_PATH}': {e}")



    def paintEvent(self, event):
        super().paintEvent(event) 
        qp = QPainter(self)
        if not self.pixmap or self.pixmap.isNull():
            qp.fillRect(self.rect(), QColor("black"))
            qp.setPen(QColor("white"))
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sin señal")
            return

        target_rect = QRectF(self.rect()) 
        pixmap_size = QSizeF(self.pixmap.size()) 
        scaled_pixmap_size = pixmap_size.scaled(target_rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
        final_draw_rect = QRectF()
        final_draw_rect.setSize(scaled_pixmap_size) 
        final_draw_rect.moveCenter(target_rect.center())
        qp.drawPixmap(final_draw_rect, self.pixmap, QRectF(self.pixmap.rect()))

        cell_w = self.width() / self.columnas
        cell_h = self.height() / self.filas
        for row in range(self.filas):
            for col in range(self.columnas):
                index = row * self.columnas + col
                estado_area = self.area[index] if index < len(self.area) else 0
                cell_tuple = (row, col)
                brush_color = None
                if cell_tuple in self.discarded_cells:
                    brush_color = QColor(200, 0, 0, 150)
                elif cell_tuple in self.selected_cells:
                    brush_color = QColor(255, 0, 0, 100)
                elif index in self.temporal:
                    brush_color = QColor(0, 255, 0, 100)
                elif estado_area == 1:
                    brush_color = QColor(255, 165, 0, 100)
                if brush_color is not None:
                    rect_to_draw = QRectF(col * cell_w, row * cell_h, cell_w, cell_h)
                    qp.fillRect(rect_to_draw, brush_color)

        if self._grid_lines_pixmap:
            qp.drawPixmap(self.rect(), self._grid_lines_pixmap)

        if self.latest_tracked_boxes and self.original_frame_size:
            orig_frame_w = self.original_frame_size.width()
            orig_frame_h = self.original_frame_size.height()
            if orig_frame_w == 0 or orig_frame_h == 0: return

            scale_x = final_draw_rect.width() / orig_frame_w
            scale_y = final_draw_rect.height() / orig_frame_h
            offset_x = final_draw_rect.left()
            offset_y = final_draw_rect.top()
            font = QFont()
            font.setPointSize(10)
            qp.setFont(font)

            for box_data in self.latest_tracked_boxes:
                if not isinstance(box_data, dict):
                    continue

                x1, y1, x2, y2 = box_data.get('bbox', (0, 0, 0, 0))
                tracker_id = box_data.get('id')
                conf = box_data.get('conf')
                conf_val = conf if isinstance(conf, (int, float)) else 0.0

                scaled_x1 = (x1 * scale_x) + offset_x
                scaled_y1 = (y1 * scale_y) + offset_y
                scaled_x2 = (x2 * scale_x) + offset_x
                scaled_y2 = (y2 * scale_y) + offset_y
                scaled_w = scaled_x2 - scaled_x1
                scaled_h = scaled_y2 - scaled_y1
                pen = QPen()
                pen.setWidth(2)
                pen.setColor(QColor("blue"))
                qp.setPen(pen)
                qp.setBrush(Qt.BrushStyle.NoBrush)
                qp.drawRect(QRectF(scaled_x1, scaled_y1, scaled_w, scaled_h))
                moving_state = box_data.get('moving')
                if moving_state is None:
                    estado = 'Procesando'
                else:
                    estado = 'En movimiento' if moving_state else 'Detenido'
                label_text = f"ID:{tracker_id} C:{conf_val:.2f} {estado}"
                qp.setPen(QColor("white"))
                text_x = scaled_x1
                text_y = scaled_y1 - 5

                if text_y < 10 : text_y = scaled_y1 + 15
                qp.drawText(QPointF(text_x, text_y), label_text)
        if hasattr(self, 'cross_counter') and self.cross_line_enabled:
            x1_rel, y1_rel = self.cross_counter.line[0]
            x2_rel, y2_rel = self.cross_counter.line[1]
            pen = QPen(QColor('yellow'))
            pen.setWidth(2)
            qp.setPen(pen)
            qp.drawLine(
                QPointF(x1_rel * self.width(), y1_rel * self.height()),
                QPointF(x2_rel * self.width(), y2_rel * self.height()),
            )
            if self.cross_line_edit_mode:
                handle_pen = QPen(QColor('red'))
                handle_pen.setWidth(4)
                qp.setPen(handle_pen)
                qp.setBrush(QBrush(QColor('red')))
                size = 6
                qp.drawEllipse(QPointF(x1_rel * self.width(), y1_rel * self.height()), size, size)
                qp.drawEllipse(QPointF(x2_rel * self.width(), y2_rel * self.height()), size, size)
            counts_parts = []
            for direc in ("Entrada", "Salida"):
                sub = self.cross_counts.get(direc, {})
                if sub:
                    sub_text = ", ".join(f"{v} {k}" for k, v in sub.items())
                    counts_parts.append(f"{direc}: {sub_text}")
            counts_text = " | ".join(counts_parts)
            if counts_text:
                qp.setPen(QColor('yellow'))
                qp.drawText(QPointF(x2_rel * self.width() + 5, y2_rel * self.height()), counts_text)
        else:
            pass