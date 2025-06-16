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

DEBUG_LOGS = False  # Deshabilitado para producción

CONFIG_FILE_PATH = "config.json"

class GrillaWidget(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self, filas=18, columnas=22, area=None, parent=None, fps_config=None):
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
        self.analytics_processor = AnalyticsProcessor(self)

        # Configuración de FPS personalizable
        if fps_config is None:
            fps_config = {
                "visual_fps": 25,
                "detection_fps": 8,
                "ui_update_fps": 15
            }
        
        self.fps_config = fps_config
        
        # Calcular intervalos basados en FPS deseados
        self.PAINT_UPDATE_INTERVAL = int(1000 / fps_config["ui_update_fps"])
        self.UI_UPDATE_INTERVAL = max(1, int(30 / fps_config["visual_fps"]))

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

        self.frame_buffer = deque(maxlen=50)
        self.pending_videos = []
        self.active_video_threads = []

        self.setFixedSize(640, 480)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._grid_lines_pixmap = None
        self._generate_grid_lines_pixmap()

        self.paint_update_timer = QTimer(self)
        self.paint_update_timer.setSingleShot(True)
        self.paint_update_timer.timeout.connect(self.perform_paint_update)
        self.paint_scheduled = False

        self.ui_frame_counter = 1
        
        # Contadores simplificados
        self.detection_count = 0

    def set_fps_config(self, visual_fps=25, detection_fps=8, ui_update_fps=15):
        """Actualizar configuración de FPS en tiempo real"""
        self.fps_config = {
            "visual_fps": visual_fps,
            "detection_fps": detection_fps, 
            "ui_update_fps": ui_update_fps
        }
        
        self.PAINT_UPDATE_INTERVAL = int(1000 / ui_update_fps)
        self.UI_UPDATE_INTERVAL = max(1, int(30 / visual_fps))
        
        if hasattr(self, 'visualizador') and self.visualizador:
            self.visualizador.update_fps_config(visual_fps, detection_fps)
        
        self.registrar_log(f"🎯 FPS actualizado - Visual: {visual_fps}, Detección: {detection_fps}, UI: {ui_update_fps}")

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

        if "fps_config" not in cam_data:
            cam_data["fps_config"] = self.fps_config

        self.cam_data = cam_data
        self.discarded_cells = set()

        # Cargar celdas descartadas
        current_cam_ip = self.cam_data.get("ip")
        if current_cam_ip:
            try:
                with open(CONFIG_FILE_PATH, 'r') as f:
                    config_data = json.load(f)
                
                camaras_config = config_data.get("camaras", [])
                for cam_config in camaras_config:
                    if cam_config.get("ip") == current_cam_ip:
                        discarded_list = cam_config.get("discarded_grid_cells")
                        if isinstance(discarded_list, list):
                            for cell_coords in discarded_list:
                                if isinstance(cell_coords, list) and len(cell_coords) == 2:
                                    self.discarded_cells.add(tuple(cell_coords))
                            self.registrar_log(f"Cargadas {len(self.discarded_cells)} celdas descartadas")
                        break 
            except Exception as e:
                self.registrar_log(f"Error cargando configuración: {e}")
        
        self.alertas = GestorAlertas(cam_id=str(uuid.uuid4())[:8], filas=self.filas, columnas=self.columnas)

        self.visualizador = VisualizadorDetector(cam_data)
        if self.visualizador:
            self.detector = getattr(self.visualizador, "detectors", [])

        self.visualizador.result_ready.connect(self.actualizar_boxes)
        self.visualizador.log_signal.connect(self.registrar_log)
        self.visualizador.iniciar()
        
        if self.visualizador and self.visualizador.video_sink:
            self.visualizador.video_sink.videoFrameChanged.connect(self.actualizar_pixmap_y_frame)
            
        self.registrar_log(f"🎥 Vista configurada para {current_cam_ip}")

    def actualizar_boxes(self, boxes):
        """Método principal que recibe las detecciones del visualizador"""
        self.detection_count += 1
        
        # Solo mostrar log cada 50 detecciones para evitar spam
        if DEBUG_LOGS and self.detection_count % 50 == 0:
            self.registrar_log(f"📊 Detecciones procesadas: {self.detection_count}")
        
        self.latest_tracked_boxes = boxes
        
        if self.cross_line_enabled and self.original_frame_size:
            size = (
                self.original_frame_size.width(),
                self.original_frame_size.height(),
            )
            self.cross_counter.update_boxes(boxes, size)
        
        # Procesar detecciones para alertas
        nuevas_detecciones_para_alertas = []
        for box_data in boxes:
            if not isinstance(box_data, dict):
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
                    clase_nombre = {0: "Persona", 2: "Auto", 8: "Barco", 9: "Barco"}.get(cls, f"Clase {cls}")
                conf_val = conf if isinstance(conf, (int, float)) else 0.0
                
                # Solo mostrar movimientos significativos
                self.registrar_log(f"🟢 {clase_nombre} detectada (ID: {tracker_id}, Conf: {conf_val:.2f})")
                    
            self.objetos_previos[cls] = current_cls_positions[-10:]

        # Procesar alertas
        if self.alertas and self.last_frame is not None:
            detecciones_filtradas = []
            if self.original_frame_size and self.original_frame_size.width() > 0 and self.original_frame_size.height() > 0:
                cell_w_video = self.original_frame_size.width() / self.columnas
                cell_h_video = self.original_frame_size.height() / self.filas

                if cell_w_video > 0 and cell_h_video > 0:
                    for detection_data in nuevas_detecciones_para_alertas:
                        x1_orig, y1_orig, x2_orig, y2_orig, cls_orig = detection_data
                        
                        cx_orig = (x1_orig + x2_orig) / 2
                        cy_orig = (y1_orig + y2_orig) / 2

                        if not (0 <= cx_orig < self.original_frame_size.width() and \
                                0 <= cy_orig < self.original_frame_size.height()):
                            detecciones_filtradas.append(detection_data)
                            continue

                        col_video = int(cx_orig / cell_w_video)
                        row_video = int(cy_orig / cell_h_video)
                        
                        col_video = max(0, min(col_video, self.columnas - 1))
                        row_video = max(0, min(row_video, self.filas - 1))

                        if (row_video, col_video) not in self.discarded_cells:
                            detecciones_filtradas.append(detection_data)
                else: 
                    detecciones_filtradas = list(nuevas_detecciones_para_alertas) 
            else: 
                detecciones_filtradas = list(nuevas_detecciones_para_alertas)

            self.alertas.procesar_detecciones(
                detecciones_filtradas, 
                self.last_frame,
                self.registrar_log,
                self.cam_data
            )
            self.temporal = self.alertas.temporal
        
        self.request_paint_update() 

    def actualizar_pixmap_y_frame(self, frame):
        if not frame.isValid():
            return

        self.ui_frame_counter += 1

        if self.ui_frame_counter % self.UI_UPDATE_INTERVAL != 0:
            return

        image = None
        numpy_frame = None
        img_converted = None

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
        
        if hasattr(self, 'analytics_processor') and self.analytics_processor:
            self.analytics_processor.stop_processing()

        if hasattr(self, 'visualizador') and self.visualizador:
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
        if not self.selected_cells:
            return

        self.discarded_cells.update(self.selected_cells)
        self._save_discarded_cells_to_config() 
        self.selected_cells.clear()
        self.request_paint_update()

    def handle_enable_discarded_cells(self):
        if not self.selected_cells:
            return

        cells_to_enable = self.selected_cells.intersection(self.discarded_cells)
        if not cells_to_enable:
            self.selected_cells.clear()
            self.request_paint_update()
            return

        for cell in cells_to_enable:
            self.discarded_cells.remove(cell)
        
        self.registrar_log(f"Celdas habilitadas: {len(cells_to_enable)}")
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
            self.registrar_log("Error: No se pudo obtener la IP de la cámara")
            return

        current_cam_ip = self.cam_data.get("ip")
        discarded_list_for_json = sorted([list(cell) for cell in self.discarded_cells])

        config_data = None
        try:
            with open(CONFIG_FILE_PATH, 'r') as f:
                config_data = json.load(f)
        except FileNotFoundError:
            config_data = {"camaras": [], "configuracion": {}}
        except json.JSONDecodeError:
            self.registrar_log("Error: Archivo de configuración corrupto")
            return
        except Exception as e:
            self.registrar_log(f"Error leyendo configuración: {e}")
            return

        camara_encontrada = False
        if "camaras" not in config_data:
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

        try:
            with open(CONFIG_FILE_PATH, 'w') as f:
                json.dump(config_data, f, indent=4)
            self.registrar_log("✅ Configuración guardada")
        except Exception as e:
            self.registrar_log(f"Error guardando configuración: {e}")

    def paintEvent(self, event):
        super().paintEvent(event) 
        qp = QPainter(self)
        
        if not self.pixmap or self.pixmap.isNull():
            qp.fillRect(self.rect(), QColor("black"))
            qp.setPen(QColor("white"))
            qp.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sin señal")
            return

        # Calcular el área donde se dibuja el video (manteniendo aspect ratio)
        widget_rect = QRectF(self.rect())
        pixmap_size = QSizeF(self.pixmap.size())
        
        # Calcular el tamaño escalado manteniendo aspect ratio
        scaled_size = pixmap_size.scaled(widget_rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
        
        # Centrar el video en el widget
        video_rect = QRectF()
        video_rect.setSize(scaled_size)
        video_rect.moveCenter(widget_rect.center())
        
        # Dibujar el video
        qp.drawPixmap(video_rect, self.pixmap, QRectF(self.pixmap.rect()))

        # Dibujar la grilla de celdas
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

        # Dibujar líneas de la grilla
        if self._grid_lines_pixmap:
            qp.drawPixmap(self.rect(), self._grid_lines_pixmap)

        # Dibujar las cajas de detección con coordenadas corregidas
        if self.latest_tracked_boxes and self.original_frame_size:
            orig_frame_w = self.original_frame_size.width()
            orig_frame_h = self.original_frame_size.height()
            
            if orig_frame_w == 0 or orig_frame_h == 0:
                return

            # CORRECCIÓN CLAVE: Usar las dimensiones del área real del video, no del widget completo
            scale_x = video_rect.width() / orig_frame_w
            scale_y = video_rect.height() / orig_frame_h
            offset_x = video_rect.left()
            offset_y = video_rect.top()
            
            font = QFont()
            font.setPointSize(10)
            qp.setFont(font)

            for box_data in self.latest_tracked_boxes:
                if not isinstance(box_data, dict):
                    continue

                bbox = box_data.get('bbox', (0, 0, 0, 0))
                if len(bbox) != 4:
                    continue
                    
                x1, y1, x2, y2 = bbox
                tracker_id = box_data.get('id', 'N/A')
                conf = box_data.get('conf', 0)
                conf_val = conf if isinstance(conf, (int, float)) else 0.0

                # Aplicar escalado y offset correctos
                scaled_x1 = (x1 * scale_x) + offset_x
                scaled_y1 = (y1 * scale_y) + offset_y
                scaled_x2 = (x2 * scale_x) + offset_x
                scaled_y2 = (y2 * scale_y) + offset_y
                
                scaled_w = scaled_x2 - scaled_x1
                scaled_h = scaled_y2 - scaled_y1
                
                # Verificar que las coordenadas estén dentro del área del video
                if (scaled_x1 < video_rect.left() or scaled_y1 < video_rect.top() or 
                    scaled_x2 > video_rect.right() or scaled_y2 > video_rect.bottom()):
                    # Las coordenadas están fuera del área del video, skip
                    continue
                
                # Dibujar el rectángulo de detección
                pen = QPen()
                pen.setWidth(3)
                pen.setColor(QColor("lime"))  # Verde más visible
                qp.setPen(pen)
                qp.setBrush(Qt.BrushStyle.NoBrush)
                qp.drawRect(QRectF(scaled_x1, scaled_y1, scaled_w, scaled_h))
                
                # Determinar estado de movimiento
                moving_state = box_data.get('moving')
                if moving_state is None:
                    estado = 'Procesando'
                else:
                    estado = 'En movimiento' if moving_state else 'Detenido'
                
                # Preparar texto de la etiqueta
                label_text = f"ID:{tracker_id} C:{conf_val:.2f} {estado}"
                
                # Dibujar fondo para el texto
                text_rect = qp.fontMetrics().boundingRect(label_text)
                text_bg_rect = QRectF(scaled_x1, scaled_y1 - text_rect.height() - 2, 
                                     text_rect.width() + 4, text_rect.height() + 2)
                
                # Si el texto se sale por arriba, ponerlo abajo del box
                if text_bg_rect.top() < video_rect.top():
                    text_bg_rect.moveTop(scaled_y2 + 2)
                
                qp.fillRect(text_bg_rect, QColor(0, 0, 0, 180))  # Fondo semi-transparente
                
                # Dibujar el texto
                qp.setPen(QColor("white"))
                text_x = text_bg_rect.left() + 2
                text_y = text_bg_rect.bottom() - 2
                qp.drawText(QPointF(text_x, text_y), label_text)

        # Dibujar línea de conteo si está activa
        if hasattr(self, 'cross_counter') and self.cross_line_enabled:
            x1_rel, y1_rel = self.cross_counter.line[0]
            x2_rel, y2_rel = self.cross_counter.line[1]
            
            pen = QPen(QColor('yellow'))
            pen.setWidth(3)
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
            
            # Mostrar conteos
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
