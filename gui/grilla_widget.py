from PyQt6.QtWidgets import QWidget, QSizePolicy, QMenu
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QFont, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QSizeF, QSize, QPointF, QTimer 
from gui.visualizador_detector import VisualizadorDetector
from core.gestor_alertas import GestorAlertas
from core.rtsp_builder import generar_rtsp
from core.analytics_processor import AnalyticsProcessor
from gui.image_saver import ImageSaverThread 
import numpy as np
from datetime import datetime
import uuid

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
        self.detector = None 
        self.analytics_processor = AnalyticsProcessor(self) # Pass self as parent for Qt object management
        # You might want to connect its signals if you were using it actively
        # self.analytics_processor.log_signal.connect(self.registrar_log) 
        # self.analytics_processor.processing_finished.connect(self.handle_analytics_results)

        self.setFixedSize(640, 480) 
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Atributos para la optimización de paintEvent
        self.PAINT_UPDATE_INTERVAL = 40  # ms, para aprox. 25 FPS de UI
        self.paint_update_timer = QTimer(self)
        self.paint_update_timer.setSingleShot(True)
        self.paint_update_timer.timeout.connect(self.perform_paint_update)
        self.paint_scheduled = False

        # Atributos para limitar la frecuencia de actualización de la UI por frames de video
        self.ui_frame_counter = 0
        self.UI_UPDATE_INTERVAL = 2  # Procesar 1 de cada N frames para la UI

    def perform_paint_update(self):
        self.paint_scheduled = False
        self.update() 

    def request_paint_update(self):
        if not self.paint_scheduled:
            self.paint_scheduled = True
            self.paint_update_timer.start(self.PAINT_UPDATE_INTERVAL)

    def mostrar_vista(self, cam_data):
        if hasattr(self, 'visualizador') and self.visualizador: 
            self.visualizador.detener()

        if "rtsp" not in cam_data:
            cam_data["rtsp"] = generar_rtsp(cam_data)

        self.cam_data = cam_data
        self.alertas = GestorAlertas(cam_id=str(uuid.uuid4())[:8], filas=self.filas, columnas=self.columnas)

        self.visualizador = VisualizadorDetector(cam_data)
        if self.visualizador:
            self.detector = self.visualizador.detector 

        self.visualizador.result_ready.connect(self.actualizar_boxes)
        self.visualizador.log_signal.connect(self.registrar_log)
        self.visualizador.iniciar()
        if self.visualizador and self.visualizador.video_sink:
             self.visualizador.video_sink.videoFrameChanged.connect(self.actualizar_pixmap_y_frame)

    def actualizar_boxes(self, boxes): 
        self.latest_tracked_boxes = boxes 
        nuevas_detecciones_para_alertas = [] 
        for box_data in boxes:
            if len(box_data) == 8:
                x1, y1, x2, y2, tracker_id, cls, conf, es_prediccion = box_data
            elif len(box_data) == 7: 
                x1, y1, x2, y2, tracker_id, cls, conf = box_data
                es_prediccion = False 
            else:
                self.registrar_log(f"⚠️ formato de 'box' inesperado: {len(box_data)} elementos. Contenido: {box_data}")
                continue
            
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
                if self.cam_data and self.cam_data.get("modelo") == "Embarcaciones":
                    clase_nombre = "Embarcación"
                else:
                    clase_nombre = {0: "Persona", 2: "Auto", 9: "Barco"}.get(cls, f"Clase {cls}")
                log_msg_detail = "PRED" if es_prediccion else "DET"
                self.registrar_log(f"🟢 Movimiento {log_msg_detail} (ID: {tracker_id}) - {clase_nombre} ({conf:.2f}) en ({cx}, {cy})")
            self.objetos_previos[cls] = current_cls_positions[-10:]

        if self.alertas and self.last_frame is not None:
            detecciones_filtradas = []
            if self.original_frame_size and self.original_frame_size.width() > 0 and self.original_frame_size.height() > 0:
                cell_w_video = self.original_frame_size.width() / self.columnas
                cell_h_video = self.original_frame_size.height() / self.filas

                if cell_w_video > 0 and cell_h_video > 0: # Avoid division by zero
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

                        if (row_video, col_video) not in self.discarded_cells:
                            detecciones_filtradas.append(detection_data)
                        else:
                            # Optional: Log that a detection was ignored
                            # self.registrar_log(f"ℹ️ Detección en ({x1_orig:.0f},{y1_orig:.0f})-({x2_orig:.0f},{y2_orig:.0f}) ignorada en celda descartada ({row_video}, {col_video})")
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

    def actualizar_pixmap_y_frame(self, frame): # frame es QVideoFrame
        if not frame.isValid():
            return

        self.ui_frame_counter += 1

        if self.ui_frame_counter % self.UI_UPDATE_INTERVAL == 0:
            image = frame.toImage() 
            if not image.isNull():
                img_converted = image.convertToFormat(QImage.Format.Format_RGB888)
                
                current_frame_width = img_converted.width()
                current_frame_height = img_converted.height()
                if self.original_frame_size is None or \
                   self.original_frame_size.width() != current_frame_width or \
                   self.original_frame_size.height() != current_frame_height:
                    self.original_frame_size = QSize(current_frame_width, current_frame_height) 
                
                ptr = img_converted.constBits()
                ptr.setsize(img_converted.width() * img_converted.height() * 3)

                arr = np.frombuffer(ptr, dtype=np.uint8).reshape(
                    (img_converted.height(), img_converted.width(), 3)
                ).copy()
                self.last_frame = arr 

                self.pixmap = QPixmap.fromImage(img_converted) 
                self.request_paint_update() 

    def registrar_log(self, mensaje):
        fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ip = self.cam_data.get("ip", "IP-desconocida") if self.cam_data else "IP-indefinida"
        mensaje_completo = f"[{fecha_hora}] Cámara {ip}: {mensaje}"
        self.log_signal.emit(mensaje_completo)
        with open("eventos_detectados.txt", "a", encoding="utf-8") as f:
            f.write(mensaje_completo + "\n")

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

    def mousePressEvent(self, event):
        pos = event.pos()
        cell_w = self.width() / self.columnas
        cell_h = self.height() / self.filas

        if cell_w == 0 or cell_h == 0: # Avoid division by zero if widget not fully initialized
            return

        col = int(pos.x() / cell_w)
        row = int(pos.y() / cell_h)

        if not (0 <= row < self.filas and 0 <= col < self.columnas): # Click is outside grid
            return

        clicked_cell = (row, col)

        if event.button() == Qt.MouseButton.LeftButton:
            if clicked_cell in self.selected_cells:
                self.selected_cells.remove(clicked_cell)
            else:
                self.selected_cells.add(clicked_cell)
            self.request_paint_update() # Request repaint to show selection
        elif event.button() == Qt.MouseButton.RightButton:
            if self.selected_cells: # Only show menu if there are selected cells
                menu = QMenu(self)
                discard_action = menu.addAction("Descartar celdas para analíticas")
                discard_action.triggered.connect(self.handle_discard_cells)

                # Nueva acción para habilitar celdas
                enable_action = menu.addAction("Habilitar celdas para analíticas")
                enable_action.triggered.connect(self.handle_enable_discarded_cells)

                menu.exec(event.globalPosition().toPoint())


    def handle_discard_cells(self):
        if not self.selected_cells: # Guard clause, though menu shouldn't appear if empty
            return

        self.discarded_cells.update(self.selected_cells)
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
        self.selected_cells.clear()
        self.request_paint_update()


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
        qp.setBrush(Qt.BrushStyle.NoBrush) 
        for row in range(self.filas):
            for col in range(self.columnas):
                index = row * self.columnas + col
                estado_area = self.area[index] if index < len(self.area) else 0
                rect_to_draw = QRectF(col * cell_w, row * cell_h, cell_w, cell_h)
                current_brush = Qt.BrushStyle.NoBrush # Default to no brush

                cell_tuple = (row, col) # Create tuple for checking in sets

                if cell_tuple in self.discarded_cells:
                    current_brush = QBrush(QColor(200, 0, 0, 150)) # Darker, more opaque red for discarded
                elif cell_tuple in self.selected_cells:
                    current_brush = QBrush(QColor(255, 0, 0, 100)) # Transparent red for selected
                elif index in self.temporal: 
                    current_brush = QBrush(QColor(0, 255, 0, 100)) # Existing temporal color
                elif estado_area == 1: 
                    # This was QColor(255, 0, 0, 100), same as selected_cells.
                    # Let's keep it for now, but it might need differentiation if estado_area == 1
                    # should look different from selected_cells.
                    # For now, selected_cells will take precedence if a cell is both area=1 and selected.
                    current_brush = QBrush(QColor(255, 165, 0, 100)) # Orange for area=1 to differentiate for now
                
                qp.setBrush(current_brush)
                qp.setPen(QColor(100, 100, 100, 100)) 
                qp.drawRect(rect_to_draw)

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
                if len(box_data) == 8:
                    x1, y1, x2, y2, tracker_id, cls, conf, es_prediccion = box_data
                elif len(box_data) == 7: 
                    x1, y1, x2, y2, tracker_id, cls, conf = box_data
                    es_prediccion = False 
                else: continue 

                scaled_x1 = (x1 * scale_x) + offset_x
                scaled_y1 = (y1 * scale_y) + offset_y
                scaled_x2 = (x2 * scale_x) + offset_x
                scaled_y2 = (y2 * scale_y) + offset_y
                scaled_w = scaled_x2 - scaled_x1
                scaled_h = scaled_y2 - scaled_y1
                pen = QPen()
                pen.setWidth(2)
                if es_prediccion: pen.setColor(QColor("green")) 
                else: pen.setColor(QColor("blue"))  
                qp.setPen(pen)
                qp.setBrush(Qt.BrushStyle.NoBrush) 
                qp.drawRect(QRectF(scaled_x1, scaled_y1, scaled_w, scaled_h)) 
                label_text = f"ID:{tracker_id} C:{conf:.2f}"
                if es_prediccion: label_text += " (P)"
                qp.setPen(QColor("white")) 
                text_x = scaled_x1
                text_y = scaled_y1 - 5
                if text_y < 10 : text_y = scaled_y1 + 15 
                qp.drawText(QPointF(text_x, text_y), label_text) 
        else: pass
