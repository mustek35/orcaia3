from PyQt6.QtWidgets import QWidget, QSizePolicy
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QFont, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QSizeF, QSize, QPointF, QTimer # QTimer añadido
from gui.visualizador_detector import VisualizadorDetector
from core.gestor_alertas import GestorAlertas
from core.rtsp_builder import generar_rtsp
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

        self.cam_data = None
        self.alertas = None
        self.objetos_previos = {}
        self.umbral_movimiento = 20
        self.detector = None 

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

    def perform_paint_update(self):
        self.paint_scheduled = False
        self.update() # Llama al self.update() original

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
            self.alertas.procesar_detecciones(
                nuevas_detecciones_para_alertas, 
                self.last_frame, 
                self.registrar_log, 
                self.cam_data
            )
            self.temporal = self.alertas.temporal
        
        self.request_paint_update() # Modificado

    def actualizar_pixmap_y_frame(self, frame): 
        image = frame.toImage() 
        if not image.isNull():
            img_converted = image.convertToFormat(QImage.Format.Format_RGB888)
            current_frame_width = img_converted.width()
            current_frame_height = img_converted.height()
            if self.original_frame_size is None or \
               self.original_frame_size.width() != current_frame_width or \
               self.original_frame_size.height() != current_frame_height:
                self.original_frame_size = QSize(current_frame_width, current_frame_height) 
            ptr = img_converted.bits()
            ptr.setsize(img_converted.sizeInBytes()) 
            arr = np.array(ptr).reshape((img_converted.height(), img_converted.width(), 3))
            self.last_frame = arr.copy() 
            self.pixmap = QPixmap.fromImage(img_converted) 
            self.request_paint_update() # Modificado

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
            self.visualizador = None 
        if self.detector: 
             self.detector = None
        self.paint_update_timer.stop() # Detener el timer si el widget se detiene


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
                current_brush = Qt.BrushStyle.NoBrush
                if index in self.temporal: 
                    current_brush = QBrush(QColor(0, 255, 0, 100)) 
                elif estado_area == 1: 
                    current_brush = QBrush(QColor(255, 0, 0, 100)) 
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
