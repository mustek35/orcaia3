import os
import uuid
import cv2
from datetime import datetime, timedelta
from gui.image_saver import ImageSaverThread

class GestorAlertas:
    def __init__(self, cam_id, filas, columnas):
        self.cam_id = cam_id
        self.filas = filas
        self.columnas = columnas
        self.box_streak = 0
        self.deteccion_bote_streak = 0
        self.capturas_realizadas = 0
        self.max_capturas = 3
        self.ultimo_reset = datetime.now()
        self.temporal = set()
        self.hilos_guardado = []
        self.ultimas_posiciones = {}

    def procesar_detecciones(self, boxes, last_frame, log_callback, cam_data):
        if datetime.now() - self.ultimo_reset > timedelta(minutes=1):
            self.capturas_realizadas = 0
            self.ultimo_reset = datetime.now()

        hay_persona = hay_bote = hay_auto = hay_embarcacion = False
        boxes_personas, boxes_botes, boxes_autos, boxes_embarcaciones = [], [], [], []

        for box in boxes:
            if len(box) != 5:
                continue

            x1, y1, x2, y2, cls = box
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if cam_data.get("modelo") == "Embarcaciones":
                boxes_embarcaciones.append((x1, y1, x2, y2, cls, cx, cy))
                hay_embarcacion = True
                continue

            if cls == 0:
                boxes_personas.append((x1, y1, x2, y2, cls, cx, cy))
                hay_persona = True
            elif cls == 2:
                boxes_autos.append((x1, y1, x2, y2, cls, cx, cy))
                hay_auto = True
            elif cls == 9:
                boxes_botes.append((x1, y1, x2, y2, cls, cx, cy))
                hay_bote = True

        if hay_persona:
            self.box_streak += 1
        else:
            self.box_streak = 0

        if self.box_streak >= 3:
            self._guardar(boxes_personas, last_frame, log_callback, tipo='personas', cam_data=cam_data)

        if hay_bote: # Handles class 9 detections from non-"Embarcaciones" models
            self._guardar(boxes_botes, last_frame, log_callback, tipo='barcos', cam_data=cam_data)

        if hay_auto:
            self._guardar(boxes_autos, last_frame, log_callback, tipo='autos', cam_data=cam_data)

        if hay_embarcacion: # Handles detections from "Embarcaciones" model (which are class 0)
            self._guardar(boxes_embarcaciones, last_frame, log_callback, tipo='embarcaciones', cam_data=cam_data)

        self.temporal.clear()
        if last_frame is not None:
            h, w, _ = last_frame.shape
            for box in boxes:
                if len(box) != 5:
                    continue
                x1, y1, x2, y2, cls = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                fila = int(cy / h * self.filas)
                columna = int(cx / w * self.columnas)
                index = fila * self.columnas + columna
                self.temporal.add(index)

    def _ha_habido_movimiento(self, clase, cx, cy, umbral=25):
        cx_prev, cy_prev = self.ultimas_posiciones.get(clase, (None, None))
        if cx_prev is None:
            self.ultimas_posiciones[clase] = (cx, cy)
            return True

        distancia = ((cx - cx_prev)**2 + (cy - cy_prev)**2)**0.5
        if distancia > umbral:
            self.ultimas_posiciones[clase] = (cx, cy)
            return True
        return False

    def _guardar(self, boxes, frame, log_callback, tipo, cam_data):
        for (x1, y1, x2, y2, cls, cx, cy) in boxes:
            if self.capturas_realizadas >= self.max_capturas:
                break
            if frame is not None:
                if not self._ha_habido_movimiento(cls, cx, cy):
                    continue

                modelo = cam_data.get("modelo", "desconocido") # Correctly passed to ImageSaverThread
                confianza = cam_data.get("confianza", 0.5)

                hilo = ImageSaverThread(
                    frame=frame,
                    bbox=(x1, y1, x2, y2),
                    cls=cls,
                    coordenadas=(cx, cy),
                    modelo=modelo,
                    confianza=confianza
                )
                hilo.finished.connect(lambda h=hilo: self._eliminar_hilo(h))
                self.hilos_guardado.append(hilo)
                hilo.start()

                self.capturas_realizadas += 1
                # The 'tipo' variable here will now correctly be 'embarcaciones' or 'barcos' etc.
                # affecting the log message.
                log_callback(f"🟢 Movimiento detectado - {tipo[:-1].capitalize()} (clase {cls}) en ({cx}, {cy})")
                log_callback(f"🖼️ Captura {tipo} guardada en segundo plano")

    def _eliminar_hilo(self, hilo):
        if hilo in self.hilos_guardado:
            self.hilos_guardado.remove(hilo)
