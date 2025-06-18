import os
import uuid
import cv2
from datetime import datetime, timedelta
from gui.image_saver import ImageSaverThread
import math 
import time # For timestamps

class GestorAlertas:
    def __init__(self, cam_id, filas, columnas):
        self.cam_id = cam_id
        self.filas = filas
        self.columnas = columnas
        
        self.capturas_realizadas = 0
        self.max_capturas_por_minuto = 3 
        self.ultimo_reset_capturas = datetime.now()

        # Attributes for detection confirmation logic
        self.frames_necesarios_confirmacion = 5  # Renamed from frames_para_confirmar
        self.candidatos_deteccion = {} 
        # New structure for candidates:
        # {tracker_id: {'cls_inicial': clase, 'conf_inicial': conf, 
        #               'frames_confirmados_consecutivos': contador, 
        #               'bbox_para_guardar': bbox, 'conf_para_guardar': conf,
        #               'ultima_vez_visto_ts': timestamp}}
        self.max_edad_candidato_sin_ver_seg = 5.0 

        # Attributes for per-tracker cooldown
        self.objetos_guardados_recientemente = {} 
        self.cooldown_captura_por_tracker_seg = 10 
        self.umbral_movimiento_confirmado_px = 15 
        self.ultimas_posiciones_confirmadas = {} 

        # Diagnostic print for __init__
        print(f"INFO GestorAlertas [{self.cam_id}] Creado. Frames_confirmar: {self.frames_necesarios_confirmacion}, Cooldown_tracker: {self.cooldown_captura_por_tracker_seg}s")


    def _get_object_type_name(self, cls, cam_modelo):
        if cam_modelo == "Embarcaciones":
            return "embarcaciones" 
        
        class_map = {
            0: "personas",
            2: "autos",
            9: "barcos"
        }
        return class_map.get(cls, f"clase_{cls}")


    def procesar_detecciones(self, boxes, last_frame, log_callback, cam_data):
        timestamp_actual = time.time()
        
        # Reset capture count per minute
        if datetime.now() - self.ultimo_reset_capturas > timedelta(minutes=1):
            if self.capturas_realizadas > 0:
                print(f"INFO GestorAlertas [{self.cam_id}]: Reseteando contador de capturas. Realizadas en el último minuto: {self.capturas_realizadas}")
            self.capturas_realizadas = 0
            self.ultimo_reset_capturas = datetime.now()

        current_frame_tracker_ids = {det[5] for det in boxes if len(det) == 7} 

        for box_data in boxes:
            if len(box_data) != 7:
                print(f"WARN GestorAlertas [{self.cam_id}]: Detección ignorada, formato incorrecto (esperados 7 elementos): {box_data}")
                continue
            
            x1, y1, x2, y2, cls_actual, tracker_id, conf_actual = box_data # Renamed for clarity
            current_bbox = (x1, y1, x2, y2)

            if tracker_id in self.candidatos_deteccion:
                # Existing candidate
                candidate = self.candidatos_deteccion[tracker_id]
                
                # Step 1: Verify continuation conditions
                condicion_clase = (cls_actual == candidate['cls_inicial'])
                condicion_confianza = (conf_actual >= candidate['conf_inicial'])

                # Step 2: If both conditions are True
                if condicion_clase and condicion_confianza:
                    candidate['frames_confirmados_consecutivos'] += 1
                    candidate['bbox_para_guardar'] = current_bbox
                    candidate['conf_para_guardar'] = conf_actual # Update with current, potentially higher, confidence
                    candidate['ultima_vez_visto_ts'] = timestamp_actual
                    # print(f"DEBUG GestorAlertas [{self.cam_id}]: Candidato {tracker_id} (clase {cls_actual}) continúa. Frames: {candidate['frames_confirmados_consecutivos']}")
                else:
                    # Step 3: If any condition is False, reset the candidate
                    # print(f"WARN GestorAlertas [{self.cam_id}]: Candidato {tracker_id} reseteado. Clase cambió: {not condicion_clase} (antes {candidate['cls_inicial']}, ahora {cls_actual}), Confianza bajó: {not condicion_confianza} (antes {candidate['conf_inicial']:.2f}, ahora {conf_actual:.2f})")
                    self.candidatos_deteccion[tracker_id] = {
                        'cls_inicial': cls_actual,
                        'conf_inicial': conf_actual,
                        'frames_confirmados_consecutivos': 1, # Reset to 1 for the new sequence
                        'bbox_para_guardar': current_bbox,
                        'conf_para_guardar': conf_actual,
                        'ultima_vez_visto_ts': timestamp_actual
                    }
            else:
                # New candidate - Initialize with the new structure
                # print(f"DEBUG GestorAlertas [{self.cam_id}]: Nuevo candidato {tracker_id} (clase {cls_actual}, conf {conf_actual:.2f}). Frames: 1")
                self.candidatos_deteccion[tracker_id] = {
                    'cls_inicial': cls_actual,
                    'conf_inicial': conf_actual,
                    'frames_confirmados_consecutivos': 1, # Initialized to 1
                    'bbox_para_guardar': current_bbox,
                    'conf_para_guardar': conf_actual,
                    'ultima_vez_visto_ts': timestamp_actual
                }
        
        confirmed_detections_to_save = [] 

        for tracker_id, data_candidato in list(self.candidatos_deteccion.items()): 
            if data_candidato['frames_confirmados_consecutivos'] >= self.frames_necesarios_confirmacion:
                # print(f"INFO GestorAlertas [{self.cam_id}]: Tracker {tracker_id} CONFIRMADO (clase {data_candidato['cls_inicial']}). Frames: {data_candidato['frames_confirmados_consecutivos']}.")
                confirmed_detections_to_save.append({
                    'tracker_id': tracker_id,
                    'cls': data_candidato['cls_inicial'], 
                    'bbox': data_candidato['bbox_para_guardar'], 
                    'conf': data_candidato['conf_para_guardar']  
                })
                self.candidatos_deteccion[tracker_id]['frames_confirmados_consecutivos'] = 0 


        if confirmed_detections_to_save:
            self._guardar_confirmados(confirmed_detections_to_save, last_frame, log_callback, cam_data, timestamp_actual)

        stale_candidates_removed = 0
        for tracker_id, data_candidato in list(self.candidatos_deteccion.items()):
            is_stale_by_absence = tracker_id not in current_frame_tracker_ids
            is_stale_by_time = (timestamp_actual - data_candidato['ultima_vez_visto_ts']) > self.max_edad_candidato_sin_ver_seg
            
            if is_stale_by_absence and is_stale_by_time: 
                # print(f"DEBUG GestorAlertas [{self.cam_id}]: Eliminando candidato stale {tracker_id}. Ausente: {is_stale_by_absence}, Viejo: {is_stale_by_time}")
                del self.candidatos_deteccion[tracker_id]
                stale_candidates_removed += 1
        # if stale_candidates_removed > 0:
            # print(f"INFO GestorAlertas [{self.cam_id}]: Eliminados {stale_candidates_removed} candidatos stale.")
        
        self.temporal.clear()
        if last_frame is not None:
            h, w, _ = last_frame.shape
            for tracker_id, data_candidato in self.candidatos_deteccion.items():
                if data_candidato['frames_confirmados_consecutivos'] > 0 : 
                    x1_g, y1_g, x2_g, y2_g = data_candidato['bbox_para_guardar']
                    cx_grid = int((x1_g + x2_g) / 2)
                    cy_grid = int((y1_g + y2_g) / 2)
                    fila = int(cy_grid / h * self.filas)
                    columna = int(cx_grid / w * self.columnas)
                    index = fila * self.columnas + columna
                    self.temporal.add(index)


    def _guardar_confirmados(self, lista_confirmados, frame, log_callback, cam_data, timestamp_actual):
        # print(f"DEBUG GestorAlertas [{self.cam_id}]: _guardar_confirmados llamado con {len(lista_confirmados)} objetos.")
        for det_confirmada in lista_confirmados:
            tracker_id = det_confirmada['tracker_id']
            cls = det_confirmada['cls']
            bbox = det_confirmada['bbox']
            conf = det_confirmada['conf'] 
            x1, y1, x2, y2 = bbox
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            if self.capturas_realizadas >= self.max_capturas_por_minuto:
                # print(f"DEBUG GestorAlertas [{self.cam_id}]: No se guarda imagen para tracker {tracker_id}. Razón: Límite de capturas.")
                continue 

            last_capture_ts = self.objetos_guardados_recientemente.get(tracker_id)
            if last_capture_ts and (timestamp_actual - last_capture_ts < self.cooldown_captura_por_tracker_seg):
                # print(f"DEBUG GestorAlertas [{self.cam_id}]: No se guarda imagen para tracker {tracker_id}. Razón: Cooldown.")
                continue

            ha_movido_significativamente = True 
            if last_capture_ts: 
                prev_cx, prev_cy = self.ultimas_posiciones_confirmadas.get(tracker_id, (None, None))
                if prev_cx is not None:
                    distancia = math.hypot(cx - prev_cx, cy - prev_cy)
                    if distancia < self.umbral_movimiento_confirmado_px:
                        ha_movido_significativamente = False
                        # print(f"DEBUG GestorAlertas [{self.cam_id}]: No se guarda imagen para tracker {tracker_id}. Razón: No movimiento.")
            
            if not ha_movido_significativamente:
                continue

            object_type_name = self._get_object_type_name(cls, cam_data.get("modelo"))
            # print(f"INFO GestorAlertas [{self.cam_id}]: GUARDANDO IMAGEN para tracker {tracker_id}...")
            modelo_usado = cam_data.get("modelo", "desconocido")
            
            hilo = ImageSaverThread(
                frame=frame,
                bbox=(x1, y1, x2, y2),
                cls=cls,
                coordenadas=(cx, cy),
                modelo=modelo_usado,
                confianza=conf 
            )
            hilo.finished.connect(lambda h=hilo: self._eliminar_hilo(h))
            self.hilos_guardado.append(hilo)
            hilo.start()

            self.capturas_realizadas += 1
            self.objetos_guardados_recientemente[tracker_id] = timestamp_actual 
            self.ultimas_posiciones_confirmadas[tracker_id] = (cx, cy) 

            log_callback(f"🟢 ALERTA CONFIRMADA (ID:{tracker_id}) - {object_type_name} (clase {cls}, conf {conf:.2f}) en ({cx}, {cy})")
            log_callback(f"🖼️ Captura {object_type_name} (ID:{tracker_id}) guardada.")


    def _eliminar_hilo(self, hilo):
        if hilo in self.hilos_guardado:
            self.hilos_guardado.remove(hilo)
