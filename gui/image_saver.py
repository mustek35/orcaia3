from PyQt6.QtCore import QThread
import os
import uuid
import cv2
import json
from datetime import datetime

class ImageSaverThread(QThread):
    def __init__(self, frame, bbox, cls, coordenadas, modelo, confianza, parent=None):
        super().__init__(parent)
        self.frame = frame
        self.bbox = bbox
        self.cls = cls
        self.coordenadas = coordenadas
        self.modelo = modelo  # This holds the model name, e.g., "Embarcaciones"
        self.confianza = confianza

    def run(self):
        if self.frame is None or self.bbox is None:
            return

        h, w, _ = self.frame.shape
        x1, y1, x2, y2 = map(int, self.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = self.frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        # Determine the base folder for saving captures
        if self.cls == 0 and self.modelo == "Embarcaciones":
            carpeta_base = "embarcaciones"
        elif self.cls == 0:  # Handles Personas (cls 0 from non-"Embarcaciones" models)
            carpeta_base = "personas"
        elif self.cls == 2:
            carpeta_base = "autos"
        elif self.cls == 9:
            carpeta_base = "barcos"
        else:
            carpeta_base = "otros"

        now = datetime.now()
        fecha = now.strftime("%Y-%m-%d")
        hora = now.strftime("%H-%M-%S")

        ruta = os.path.join("capturas", carpeta_base, fecha)
        os.makedirs(ruta, exist_ok=True)  # Ensures directory exists

        nombre = f"{fecha}_{hora}_{uuid.uuid4().hex[:6]}"
        path_final = os.path.join(ruta, f"{nombre}.jpg")
        cv2.imwrite(path_final, crop)

        # Guardar metadatos
        metadata = {
            "fecha": fecha,
            "hora": hora.replace("-", ":"),  # For HH:MM:SS format
            "modelo": self.modelo,
            "coordenadas": self.coordenadas,
            "confianza": self.confianza
        }
        with open(os.path.join(ruta, f"{nombre}.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=4)
