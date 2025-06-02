from PyQt6.QtCore import QObject, pyqtSignal
import threading
import cv2
from urllib.parse import quote

class CamaraSecundariaWorker(QObject):
    frame_ready = pyqtSignal(object)  # para emitir frames si se quiere mostrar
    log_signal = pyqtSignal(str)

    def __init__(self, cam_data):
        super().__init__()
        self.cam_data = cam_data
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=self.run, daemon=True).start()

    def stop(self):
        self._running = False

    def run(self):
        ip = self.cam_data['ip']
        usuario = self.cam_data['usuario']
        contrasena = quote(self.cam_data['contrasena'])
        puerto = 554
        tipo = self.cam_data.get("tipo", "fija")
        canal = self.cam_data.get("canal", "2")
        perfil = self.cam_data.get("resolucion", "main").lower()

        if tipo == "nvr":
            perfil_id = {
                "main": "s0",
                "sub": "s1",
                "low": "s2",
                "more low": "s3"
            }.get(perfil, "s1")
            rtsp_url = f"rtsp://{usuario}:{contrasena}@{ip}:{puerto}/unicast/c{canal}/{perfil_id}/live"
        else:
            video_n = {
                "main": "video1",
                "sub": "video2",
                "low": "video3",
                "more low": "video4"
            }.get(perfil, "video1")
            rtsp_url = f"rtsp://{usuario}:{contrasena}@{ip}:{puerto}/media/{video_n}"

        self.log_signal.emit(f"🎬 Cámara secundaria conectando a: {rtsp_url}")

        cap = cv2.VideoCapture(rtsp_url)
        if not cap.isOpened():
            self.log_signal.emit("❌ No se pudo abrir la cámara secundaria")
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.log_signal.emit("⚠️ Fallo en la lectura del stream secundario")
                break

            self.frame_ready.emit(frame)

        cap.release()
        self.log_signal.emit("🛑 Cámara secundaria detenida")
