from PyQt6.QtMultimedia import QMediaPlayer, QVideoSink
from PyQt6.QtCore import QObject, pyqtSignal, QUrl
from PyQt6.QtGui import QImage
import numpy as np
from core.detector_worker import DetectorWorker

class VisualizadorDetector(QObject):
    result_ready = pyqtSignal(list) # Esta señal emite solo la lista de boxes
    log_signal = pyqtSignal(str)

    def __init__(self, cam_data, parent=None):
        super().__init__(parent)
        self.cam_data = cam_data

        self.video_player = QMediaPlayer()
        self.video_sink = QVideoSink()
        self.video_player.setVideoSink(self.video_sink)

        self.video_sink.videoFrameChanged.connect(self.on_frame)
        self.video_player.errorOccurred.connect(
            lambda e: print(f"❌ MediaPlayer error: {self.video_player.errorString()}")
        )

        frame_interval = cam_data.get("intervalo", 80)
        self.detector = DetectorWorker(
            model_key=cam_data.get("modelo", "Personas"),
            confidence=cam_data.get("confianza", 0.5),
            frame_interval=frame_interval, # Usar la variable definida
            imgsz=cam_data.get("imgsz", 640)
        )

        # Conectar la señal del DetectorWorker al nuevo slot intermediario
        self.detector.result_ready.connect(self._procesar_resultados_detector_worker)
        self.detector.start()

    def _procesar_resultados_detector_worker(self, output_for_signal, model_key):
        # Este slot recibe (list, str) del DetectorWorker
        # pero solo emite la lista (output_for_signal) a través de la señal propia de VisualizadorDetector
        self.result_ready.emit(output_for_signal)
        # model_key no se usa aquí directamente, pero se recibe. Podría usarse para logging si se desea.

    def iniciar(self):
        rtsp_url = self.cam_data.get("rtsp")
        if rtsp_url:
            print(f"🚀 Reproduciendo RTSP: {rtsp_url}")
            self.log_signal.emit(f"🎥 Streaming iniciado: {rtsp_url}")
            self.video_player.setSource(QUrl(rtsp_url))
            self.video_player.play()

    def detener(self):
        self.video_player.stop()
        self.detector.stop()

    def on_frame(self, frame): # Recibe QVideoFrame
        if frame.isValid():
            image = frame.toImage() # Convertir a QImage
            # Convertir QImage a formato RGB888 si es necesario, luego a array numpy
            image = image.convertToFormat(QImage.Format.Format_RGB888)
            ptr = image.bits()
            ptr.setsize(image.sizeInBytes()) # Importante para versiones recientes de PySide/PyQt
            arr = np.array(ptr).reshape((image.height(), image.width(), 3))
            self.detector.set_frame(arr.copy()) # Pasar copia del array numpy al detector
        # else:
            # print("Frame inválido recibido en VisualizadorDetector.on_frame") # Opcional: para debugging
            # Podría ser útil manejar este caso, quizás pasando None al detector o un frame vacío.
            # self.detector.set_frame(None) # O manejar como se prefiera.
            pass
