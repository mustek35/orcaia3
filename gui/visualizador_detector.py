from PyQt6.QtMultimedia import QMediaPlayer, QVideoSink
from PyQt6.QtCore import QObject, pyqtSignal, QUrl
from PyQt6.QtGui import QImage
import numpy as np
from core.detector_worker import DetectorWorker

class VisualizadorDetector(QObject):
    result_ready = pyqtSignal(list) 
    log_signal = pyqtSignal(str)

    def __init__(self, cam_data, parent=None):
        super().__init__(parent)
        self.cam_data = cam_data
        cam_ip_for_name = self.cam_data.get('ip', str(id(self))) 
        self.setObjectName(f"Visualizador_{cam_ip_for_name}")

        self.video_player = QMediaPlayer()
        self.video_sink = QVideoSink()
        self.video_player.setVideoSink(self.video_sink)

        self.video_sink.videoFrameChanged.connect(self.on_frame)
        self.video_player.errorOccurred.connect(
            lambda e: print(f"❌ MediaPlayer error ({self.objectName()}): {self.video_player.errorString()}")
        )

        # El intervalo de procesamiento de frames ahora se maneja aquí
        self.detector_frame_interval = cam_data.get("intervalo", 50) # Usar el valor de prueba o el de cam_data
        if self.detector_frame_interval < 1: # Asegurar un valor mínimo
            self.detector_frame_interval = 1
        self.frame_counter = 0
        
        imgsz_default = cam_data.get("imgsz", 416) # Mantener el default de prueba anterior
        
        self.detector = DetectorWorker(
            model_key=cam_data.get("modelo", "Personas"),
            confidence=cam_data.get("confianza", 0.5),
            frame_interval=1, # DetectorWorker ahora procesa cada frame que se le da
            imgsz=imgsz_default 
        )

        self.detector.result_ready.connect(self._procesar_resultados_detector_worker)
        self.detector.start()

    def _procesar_resultados_detector_worker(self, output_for_signal, model_key):
        self.result_ready.emit(output_for_signal)

    def iniciar(self):
        rtsp_url = self.cam_data.get("rtsp")
        if rtsp_url:
            print(f"🚀 [{self.objectName()}] Reproduciendo RTSP: {rtsp_url}")
            self.log_signal.emit(f"🎥 [{self.objectName()}] Streaming iniciado: {rtsp_url}")
            self.video_player.setSource(QUrl(rtsp_url))
            self.video_player.play()
        else:
            print(f"WARN [{self.objectName()}]: No se encontró URL RTSP para iniciar.")
            self.log_signal.emit(f"⚠️ [{self.objectName()}] No se encontró URL RTSP.")

    def detener(self):
        print(f"INFO [{self.objectName()}]: Deteniendo VisualizadorDetector...")
        if hasattr(self, 'detector') and self.detector:
            print(f"INFO [{self.objectName()}]: Deteniendo {self.detector.objectName()}...")
            self.detector.stop() 
        if hasattr(self, 'video_player') and self.video_player:
            player_state = self.video_player.playbackState()
            if player_state != QMediaPlayer.PlaybackState.StoppedState:
                print(f"INFO [{self.objectName()}]: Deteniendo QMediaPlayer (estado actual: {player_state})...")
                self.video_player.stop()
            print(f"INFO [{self.objectName()}]: Desvinculando salida de video del QMediaPlayer...")
            self.video_player.setVideoSink(None) 
            print(f"INFO [{self.objectName()}]: Agendando QMediaPlayer para deleteLater().")
            self.video_player.deleteLater()
            self.video_player = None 
        if hasattr(self, 'video_sink') and self.video_sink:
            self.video_sink = None 
        print(f"INFO [{self.objectName()}]: VisualizadorDetector detenido.")

    def on_frame(self, frame): # frame es QVideoFrame
        if not frame.isValid():
            # print(f"WARN [{self.objectName()}]: Frame inválido recibido en on_frame.")
            return

        self.frame_counter += 1
        
        if self.frame_counter % self.detector_frame_interval == 0:
            try:
                image = frame.toImage() 
                if image.isNull():
                    # print(f"WARN [{self.objectName()}]: frame.toImage() resultó en QImage nula.")
                    return

                img_converted = image.convertToFormat(QImage.Format.Format_RGB888)
                if img_converted.isNull():
                    # print(f"WARN [{self.objectName()}]: Conversión a RGB888 resultó en QImage nula.")
                    return
                    
                buffer_size = img_converted.sizeInBytes()
                buffer = img_converted.constBits() 
                if buffer is None:
                    # print(f"WARN [{self.objectName()}]: img_converted.constBits() devolvió None.")
                    return
                
                # Crear una copia del buffer para el array numpy
                arr = np.frombuffer(buffer, dtype=np.uint8).reshape(
                    (img_converted.height(), img_converted.width(), 3)
                ).copy() # .copy() es crucial

                if self.detector and self.detector.isRunning():
                    self.detector.set_frame(arr) # arr ya es una copia
                # else:
                    # print(f"WARN [{self.objectName()}]: Detector no disponible o no corriendo, no se pasa el frame.")
            except Exception as e:
                print(f"ERROR [{self.objectName()}] procesando frame en on_frame: {e}")
        # else:
            # No es necesario hacer nada si no es un frame a procesar por el detector
            pass
