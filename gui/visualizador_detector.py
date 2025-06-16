from PyQt6.QtMultimedia import QMediaPlayer, QVideoSink, QVideoFrameFormat, QVideoFrame
from PyQt6.QtCore import QObject, pyqtSignal, QUrl
from PyQt6.QtGui import QImage
import numpy as np

from core.detector_worker import DetectorWorker, iou
from core.advanced_tracker import AdvancedTracker

from logging_utils import get_logger

logger = get_logger(__name__)

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
            lambda e: logger.error(
                "MediaPlayer error (%s): %s", self.objectName(), self.video_player.errorString()
            )
        )

        # El intervalo de procesamiento de frames ahora se maneja aquí
        self.detector_frame_interval = cam_data.get("intervalo", 50)
        if self.detector_frame_interval < 1:
            self.detector_frame_interval = 1
        self.frame_counter = 0
        
        imgsz_default = cam_data.get("imgsz", 416)
        device = cam_data.get("device", "cpu")
        logger.debug("%s: Inicializando DetectorWorker en %s", self.objectName(), device)

        # Get models configuration
        modelos = cam_data.get("modelos")
        if not modelos:
            modelo_single = cam_data.get("modelo", "Personas")
            modelos = [modelo_single] if modelo_single else []

        # Determine primary model for tracker configuration
        primary_model = modelos[0] if modelos else "Personas"
        
        # Build tracker configuration
        tracker_config = self._build_tracker_config(primary_model, cam_data)
        
        # Initialize tracker with error handling
        self.tracker = None
        self._initialize_tracker(tracker_config, device)
        
        self._pending_detections = {}
        self._last_frame = None
        self._current_frame_id = 0

        # Create detectors
        self.detectors = []
        for m in modelos:
            try:
                detector = DetectorWorker(
                    model_key=m,
                    confidence=cam_data.get("confianza", 0.5),
                    frame_interval=1,
                    imgsz=imgsz_default,
                    device=device,
                    track=False,  # Disable individual tracking, use shared tracker
                )
                detector.result_ready.connect(
                    lambda res, _mk, fid, mk=m: self._procesar_resultados_detector_worker(res, mk, fid)
                )
                detector.start()
                self.detectors.append(detector)
                logger.debug("%s: DetectorWorker %s iniciado correctamente", self.objectName(), m)
            except Exception as e:
                logger.error("%s: Error inicializando DetectorWorker %s: %s", self.objectName(), m, e)
                self.log_signal.emit(f"❌ Error inicializando detector {m}: {e}")
            
        logger.debug("%s: %d DetectorWorker(s) started", self.objectName(), len(self.detectors))
        
        # Log tracker configuration
        if self.tracker:
            self.log_signal.emit(
                f"🚀 Tracker iniciado para {primary_model} - "
                f"Control de tamaño: {'SÍ' if tracker_config.get('enable_size_control', True) else 'NO'}, "
                f"Max cambio: {tracker_config.get('max_size_change_ratio', 1.3)*100-100:.0f}%, "
                f"Área: {tracker_config.get('min_box_area', 100)}-{tracker_config.get('max_box_area', 50000)}px²"
            )

    def _initialize_tracker(self, tracker_config, device):
        """Initialize tracker with comprehensive error handling."""
        try:
            logger.info("%s: Inicializando AdvancedTracker...", self.objectName())
            
            self.tracker = AdvancedTracker(
                max_age=tracker_config.get("max_age", 30),
                n_init=tracker_config.get("n_init", 3),
                conf_threshold=tracker_config.get("conf_threshold", self.cam_data.get("confianza", 0.5)),
                device=device,
                lost_ttl=tracker_config.get("lost_ttl", self.cam_data.get("lost_ttl", 5)),
                enable_size_control=tracker_config.get("enable_size_control", True),
                enable_velocity_prediction=tracker_config.get("enable_velocity_prediction", True)
            )
            
            # Apply additional tracker parameters
            self._configure_tracker_parameters(tracker_config)
            
            logger.info("%s: AdvancedTracker inicializado correctamente", self.objectName())
            self.log_signal.emit(f"✅ Tracker inicializado correctamente para {self.objectName()}")
            
        except ImportError as e:
            logger.error("%s: Error de importación al inicializar tracker: %s", self.objectName(), e)
            self.log_signal.emit(f"❌ Error de importación del tracker: {e}")
            self.tracker = None
        except Exception as e:
            logger.error("%s: Error general al inicializar tracker: %s", self.objectName(), e, exc_info=True)
            self.log_signal.emit(f"❌ Error inicializando tracker: {e}")
            self.tracker = None

    def _build_tracker_config(self, primary_model, cam_data):
        """Build tracker configuration based on model type and camera data."""
        # Default configuration for maritime tracking
        default_config = {
            "max_age": 30,
            "n_init": 2,  # Faster confirmation for small objects
            "conf_threshold": 0.25,
            "lost_ttl": 8,
            "enable_size_control": True,  # Always enabled
            "enable_velocity_prediction": True,
            "max_size_change_ratio": 1.3,
            "size_history_length": 15,
            "size_outlier_threshold": 2.0,
            "min_box_area": 100,
            "max_box_area": 50000,
            "max_prediction_distance": 50,
            "velocity_smoothing_factor": 0.85,
            "movement_threshold": 3.0,
            "prediction_decay_factor": 0.90,
        }
        
        # Model-specific overrides
        model_configs = {
            "Embarcaciones": {
                "max_size_change_ratio": 1.2,  # Very strict for boats
                "size_outlier_threshold": 1.5,
                "min_box_area": 100,
                "max_box_area": 20000,  # Small boats
                "max_prediction_distance": 40,
                "movement_threshold": 2.0,
                "lost_ttl": 10,
            },
            "Barcos": {
                "max_size_change_ratio": 1.3,
                "size_outlier_threshold": 1.8,
                "min_box_area": 100,
                "max_box_area": 30000,
                "max_prediction_distance": 50,
                "movement_threshold": 3.0,
                "lost_ttl": 12,
            },
            "Personas": {
                "max_size_change_ratio": 1.5,
                "size_outlier_threshold": 2.5,
                "min_box_area": 50,
                "max_box_area": 10000,
                "max_prediction_distance": 100,
                "movement_threshold": 3.0,
                "lost_ttl": 10,
            },
            "Autos": {
                "max_size_change_ratio": 2.0,
                "size_outlier_threshold": 2.5,
                "min_box_area": 200,
                "max_box_area": 40000,
                "max_prediction_distance": 150,
                "movement_threshold": 8.0,
                "lost_ttl": 7,
            }
        }
        
        # Apply model-specific config
        if primary_model in model_configs:
            default_config.update(model_configs[primary_model])
        
        # Apply custom config from cam_data if provided
        custom_config = cam_data.get("tracker_config", {})
        default_config.update(custom_config)
        
        return default_config

    def _configure_tracker_parameters(self, tracker_config):
        """Apply additional configuration parameters to the tracker."""
        if not self.tracker:
            return
            
        param_mapping = {
            'MAX_SIZE_CHANGE_RATIO': 'max_size_change_ratio',
            'SIZE_HISTORY_LENGTH': 'size_history_length',
            'SIZE_OUTLIER_THRESHOLD': 'size_outlier_threshold',
            'MIN_BOX_AREA': 'min_box_area',
            'MAX_BOX_AREA': 'max_box_area',
            'MAX_PREDICTION_DISTANCE': 'max_prediction_distance',
            'VELOCITY_SMOOTHING_FACTOR': 'velocity_smoothing_factor',
            'MOVEMENT_THRESHOLD': 'movement_threshold',
            'PREDICTION_DECAY_FACTOR': 'prediction_decay_factor',
            'MIN_DETECTION_CONFIDENCE': 'min_detection_confidence',
        }
        
        for attr_name, config_key in param_mapping.items():
            if hasattr(self.tracker, attr_name) and config_key in tracker_config:
                try:
                    setattr(self.tracker, attr_name, tracker_config[config_key])
                    logger.debug(
                        "%s: Set tracker.%s = %s",
                        self.objectName(),
                        attr_name,
                        tracker_config[config_key]
                    )
                except Exception as e:
                    logger.warning(
                        "%s: No se pudo configurar tracker.%s: %s",
                        self.objectName(),
                        attr_name,
                        e
                    )

    def _procesar_resultados_detector_worker(self, output_for_signal, model_key, frame_id):
        """Process results from detector workers with comprehensive error handling."""
        logger.debug(
            "%s: _procesar_resultados_detector_worker received %d results for model %s",
            self.objectName(),
            len(output_for_signal),
            model_key,
        )
        
        if frame_id != self._current_frame_id:
            logger.debug(
                "%s: Ignoring results for old frame %s (current %s)",
                self.objectName(),
                frame_id,
                self._current_frame_id,
            )
            return

        self._pending_detections[model_key] = output_for_signal
        
        # Wait for all detectors to report
        if len(self._pending_detections) == len(self.detectors):
            try:
                # Merge detections from all models
                merged = []
                for model_key, dets in self._pending_detections.items():
                    for det in dets:
                        duplicate = False
                        for mdet in merged:
                            # Merge if boxes overlap significantly regardless of class
                            if iou(det['bbox'], mdet['bbox']) > 0.5:
                                if det.get('conf', 0) > mdet.get('conf', 0):
                                    mdet.update(det)
                                duplicate = True
                                break
                        if not duplicate:
                            merged.append(det.copy())

                # Verify tracker is available
                if self.tracker is None:
                    logger.warning("%s: Tracker es None, intentando reinicializar...", self.objectName())
                    primary_model = self.cam_data.get("modelos", [self.cam_data.get("modelo", "Personas")])[0]
                    tracker_config = self._build_tracker_config(primary_model, self.cam_data)
                    device = self.cam_data.get("device", "cpu")
                    self._initialize_tracker(tracker_config, device)
                    
                    if self.tracker is None:
                        logger.error("%s: No se pudo reinicializar el tracker", self.objectName())
                        self.log_signal.emit(f"❌ Error crítico: No se pudo inicializar el tracker para {self.objectName()}")
                        # Emit empty results to avoid blocking
                        self.result_ready.emit([])
                        self._pending_detections = {}
                        return

                # Update tracker with merged detections
                tracks = self.tracker.update(merged, frame=self._last_frame)
                
                # Filter out anomalies if too many
                size_anomalies = [t for t in tracks if t.get('size_anomaly', False)]
                if len(size_anomalies) > 0:
                    logger.warning(
                        "%s: %d size anomalies detected this frame",
                        self.objectName(),
                        len(size_anomalies)
                    )
                
                # Log detailed statistics every 100 frames
                if self.frame_counter % 100 == 0:
                    self._log_tracker_statistics(tracks)
                
                # Check for oversized boxes and log warnings
                for track in tracks:
                    bbox = track.get('bbox', [0, 0, 0, 0])
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if len(bbox) >= 4 else 0
                    if area > 10000:  # Large box threshold
                        logger.warning(
                            "%s: Large box detected - ID:%s, Area:%dpx², Predicted:%s",
                            self.objectName(),
                            track.get('id', 'unknown'),
                            int(area),
                            track.get('predicted', False)
                        )
                
                self.result_ready.emit(tracks)
                
            except Exception as e:
                logger.error("%s: Error en _procesar_resultados_detector_worker: %s", self.objectName(), e, exc_info=True)
                self.log_signal.emit(f"❌ Error procesando detecciones: {e}")
                # Emit empty results to avoid blocking
                self.result_ready.emit([])
            finally:
                self._pending_detections = {}

    def _log_tracker_statistics(self, tracks):
        """Log detailed tracker statistics."""
        try:
            total_tracks = len(tracks)
            active_tracks = len([t for t in tracks if t.get('conf', 0) > 0.5])
            predicted_tracks = len([t for t in tracks if t.get('predicted', False)])
            corrected_tracks = len([t for t in tracks if t.get('size_corrected', False)])
            low_conf_tracks = len([t for t in tracks if t.get('confidence_decay', 1) < 0.8])
            
            # Calculate average box sizes by class
            class_sizes = {}
            for t in tracks:
                cls = t.get('cls', -1)
                bbox = t.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if cls not in class_sizes:
                        class_sizes[cls] = []
                    class_sizes[cls].append(area)
            
            avg_sizes = {cls: np.mean(sizes) for cls, sizes in class_sizes.items() if sizes}
            
            logger.info(
                "%s: Tracker stats - Total:%d, Active:%d, Predicted:%d, Corrected:%d, LowConf:%d",
                self.objectName(), total_tracks, active_tracks, predicted_tracks, 
                corrected_tracks, low_conf_tracks
            )
            
            for cls, avg_size in avg_sizes.items():
                cls_name = {0: "Person", 1: "Boat", 2: "Car", 8: "Ship"}.get(cls, f"Class{cls}")
                logger.info(
                    "%s: Average box size for %s: %.0fpx²",
                    self.objectName(), cls_name, avg_size
                )
        except Exception as e:
            logger.error("%s: Error logging tracker statistics: %s", self.objectName(), e)

    def iniciar(self):
        """Start the video stream."""
        rtsp_url = self.cam_data.get("rtsp")
        if rtsp_url:
            logger.info("%s: Reproduciendo RTSP %s", self.objectName(), rtsp_url)
            self.log_signal.emit(f"🎥 [{self.objectName()}] Streaming iniciado: {rtsp_url}")
            self.video_player.setSource(QUrl(rtsp_url))
            self.video_player.play()
        else:
            logger.warning("%s: No se encontró URL RTSP para iniciar", self.objectName())
            self.log_signal.emit(f"⚠️ [{self.objectName()}] No se encontró URL RTSP.")

    def detener(self):
        """Stop the visualizer and clean up resources."""
        logger.info("%s: Deteniendo VisualizadorDetector", self.objectName())
        
        # Stop detectors
        if hasattr(self, 'detectors'):
            for det in self.detectors:
                if det:
                    try:
                        logger.info("%s: Deteniendo %s", self.objectName(), det.objectName())
                        det.stop()
                    except Exception as e:
                        logger.error("%s: Error deteniendo detector: %s", self.objectName(), e)
                    
        # Stop video player
        if hasattr(self, 'video_player') and self.video_player:
            try:
                player_state = self.video_player.playbackState()
                if player_state != QMediaPlayer.PlaybackState.StoppedState:
                    logger.info("%s: Deteniendo QMediaPlayer estado %s", self.objectName(), player_state)
                    self.video_player.stop()
                logger.info("%s: Desvinculando salida de video del QMediaPlayer", self.objectName())
                self.video_player.setVideoSink(None)
                logger.info("%s: Agendando QMediaPlayer para deleteLater", self.objectName())
                self.video_player.deleteLater()
                self.video_player = None
            except Exception as e:
                logger.error("%s: Error deteniendo video player: %s", self.objectName(), e)
            
        # Clear video sink
        if hasattr(self, 'video_sink') and self.video_sink:
            self.video_sink = None
            
        # Clear tracker
        if hasattr(self, 'tracker') and self.tracker:
            try:
                # Log final statistics
                if hasattr(self.tracker, 'track_history'):
                    logger.info(
                        "%s: Final tracker state - %d tracks in history",
                        self.objectName(),
                        len(self.tracker.track_history)
                    )
                self.tracker = None
            except Exception as e:
                logger.error("%s: Error limpiando tracker: %s", self.objectName(), e)
            
        logger.info("%s: VisualizadorDetector detenido completamente", self.objectName())

    def on_frame(self, frame):
        """Process incoming video frame."""
        logger.debug(
            "%s: on_frame called %d (interval %d)",
            self.objectName(),
            self.frame_counter,
            self.detector_frame_interval,
        )
        
        if not frame.isValid():
            return

        handle_type = frame.handleType()
        logger.debug("%s: frame handle type %s", self.objectName(), handle_type)

        self.frame_counter += 1
        
        # Process frame at specified interval
        if self.frame_counter % self.detector_frame_interval == 0:
            try:
                qimg = self._qimage_from_frame(frame)
                if qimg is None:
                    return
                    
                # Convert to RGB888 if needed
                if qimg.format() != QImage.Format.Format_RGB888:
                    img_converted = qimg.convertToFormat(QImage.Format.Format_RGB888)
                else:
                    img_converted = qimg

                # Convert to numpy array
                buffer = img_converted.constBits()
                bytes_per_pixel = img_converted.depth() // 8
                buffer.setsize(img_converted.height() * img_converted.width() * bytes_per_pixel)

                arr = (
                    np.frombuffer(buffer, dtype=np.uint8)
                    .reshape((img_converted.height(), img_converted.width(), bytes_per_pixel))
                    .copy()
                )

                self._last_frame = arr
                self._pending_detections = {}
                self._current_frame_id += 1

                # Send frame to all detectors
                if hasattr(self, 'detectors') and self.detectors:
                    active_detectors = 0
                    for det in self.detectors:
                        if det and det.isRunning():
                            det.set_frame(arr, self._current_frame_id)
                            active_detectors += 1
                        else:
                            logger.warning(
                                "%s: Detector %s not running",
                                self.objectName(),
                                det.objectName() if det else "None"
                            )
                    
                    if active_detectors == 0:
                        logger.warning("%s: No hay detectores activos", self.objectName())

            except Exception as e:
                logger.error("%s: Error processing frame: %s", self.objectName(), e, exc_info=True)

    def _qimage_from_frame(self, frame: QVideoFrame) -> QImage | None:
        """Convert QVideoFrame to QImage."""
        try:
            # Try to map frame directly for efficiency
            if frame.map(QVideoFrame.MapMode.ReadOnly):
                try:
                    pf = frame.pixelFormat()
                    rgb_formats = {
                        getattr(QVideoFrameFormat.PixelFormat, name)
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
                        ]
                        if hasattr(QVideoFrameFormat.PixelFormat, name)
                    }
                    
                    if pf in rgb_formats:
                        img_format = QVideoFrameFormat.imageFormatFromPixelFormat(pf)
                        if img_format != QImage.Format.Format_Invalid:
                            return QImage(
                                frame.bits(),
                                frame.width(),
                                frame.height(),
                                frame.bytesPerLine(),
                                img_format,
                            ).copy()
                finally:
                    frame.unmap()
                    
            # Fallback to toImage method
            image = frame.toImage()
            return image if not image.isNull() else None
            
        except Exception as e:
            logger.error("%s: Error converting frame to QImage: %s", self.objectName(), e)
            return None
