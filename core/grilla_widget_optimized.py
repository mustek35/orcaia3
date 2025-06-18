"""
GrillaWidget optimizado que integra OptimizedVideoReader y OptimizedDetectorWorker.
Reemplaza la implementación anterior con mejor rendimiento y gestión de recursos.
"""

import numpy as np
import json
import os
import uuid
from datetime import datetime
from collections import deque
from PyQt6.QtWidgets import QWidget, QSizePolicy, QMenu
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush, QFont, QImage
from PyQt6.QtCore import Qt, pyqtSignal, QRectF, QSizeF, QSize, QPointF, QTimer
from pyqt6.test3.ptz_tracker.ptz_tracker.core.windows_native_video_reader import OptimizedVideoReader, VideoReaderFactory
from core.detector_worker_optimized import OptimizedDetectorWorker, detector_manager
from core.gestor_alertas import GestorAlertas
from core.rtsp_builder import generar_rtsp
from logging_utils import get_logger

logger = get_logger(__name__)

CONFIG_FILE_PATH = "config.json"

class OptimizedGrillaWidget(QWidget):
    """
    Widget de grilla optimizado que integra:
    - OptimizedVideoReader para captura eficiente
    - OptimizedDetectorWorker para análisis IA optimizado
    - Gestión inteligente de recursos y memoria
    - UI responsiva con threading optimizado
    """
    
    log_signal = pyqtSignal(str)
    performance_signal = pyqtSignal(dict)  # Para estadísticas de rendimiento
    
    def __init__(self, filas=18, columnas=22, area=None, parent=None):
        super().__init__(parent)
        
        # Configuración de grilla
        self.filas = filas
        self.columnas = columnas
        self.area = area if area else [0] * (filas * columnas)
        
        # Estado del widget
        self.cam_data = None
        self.widget_id = str(uuid.uuid4())[:8]
        
        # Componentes optimizados
        self.video_reader = None
        self.detector = None
        self.alertas = None
        
        # Datos de visualización
        self.current_pixmap = None
        self.latest_tracked_boxes = []
        self.last_frame = None
        self.original_frame_size = None
        
        # Grilla y selección
        self.temporal = set()
        self.selected_cells = set()
        self.discarded_cells = set()
        
        # Control de rendimiento
        self.frame_skip_counter = 0
        self.FRAME_SKIP_INTERVAL = 2  # Procesar 1 de cada N frames para UI
        
        # Buffers para optimización
        self.detection_buffer = deque(maxlen=5)
        self.ui_update_buffer = deque(maxlen=3)
        
        # Configuración de UI
        self.setFixedSize(640, 480)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        
        # Pixmap pre-renderizado de líneas de grilla
        self._grid_lines_pixmap = None
        self._generate_grid_lines_pixmap()
        
        # Timers para optimización de UI
        self.ui_update_timer = QTimer(self)
        self.ui_update_timer.setSingleShot(True)
        self.ui_update_timer.timeout.connect(self._perform_ui_update)
        self.ui_update_scheduled = False
        
        # Timer para estadísticas
        self.stats_timer = QTimer(self)
        self.stats_timer.timeout.connect(self._update_performance_stats)
        self.stats_timer.start(2000)  # Cada 2 segundos
        
        # Estadísticas de rendimiento
        self.performance_stats = {
            'video_fps': 0.0,
            'detection_fps': 0.0,
            'ui_fps': 0.0,
            'memory_usage': 0.0,
            'cpu_usage': 0.0,
            'active_tracks': 0,
            'detections_count': 0,
            'frames_processed': 0,
            'frames_skipped': 0
        }
        
        self.setObjectName(f"OptGrilla_{self.widget_id}")
        logger.info(f"{self.objectName()}: Widget optimizado inicializado")
    
    def resizeEvent(self, event):
        """Maneja el redimensionamiento del widget"""
        super().resizeEvent(event)
        self._generate_grid_lines_pixmap()
    
    def _generate_grid_lines_pixmap(self):
        """Genera pixmap pre-renderizado de líneas de grilla"""
        if self.width() <= 0 or self.height() <= 0:
            self._grid_lines_pixmap = None
            return
        
        try:
            pixmap = QPixmap(self.size())
            pixmap.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(pixmap)
            painter.setPen(QColor(100, 100, 100, 100))
            
            cell_w = self.width() / self.columnas
            cell_h = self.height() / self.filas
            
            # Líneas horizontales
            for row in range(self.filas + 1):
                y = int(row * cell_h)
                painter.drawLine(0, y, self.width(), y)
            
            # Líneas verticales
            for col in range(self.columnas + 1):
                x = int(col * cell_w)
                painter.drawLine(x, 0, x, self.height())
            
            painter.end()
            self._grid_lines_pixmap = pixmap
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error generando grilla: {e}")
            self._grid_lines_pixmap = None
    
    def mostrar_vista(self, cam_data):
        """Inicia la visualización de una cámara con componentes optimizados"""
        try:
            logger.info(f"{self.objectName()}: Iniciando vista para cámara {cam_data.get('ip')}")
            
            # Detener componentes anteriores si existen
            self._stop_current_components()
            
            # Guardar configuración de cámara
            self.cam_data = cam_data.copy()
            
            # Generar RTSP si no existe
            if "rtsp" not in self.cam_data:
                self.cam_data["rtsp"] = generar_rtsp(self.cam_data)
            
            # Cargar configuración de celdas descartadas
            self._load_discarded_cells()
            
            # Configurar gestor de alertas
            self._setup_alertas()
            
            # Iniciar componentes optimizados
            self._start_video_reader()
            self._start_detector()
            
            self.registrar_log(f"Vista iniciada para {self.cam_data.get('ip')}")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error iniciando vista: {e}")
            self.registrar_log(f"Error iniciando vista: {e}")
    
    def _stop_current_components(self):
        """Detiene los componentes actuales de forma segura"""
        try:
            # Detener detector
            if self.detector:
                logger.info(f"{self.objectName()}: Deteniendo detector")
                self.detector.stop_detection()
                detector_manager.remove_detector(self.widget_id)
                self.detector = None
            
            # Detener video reader
            if self.video_reader:
                logger.info(f"{self.objectName()}: Deteniendo video reader")
                self.video_reader.stop()
                self.video_reader.deleteLater()
                self.video_reader = None
            
            # Limpiar buffers
            self.detection_buffer.clear()
            self.ui_update_buffer.clear()
            self.latest_tracked_boxes.clear()
            
            logger.info(f"{self.objectName()}: Componentes detenidos")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error deteniendo componentes: {e}")
    
    def _load_discarded_cells(self):
        """Carga la configuración de celdas descartadas"""
        self.discarded_cells = set()
        
        current_cam_ip = self.cam_data.get("ip")
        if not current_cam_ip:
            return
        
        try:
            with open(CONFIG_FILE_PATH, 'r') as f:
                config_data = json.load(f)
            
            camaras_config = config_data.get("camaras", [])
            for cam_config in camaras_config:
                if cam_config.get("ip") == current_cam_ip:
                    discarded_list = cam_config.get("discarded_grid_cells", [])
                    for cell_coords in discarded_list:
                        if isinstance(cell_coords, list) and len(cell_coords) == 2:
                            self.discarded_cells.add(tuple(cell_coords))
                    
                    logger.info(f"{self.objectName()}: Cargadas {len(self.discarded_cells)} celdas descartadas")
                    break
                    
        except Exception as e:
            logger.warning(f"{self.objectName()}: Error cargando celdas descartadas: {e}")
    
    def _setup_alertas(self):
        """Configura el gestor de alertas"""
        try:
            self.alertas = GestorAlertas(
                cam_id=self.widget_id,
                filas=self.filas,
                columnas=self.columnas
            )
            logger.info(f"{self.objectName()}: Gestor de alertas configurado")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error configurando alertas: {e}")
            self.alertas = None
    
    def _start_video_reader(self):
        """Inicia el lector de video optimizado"""
        try:
            rtsp_url = self.cam_data.get("rtsp")
            if not rtsp_url:
                raise ValueError("URL RTSP no disponible")
            
            # Determinar tipo de cámara para configuración óptima
            camera_type = self.cam_data.get("tipo", "fija")
            
            # Crear video reader optimizado
            self.video_reader = VideoReaderFactory.create_reader(
                rtsp_url=rtsp_url,
                camera_type=camera_type,
                hardware_tier="medium"  # Para RTX 3050
            )
            
            # Conectar señales
            self.video_reader.display_ready.connect(self._on_video_frame)
            self.video_reader.frame_ready.connect(self._on_analysis_frame)
            self.video_reader.stats_updated.connect(self._on_video_stats)
            self.video_reader.error_occurred.connect(self._on_video_error)
            
            # Iniciar captura
            self.video_reader.start()
            
            logger.info(f"{self.objectName()}: Video reader iniciado")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error iniciando video reader: {e}")
            self.registrar_log(f"Error iniciando video: {e}")
    
    def _start_detector(self):
        """Inicia el detector optimizado"""
        try:
            model_key = self.cam_data.get("modelo", "Personas")
            
            # Configuración del detector
            detector_config = {
                'confidence': self.cam_data.get("confianza", 0.5),
                'imgsz': self.cam_data.get("imgsz", 640),
                'device': self.cam_data.get("device", "cuda"),
                'enable_tracking': True
            }
            
            # Crear detector usando el manager
            self.detector = detector_manager.create_detector(
                detector_id=self.widget_id,
                model_key=model_key,
                config=detector_config
            )
            
            if not self.detector:
                raise RuntimeError("No se pudo crear el detector")
            
            # Conectar señales
            self.detector.result_ready.connect(self._on_detection_results)
            self.detector.performance_stats.connect(self._on_detector_stats)
            self.detector.error_occurred.connect(self._on_detector_error)
            
            # Iniciar detección
            self.detector.start_detection()
            
            logger.info(f"{self.objectName()}: Detector iniciado para modelo {model_key}")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error iniciando detector: {e}")
            self.registrar_log(f"Error iniciando detector: {e}")
    
    def _on_video_frame(self, pixmap):
        """Maneja frames de video para display"""
        try:
            self.current_pixmap = pixmap
            self._schedule_ui_update()
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error procesando frame de video: {e}")
    
    def _on_analysis_frame(self, frame):
        """Maneja frames para análisis IA"""
        try:
            # Control de skip frames para reducir carga
            self.frame_skip_counter += 1
            if self.frame_skip_counter % self.FRAME_SKIP_INTERVAL != 0:
                self.performance_stats['frames_skipped'] += 1
                return
            
            # Guardar frame para análisis posterior
            self.last_frame = frame.copy()
            
            # Actualizar tamaño original si cambió
            if frame.shape[:2] != getattr(self, '_last_frame_shape', None):
                self.original_frame_size = QSize(frame.shape[1], frame.shape[0])
                self._last_frame_shape = frame.shape[:2]
            
            # Enviar frame al detector
            if self.detector:
                self.detector.add_frame(frame)
            
            self.performance_stats['frames_processed'] += 1
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error procesando frame de análisis: {e}")
    
    def _on_detection_results(self, results, model_key):
        """Procesa resultados de detección"""
        try:
            # Actualizar datos de seguimiento
            self.latest_tracked_boxes = results
            self.performance_stats['active_tracks'] = len(results)
            self.performance_stats['detections_count'] += len(results)
            
            # Procesar para alertas si hay frame disponible
            if self.alertas and self.last_frame is not None:
                self._process_detections_for_alerts(results)
            
            # Programar actualización de UI
            self._schedule_ui_update()
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error procesando resultados: {e}")
    
    def _process_detections_for_alerts(self, tracked_results):
        """Procesa detecciones para el sistema de alertas"""
        try:
            # Convertir resultados tracked a formato de alertas
            alert_detections = []
            
            for result in tracked_results:
                bbox = result.get('bbox', [0, 0, 0, 0])
                cls = result.get('cls', 0)
                
                # Verificar si la detección está en una celda descartada
                if self._is_detection_in_discarded_cell(bbox):
                    continue
                
                # Convertir a formato de alertas: (x1, y1, x2, y2, cls)
                alert_detections.append((
                    bbox[0], bbox[1], bbox[2], bbox[3], cls
                ))
            
            # Procesar con el gestor de alertas
            if alert_detections:
                self.alertas.procesar_detecciones(
                    alert_detections,
                    self.last_frame,
                    self.registrar_log,
                    self.cam_data
                )
                
                # Actualizar temporal para visualización
                self.temporal = self.alertas.temporal.copy()
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error procesando alertas: {e}")
    
    def _is_detection_in_discarded_cell(self, bbox):
        """Verifica si una detección está en una celda descartada"""
        try:
            if not self.original_frame_size or not self.discarded_cells:
                return False
            
            # Calcular centro de la detección
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            
            # Calcular tamaño de celda en coordenadas de video
            cell_w = self.original_frame_size.width() / self.columnas
            cell_h = self.original_frame_size.height() / self.filas
            
            # Determinar celda
            col = int(cx / cell_w)
            row = int(cy / cell_h)
            
            # Asegurar que está dentro de los límites
            col = max(0, min(col, self.columnas - 1))
            row = max(0, min(row, self.filas - 1))
            
            return (row, col) in self.discarded_cells
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error verificando celda descartada: {e}")
            return False
    
    def _on_video_stats(self, stats):
        """Actualiza estadísticas de video"""
        self.performance_stats.update({
            'video_fps': stats.get('fps_capture', 0.0),
            'video_display_fps': stats.get('fps_display', 0.0),
            'connection_stable': stats.get('connection_stable', False),
            'frames_dropped': stats.get('frames_dropped', 0)
        })
    
    def _on_detector_stats(self, stats):
        """Actualiza estadísticas de detector"""
        self.performance_stats.update({
            'detection_fps': stats.get('fps', 0.0),
            'inference_time': stats.get('inference_time', 0.0),
            'tracking_time': stats.get('tracking_time', 0.0),
            'memory_usage': stats.get('memory_usage', 0.0),
            'queue_size': stats.get('queue_size', 0)
        })
    
    def _on_video_error(self, error_msg):
        """Maneja errores de video"""
        logger.error(f"{self.objectName()}: Error de video: {error_msg}")
        self.registrar_log(f"Error de video: {error_msg}")
    
    def _on_detector_error(self, error_msg):
        """Maneja errores de detector"""
        logger.error(f"{self.objectName()}: Error de detector: {error_msg}")
        self.registrar_log(f"Error de detector: {error_msg}")
    
    def _schedule_ui_update(self):
        """Programa una actualización de UI de manera eficiente"""
        if not self.ui_update_scheduled:
            self.ui_update_scheduled = True
            self.ui_update_timer.start(33)  # ~30 FPS máximo para UI
    
    def _perform_ui_update(self):
        """Ejecuta la actualización de UI"""
        self.ui_update_scheduled = False
        self.update()  # Trigger paintEvent
    
    def _update_performance_stats(self):
        """Actualiza y emite estadísticas de rendimiento"""
        try:
            # Calcular UI FPS
            current_time = datetime.now().timestamp()
            if hasattr(self, '_last_stats_time'):
                time_diff = current_time - self._last_stats_time
                if time_diff > 0:
                    frames_in_period = getattr(self, '_ui_frames_in_period', 0)
                    self.performance_stats['ui_fps'] = frames_in_period / time_diff
            
            self._last_stats_time = current_time
            self._ui_frames_in_period = 0
            
            # Emitir estadísticas
            self.performance_signal.emit(self.performance_stats.copy())
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error actualizando estadísticas: {e}")
    
    def paintEvent(self, event):
        """Renderiza el widget optimizado"""
        try:
            painter = QPainter(self)
            
            # Dibujar video de fondo
            if self.current_pixmap and not self.current_pixmap.isNull():
                self._draw_video_background(painter)
            else:
                self._draw_no_signal(painter)
            
            # Dibujar celdas de grilla
            self._draw_grid_cells(painter)
            
            # Dibujar líneas de grilla
            if self._grid_lines_pixmap:
                painter.drawPixmap(self.rect(), self._grid_lines_pixmap)
            
            # Dibujar detecciones
            self._draw_detections(painter)
            
            # Actualizar contador de UI FPS
            self._ui_frames_in_period = getattr(self, '_ui_frames_in_period', 0) + 1
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error en paintEvent: {e}")
        finally:
            painter.end()
    
    def _draw_video_background(self, painter):
        """Dibuja el fondo de video escalado"""
        try:
            target_rect = QRectF(self.rect())
            pixmap_size = QSizeF(self.current_pixmap.size())
            
            # Calcular rectángulo escalado manteniendo aspecto
            scaled_size = pixmap_size.scaled(
                target_rect.size(), 
                Qt.AspectRatioMode.KeepAspectRatio
            )
            
            draw_rect = QRectF()
            draw_rect.setSize(scaled_size)
            draw_rect.moveCenter(target_rect.center())
            
            painter.drawPixmap(draw_rect, self.current_pixmap, QRectF(self.current_pixmap.rect()))
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error dibujando video: {e}")
    
    def _draw_no_signal(self, painter):
        """Dibuja indicador de sin señal"""
        painter.fillRect(self.rect(), QColor("black"))
        painter.setPen(QColor("white"))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Sin señal")
    
    def _draw_grid_cells(self, painter):
        """Dibuja las celdas de grilla con colores"""
        try:
            cell_w = self.width() / self.columnas
            cell_h = self.height() / self.filas
            
            for row in range(self.filas):
                for col in range(self.columnas):
                    index = row * self.columnas + col
                    cell_tuple = (row, col)
                    
                    # Determinar color de celda
                    brush_color = None
                    if cell_tuple in self.discarded_cells:
                        brush_color = QColor(200, 0, 0, 150)  # Rojo para descartadas
                    elif cell_tuple in self.selected_cells:
                        brush_color = QColor(255, 0, 0, 100)  # Rojo claro para seleccionadas
                    elif index in self.temporal:
                        brush_color = QColor(0, 255, 0, 100)  # Verde para detecciones
                    elif index < len(self.area) and self.area[index] == 1:
                        brush_color = QColor(255, 165, 0, 100)  # Naranja para configuradas
                    
                    if brush_color:
                        rect = QRectF(col * cell_w, row * cell_h, cell_w, cell_h)
                        painter.fillRect(rect, brush_color)
                        
        except Exception as e:
            logger.error(f"{self.objectName()}: Error dibujando celdas: {e}")
    
    def _draw_detections(self, painter):
        """Dibuja las detecciones con información de tracking"""
        try:
            if not self.latest_tracked_boxes or not self.original_frame_size:
                return
            
            # Calcular escalas
            orig_w = self.original_frame_size.width()
            orig_h = self.original_frame_size.height()
            
            if orig_w == 0 or orig_h == 0:
                return
            
            # Determinar área de video en el widget
            target_rect = QRectF(self.rect())
            pixmap_size = QSizeF(orig_w, orig_h)
            scaled_size = pixmap_size.scaled(
                target_rect.size(), 
                Qt.AspectRatioMode.KeepAspectRatio
            )
            
            draw_rect = QRectF()
            draw_rect.setSize(scaled_size)
            draw_rect.moveCenter(target_rect.center())
            
            scale_x = draw_rect.width() / orig_w
            scale_y = draw_rect.height() / orig_h
            offset_x = draw_rect.left()
            offset_y = draw_rect.top()
            
            # Configurar fuente
            font = QFont()
            font.setPointSize(10)
            painter.setFont(font)
            
            # Dibujar cada detección
            for detection in self.latest_tracked_boxes:
                self._draw_single_detection(
                    painter, detection, scale_x, scale_y, offset_x, offset_y
                )
                
        except Exception as e:
            logger.error(f"{self.objectName()}: Error dibujando detecciones: {e}")
    
    def _draw_single_detection(self, painter, detection, scale_x, scale_y, offset_x, offset_y):
        """Dibuja una detección individual"""
        try:
            bbox = detection.get('bbox', [0, 0, 0, 0])
            tracker_id = detection.get('id', 'N/A')
            conf = detection.get('conf', 0.0)
            moving_state = detection.get('moving')
            
            # Escalar coordenadas
            x1 = (bbox[0] * scale_x) + offset_x
            y1 = (bbox[1] * scale_y) + offset_y
            x2 = (bbox[2] * scale_x) + offset_x
            y2 = (bbox[3] * scale_y) + offset_y
            
            w = x2 - x1
            h = y2 - y1
            
            # Configurar colores basados en estado
            if moving_state is True:
                color = QColor(0, 255, 0)  # Verde para movimiento
            elif moving_state is False:
                color = QColor(255, 165, 0)  # Naranja para detenido
            else:
                color = QColor(0, 150, 255)  # Azul para procesando
            
            # Dibujar bounding box
            pen = QPen(color, 2)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(x1, y1, w, h))
            
            # Determinar estado de movimiento
            if moving_state is None:
                estado = 'Procesando'
            else:
                estado = 'En movimiento' if moving_state else 'Detenido'
            
            # Dibujar etiqueta
            label_text = f"ID:{tracker_id} C:{conf:.2f} {estado}"
            painter.setPen(QColor("white"))
            
            text_y = y1 - 5
            if text_y < 15:  # Si está muy arriba, mover abajo
                text_y = y1 + 15
            
            painter.drawText(QPointF(x1, text_y), label_text)
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error dibujando detección individual: {e}")
    
    def mousePressEvent(self, event):
        """Maneja clics del mouse para selección de celdas"""
        try:
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
                # Alternar selección
                if clicked_cell in self.selected_cells:
                    self.selected_cells.remove(clicked_cell)
                else:
                    self.selected_cells.add(clicked_cell)
                self._schedule_ui_update()
                
            elif event.button() == Qt.MouseButton.RightButton:
                # Mostrar menú contextual si hay celdas seleccionadas
                if self.selected_cells:
                    self._show_context_menu(event.globalPosition().toPoint())
                    
        except Exception as e:
            logger.error(f"{self.objectName()}: Error en mousePressEvent: {e}")
    
    def _show_context_menu(self, global_pos):
        """Muestra menú contextual para celdas"""
        try:
            menu = QMenu(self)
            
            discard_action = menu.addAction("Descartar celdas para analíticas")
            discard_action.triggered.connect(self._handle_discard_cells)
            
            enable_action = menu.addAction("Habilitar celdas para analíticas")
            enable_action.triggered.connect(self._handle_enable_cells)
            
            menu.exec(global_pos)
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error mostrando menú contextual: {e}")
    
    def _handle_discard_cells(self):
        """Descarta las celdas seleccionadas"""
        try:
            self.discarded_cells.update(self.selected_cells)
            self._save_discarded_cells_to_config()
            self.selected_cells.clear()
            self._schedule_ui_update()
            
            self.registrar_log(f"Descartadas {len(self.selected_cells)} celdas")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error descartando celdas: {e}")
    
    def _handle_enable_cells(self):
        """Habilita las celdas seleccionadas que estaban descartadas"""
        try:
            cells_to_enable = self.selected_cells.intersection(self.discarded_cells)
            
            for cell in cells_to_enable:
                self.discarded_cells.remove(cell)
            
            if cells_to_enable:
                self._save_discarded_cells_to_config()
                self.registrar_log(f"Habilitadas {len(cells_to_enable)} celdas")
            
            self.selected_cells.clear()
            self._schedule_ui_update()
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error habilitando celdas: {e}")
    
    def _save_discarded_cells_to_config(self):
        """Guarda la configuración de celdas descartadas"""
        try:
            if not self.cam_data or not self.cam_data.get("ip"):
                logger.warning(f"{self.objectName()}: No hay IP de cámara para guardar config")
                return
            
            current_cam_ip = self.cam_data.get("ip")
            discarded_list = sorted([list(cell) for cell in self.discarded_cells])
            
            # Cargar configuración existente
            config_data = {"camaras": [], "configuracion": {}}
            try:
                with open(CONFIG_FILE_PATH, 'r') as f:
                    config_data = json.load(f)
            except FileNotFoundError:
                pass
            
            # Actualizar o agregar entrada de cámara
            camera_found = False
            for cam_config in config_data.get("camaras", []):
                if cam_config.get("ip") == current_cam_ip:
                    cam_config["discarded_grid_cells"] = discarded_list
                    camera_found = True
                    break
            
            if not camera_found:
                new_entry = self.cam_data.copy()
                new_entry["discarded_grid_cells"] = discarded_list
                config_data["camaras"].append(new_entry)
            
            # Guardar configuración
            with open(CONFIG_FILE_PATH, 'w') as f:
                json.dump(config_data, f, indent=4)
            
            logger.info(f"{self.objectName()}: Configuración de celdas guardada")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error guardando configuración: {e}")
    
    def registrar_log(self, mensaje):
        """Registra un mensaje de log"""
        try:
            fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ip = self.cam_data.get("ip", "IP-desconocida") if self.cam_data else "IP-indefinida"
            mensaje_completo = f"[{fecha_hora}] {self.objectName()} ({ip}): {mensaje}"
            
            self.log_signal.emit(mensaje_completo)
            
            # Escribir a archivo de log
            try:
                with open("eventos_detectados.txt", "a", encoding="utf-8") as f:
                    f.write(mensaje_completo + "\n")
            except:
                pass  # No fallar si no se puede escribir al archivo
                
        except Exception as e:
            logger.error(f"{self.objectName()}: Error registrando log: {e}")
    
    def detener(self):
        """Detiene todos los componentes del widget"""
        try:
            logger.info(f"{self.objectName()}: Iniciando detención...")
            
            # Detener timers
            self.ui_update_timer.stop()
            self.stats_timer.stop()
            
            # Detener componentes
            self._stop_current_components()
            
            # Limpiar datos
            self.current_pixmap = None
            self.last_frame = None
            self.latest_tracked_boxes.clear()
            self.temporal.clear()
            self.selected_cells.clear()
            
            logger.info(f"{self.objectName()}: Detención completada")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error durante detención: {e}")
    
    def pause(self):
        """Pausa el procesamiento temporalmente"""
        try:
            if self.video_reader:
                self.video_reader.pause()
            if self.detector:
                self.detector.pause_detection()
            
            logger.info(f"{self.objectName()}: Pausado")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error pausando: {e}")
    
    def resume(self):
        """Reanuda el procesamiento"""
        try:
            if self.video_reader:
                self.video_reader.resume()
            if self.detector:
                self.detector.resume_detection()
            
            logger.info(f"{self.objectName()}: Reanudado")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error reanudando: {e}")
    
    def get_performance_stats(self):
        """Obtiene estadísticas de rendimiento actuales"""
        return self.performance_stats.copy()
    
    def update_detector_confidence(self, new_confidence):
        """Actualiza la confianza del detector"""
        try:
            if self.detector:
                self.detector.update_confidence(new_confidence)
                self.cam_data["confianza"] = new_confidence
                logger.info(f"{self.objectName()}: Confianza actualizada a {new_confidence}")
            
        except Exception as e:
            logger.error(f"{self.objectName()}: Error actualizando confianza: {e}")


# Función de conveniencia para crear widgets optimizados
def create_optimized_grilla_widget(cam_data, parent=None):
    """
    Crea un widget de grilla optimizado para una cámara específica.
    
    Args:
        cam_data: Datos de configuración de la cámara
        parent: Widget padre
        
    Returns:
        OptimizedGrillaWidget configurado
    """
    try:
        widget = OptimizedGrillaWidget(parent=parent)
        widget.mostrar_vista(cam_data)
        return widget
        
    except Exception as e:
        logger.error(f"Error creando widget optimizado: {e}")
        return None


# Ejemplo de uso para testing
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit
    
    app = QApplication(sys.argv)
    
    class TestWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Test OptimizedGrillaWidget")
            self.resize(1200, 800)
            
            # Widget central
            central_widget = QWidget()
            self.setCentralWidget(central_widget)
            
            layout = QVBoxLayout()
            central_widget.setLayout(layout)
            
            # Widget de grilla optimizado
            cam_data_test = {
                "ip": "192.168.1.100",
                "usuario": "admin", 
                "contrasena": "password",
                "tipo": "fija",
                "rtsp": "rtsp://admin:password@192.168.1.100:554/stream",
                "modelo": "Personas",
                "confianza": 0.5,
                "imgsz": 640,
                "device": "cuda"
            }
            
            self.grilla_widget = OptimizedGrillaWidget()
            layout.addWidget(self.grilla_widget)
            
            # Log area
            self.log_area = QTextEdit()
            self.log_area.setMaximumHeight(200)
            layout.addWidget(self.log_area)
            
            # Conectar señales
            self.grilla_widget.log_signal.connect(self.log_area.append)
            self.grilla_widget.performance_signal.connect(self.update_performance)
            
            # Iniciar vista (descomenta para test real)
            # self.grilla_widget.mostrar_vista(cam_data_test)
        
        def update_performance(self, stats):
            performance_text = f"Video: {stats.get('video_fps', 0):.1f} FPS | "
            performance_text += f"Detection: {stats.get('detection_fps', 0):.1f} FPS | "
            performance_text += f"Tracks: {stats.get('active_tracks', 0)} | "
            performance_text += f"Memory: {stats.get('memory_usage', 0):.1f} MB"
            
            print(performance_text)  # En aplicación real, mostrar en status bar
        
        def closeEvent(self, event):
            if hasattr(self, 'grilla_widget'):
                self.grilla_widget.detener()
            event.accept()
    
    # Solo ejecutar si este archivo se corre directamente
    # window = TestWindow()
    # window.show()
    # sys.exit(app.exec())