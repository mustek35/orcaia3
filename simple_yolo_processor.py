# simple_yolo_processor.py
"""
Procesador YOLO simplificado que evita la complejidad del CUDAPipelineProcessor
"""
import time
import threading
import queue
import numpy as np
import torch
from ultralytics import YOLO
from PyQt6.QtCore import QObject, pyqtSignal
from logging_utils import get_logger

logger = get_logger(__name__)

class SimpleYOLOProcessor(QObject):
    """Procesador YOLO simple y confiable"""
    
    results_ready = pyqtSignal(list, dict)  # detections, metadata
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_path="yolov8n.pt", device="cuda", **kwargs):
        super().__init__()
        
        # Configuración simple
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence = kwargs.get('confidence_threshold', 0.5)
        self.imgsz = kwargs.get('input_size', kwargs.get('imgsz', 640))
        self.max_det = kwargs.get('max_det', 300)
        self.classes = kwargs.get('classes', None)
        
        # Estado interno
        self.model = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.processing_thread = None
        
        logger.info(f"SimpleYOLOProcessor creado: {model_path} en {self.device}")
    
    def load_model(self):
        """Cargar modelo YOLO"""
        try:
            logger.info(f"Cargando modelo: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Mover a GPU si está disponible
            if self.device.startswith('cuda') and torch.cuda.is_available():
                # El modelo se mueve automáticamente con ultralytics
                logger.info(f"Modelo en GPU: {self.device}")
            else:
                logger.info("Modelo en CPU")
            
            # Test simple
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = self.model(test_img, verbose=False, imgsz=self.imgsz)
            logger.info("✅ Modelo cargado y probado exitosamente")
            
            return True
            
        except Exception as e:
            error_msg = f"Error cargando modelo: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def start_processing(self):
        """Iniciar procesamiento"""
        if self.running or not self.model:
            return
        
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("Procesamiento iniciado")
    
    def stop_processing(self):
        """Detener procesamiento"""
        if not self.running:
            return
        
        self.running = False
        
        # Limpiar queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Esperar thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2.0)
        
        logger.info("Procesamiento detenido")
    
    def add_frame(self, frame, metadata=None):
        """Añadir frame para procesamiento"""
        if not self.running:
            return
        
        try:
            frame_data = {
                'frame': frame.copy() if isinstance(frame, np.ndarray) else frame,
                'metadata': metadata or {},
                'timestamp': time.time()
            }
            
            # Añadir sin bloquear
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame_data)
            else:
                # Si la queue está llena, descartar frame más viejo
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame_data)
                except queue.Empty:
                    pass
                    
        except Exception as e:
            logger.warning(f"Error añadiendo frame: {e}")
    
    def _processing_loop(self):
        """Loop principal de procesamiento"""
        logger.info("Loop de procesamiento iniciado")
        
        while self.running:
            try:
                # Obtener frame con timeout
                try:
                    frame_data = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Procesar frame
                detections = self._process_frame(frame_data['frame'])
                
                # Emitir resultados
                self.results_ready.emit(detections, frame_data['metadata'])
                
                # Marcar como procesado
                self.frame_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error en loop de procesamiento: {e}")
                self.error_occurred.emit(str(e))
                time.sleep(0.1)  # Pausa antes de continuar
        
        logger.info("Loop de procesamiento terminado")
    
    def _process_frame(self, frame):
        """Procesar un frame individual"""
        try:
            # Ejecutar inferencia
            results = self.model(
                frame,
                conf=self.confidence,
                imgsz=self.imgsz,
                max_det=self.max_det,
                classes=self.classes,
                verbose=False
            )
            
            # Extraer detecciones
            detections = []
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                
                if boxes is not None and len(boxes) > 0:
                    # Convertir a formato estándar
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': conf,
                            'class': cls,
                            'class_name': self.model.names.get(cls, f'class_{cls}')
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
            return []


class SimpleProcessorManager:
    """Manager simple para procesadores YOLO"""
    
    def __init__(self):
        self.processors = {}
    
    def create_processor(self, processor_id, config):
        """Crear nuevo procesador"""
        if processor_id in self.processors:
            self.remove_processor(processor_id)
        
        processor = SimpleYOLOProcessor(**config)
        self.processors[processor_id] = processor
        
        logger.info(f"Procesador simple {processor_id} creado")
        return processor
    
    def remove_processor(self, processor_id):
        """Remover procesador"""
        if processor_id in self.processors:
            processor = self.processors[processor_id]
            processor.stop_processing()
            del self.processors[processor_id]
            logger.info(f"Procesador {processor_id} removido")
    
    def stop_all_processors(self):
        """Detener todos los procesadores"""
        for processor_id in list(self.processors.keys()):
            self.remove_processor(processor_id)
        logger.info("Todos los procesadores simples detenidos")


# Instancia global
simple_processor_manager = SimpleProcessorManager()


def test_simple_processor():
    """Test del procesador simple"""
    print("🧪 TEST PROCESADOR SIMPLE")
    print("=" * 40)
    
    try:
        # Configuración
        config = {
            'model_path': 'yolov8n.pt',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'confidence_threshold': 0.5,
            'input_size': 640,
            'max_det': 100
        }
        
        # Crear procesador
        processor = simple_processor_manager.create_processor('test_simple', config)
        
        # Cargar modelo
        if not processor.load_model():
            print("❌ Fallo cargando modelo")
            return False
        
        # Iniciar procesamiento
        processor.start_processing()
        
        # Test con frames
        results_count = 0
        
        def on_results(detections, metadata):
            nonlocal results_count
            results_count += 1
            print(f"🎯 Frame {metadata.get('frame_id', '?')}: {len(detections)} detecciones")
            
            # Mostrar primera detección si existe
            if detections:
                det = detections[0]
                print(f"   📦 {det['class_name']}: {det['confidence']:.2f}")
        
        processor.results_ready.connect(on_results)
        
        # Enviar frames de prueba
        print("📤 Enviando frames...")
        for i in range(5):
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            processor.add_frame(frame, {'frame_id': i})
            time.sleep(0.2)
        
        # Esperar resultados
        time.sleep(2)
        
        # Cleanup
        simple_processor_manager.stop_all_processors()
        
        print(f"✅ Test completado: {results_count} resultados")
        return results_count > 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_simple_processor()