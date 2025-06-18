# test_fixed_simple_processor.py
"""
Test del procesador simple con fixes para que funcione correctamente
"""
import time
import threading
import queue
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from PyQt6.QtWidgets import QApplication
from logging_utils import get_logger

logger = get_logger(__name__)

class FixedSimpleYOLOProcessor(QObject):
    """Procesador YOLO simple y funcional"""
    
    results_ready = pyqtSignal(list, dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, model_path="yolov8n.pt", device="cuda", **kwargs):
        super().__init__()
        
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence = kwargs.get('confidence_threshold', 0.5)
        self.imgsz = kwargs.get('input_size', kwargs.get('imgsz', 640))
        self.max_det = kwargs.get('max_det', 300)
        self.classes = kwargs.get('classes', None)
        
        self.model = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)  # Reducir tamaño de queue
        self.processing_thread = None
        self.results_count = 0
        
        logger.info(f"FixedSimpleYOLOProcessor creado: {device}")
    
    def load_model(self):
        """Cargar modelo YOLO"""
        try:
            logger.info(f"Cargando modelo: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Test simple para verificar que funciona
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = self.model(test_img, verbose=False, imgsz=self.imgsz, conf=self.confidence)
            
            if results:
                logger.info("✅ Modelo cargado y probado exitosamente")
                return True
            else:
                logger.error("❌ Modelo no devuelve resultados")
                return False
            
        except Exception as e:
            error_msg = f"Error cargando modelo: {e}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return False
    
    def start_processing(self):
        """Iniciar procesamiento"""
        if self.running or not self.model:
            logger.warning("Ya está corriendo o modelo no cargado")
            return
        
        self.running = True
        self.results_count = 0
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        logger.info("✅ Procesamiento iniciado")
    
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
            self.processing_thread.join(timeout=3.0)
        
        logger.info(f"✅ Procesamiento detenido. Total procesados: {self.results_count}")
    
    def add_frame(self, frame, metadata=None):
        """Añadir frame para procesamiento"""
        if not self.running:
            logger.warning("Procesador no está corriendo")
            return
        
        if not isinstance(frame, np.ndarray):
            logger.warning("Frame no es numpy array")
            return
        
        try:
            frame_data = {
                'frame': frame.copy(),
                'metadata': metadata or {},
                'timestamp': time.time()
            }
            
            # Añadir sin bloquear
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame_data)
                logger.debug(f"Frame añadido a queue (size: {self.frame_queue.qsize()})")
            else:
                # Si está llena, descartar frame más viejo
                try:
                    old_frame = self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame_data)
                    logger.debug("Frame viejo descartado, nuevo añadido")
                except queue.Empty:
                    self.frame_queue.put_nowait(frame_data)
                    
        except Exception as e:
            logger.error(f"Error añadiendo frame: {e}")
    
    def _processing_loop(self):
        """Loop principal de procesamiento"""
        logger.info("🚀 Loop de procesamiento iniciado")
        
        while self.running:
            try:
                # Obtener frame con timeout
                try:
                    frame_data = self.frame_queue.get(timeout=0.5)  # Timeout más largo
                    logger.debug("Frame obtenido de queue")
                except queue.Empty:
                    continue
                
                # Procesar frame
                detections = self._process_frame(frame_data['frame'])
                self.results_count += 1
                
                logger.info(f"📦 Frame procesado {self.results_count}: {len(detections)} detecciones")
                
                # Emitir resultados
                self.results_ready.emit(detections, frame_data['metadata'])
                
                # Marcar como procesado
                self.frame_queue.task_done()
                
                # Pequeña pausa para no saturar
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error en loop de procesamiento: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        logger.info("🛑 Loop de procesamiento terminado")
    
    def _process_frame(self, frame):
        """Procesar un frame individual"""
        try:
            # Verificar frame
            if frame is None or frame.size == 0:
                logger.warning("Frame vacío o None")
                return []
            
            logger.debug(f"Procesando frame shape: {frame.shape}")
            
            # Ejecutar inferencia con configuración simple
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                imgsz=self.imgsz,
                max_det=self.max_det,
                classes=self.classes,
                verbose=False,
                save=False,
                show=False
            )
            
            # Extraer detecciones
            detections = []
            
            if results and len(results) > 0:
                result = results[0]  # Primer resultado
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    if len(boxes) > 0:
                        # Convertir a formato estándar
                        for i in range(len(boxes)):
                            try:
                                box = boxes[i]
                                
                                # Coordenadas
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                
                                # Confianza
                                conf = float(box.conf[0].cpu().numpy())
                                
                                # Clase
                                cls = int(box.cls[0].cpu().numpy())
                                class_name = self.model.names.get(cls, f'class_{cls}')
                                
                                detection = {
                                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                                    'confidence': conf,
                                    'class': cls,
                                    'class_name': class_name
                                }
                                
                                detections.append(detection)
                                
                            except Exception as e:
                                logger.warning(f"Error procesando detección {i}: {e}")
                                continue
            
            logger.debug(f"Detecciones extraídas: {len(detections)}")
            return detections
            
        except Exception as e:
            logger.error(f"Error procesando frame: {e}")
            import traceback
            traceback.print_exc()
            return []


def test_fixed_simple_processor():
    """Test del procesador simple corregido"""
    print("🧪 TEST PROCESADOR SIMPLE CORREGIDO")
    print("=" * 50)
    
    # Crear aplicación Qt para signals
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Configuración
        config = {
            'model_path': 'yolov8n.pt',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'confidence_threshold': 0.3,  # Más bajo para ver más detecciones
            'input_size': 640,
            'max_det': 100
        }
        
        print(f"📋 Configuración: {config}")
        
        # Crear procesador
        processor = FixedSimpleYOLOProcessor(**config)
        
        # Contador de resultados
        results_received = []
        
        def on_results(detections, metadata):
            results_received.append((detections, metadata))
            frame_id = metadata.get('frame_id', '?')
            print(f"🎯 Frame {frame_id}: {len(detections)} detecciones")
            
            # Mostrar primeras detecciones
            for i, det in enumerate(detections[:3]):  # Primeras 3
                print(f"   📦 {det['class_name']}: {det['confidence']:.2f}")
        
        def on_error(error_msg):
            print(f"❌ Error: {error_msg}")
        
        # Conectar signals
        processor.results_ready.connect(on_results)
        processor.error_occurred.connect(on_error)
        
        # Cargar modelo
        print("🔧 Cargando modelo...")
        if not processor.load_model():
            print("❌ Fallo cargando modelo")
            return False
        
        # Iniciar procesamiento
        print("🚀 Iniciando procesamiento...")
        processor.start_processing()
        
        # Crear frames más interesantes (no completamente aleatorios)
        print("📤 Enviando frames...")
        
        for i in range(5):
            # Crear frame con algunos patrones (más probable de tener "detecciones")
            frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Añadir algunos rectangulos para simular objetos
            cv2.rectangle(frame, (100, 100), (200, 200), (255, 0, 0), -1)
            cv2.rectangle(frame, (300, 300), (400, 450), (0, 255, 0), -1)
            
            metadata = {
                'frame_id': i,
                'timestamp': time.time(),
                'source': 'test'
            }
            
            processor.add_frame(frame, metadata)
            print(f"   📤 Frame {i} enviado")
            time.sleep(0.5)  # Pausa entre frames
        
        # Esperar resultados
        print("⏳ Esperando resultados...")
        time.sleep(3)
        
        # Detener
        print("🛑 Deteniendo procesador...")
        processor.stop_processing()
        
        # Resultados
        print(f"\n📊 RESULTADOS:")
        print(f"   📤 Frames enviados: 5")
        print(f"   🎯 Resultados recibidos: {len(results_received)}")
        print(f"   ✅ Pipeline funcional: {'Sí' if len(results_received) > 0 else 'No'}")
        
        total_detections = sum(len(detections) for detections, _ in results_received)
        print(f"   📦 Total detecciones: {total_detections}")
        
        if len(results_received) > 0:
            print("\n📋 Detalle de resultados:")
            for i, (detections, metadata) in enumerate(results_received):
                frame_id = metadata.get('frame_id', '?')
                print(f"   Frame {frame_id}: {len(detections)} detecciones")
        
        return len(results_received) > 0
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_with_rtsp():
    """Test con stream RTSP real"""
    print("\n🎥 TEST CON RTSP REAL")
    print("=" * 40)
    
    # Crear aplicación Qt para signals
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    
    try:
        # Configurar RTSP
        rtsp_url = "rtsp://root:%40Remoto753524@19.10.10.132:554/media/video1"
        
        cap = cv2.VideoCapture()
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not cap.open(rtsp_url, cv2.CAP_FFMPEG):
            print("❌ No se pudo conectar a RTSP")
            return False
        
        print("✅ RTSP conectado")
        
        # Configurar procesador
        config = {
            'model_path': 'yolov8n.pt',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'confidence_threshold': 0.25,
            'input_size': 640
        }
        
        processor = FixedSimpleYOLOProcessor(**config)
        
        results_received = []
        
        def on_results(detections, metadata):
            results_received.append((detections, metadata))
            frame_id = metadata.get('frame_id', '?')
            print(f"🎯 RTSP Frame {frame_id}: {len(detections)} detecciones")
            
            # Mostrar detecciones interesantes
            for det in detections:
                if det['confidence'] > 0.5:  # Solo alta confianza
                    print(f"   📦 {det['class_name']}: {det['confidence']:.2f}")
        
        processor.results_ready.connect(on_results)
        
        # Cargar modelo
        if not processor.load_model():
            print("❌ Fallo cargando modelo")
            cap.release()
            return False
        
        # Iniciar procesamiento
        processor.start_processing()
        
        # Procesar frames RTSP
        print("📹 Procesando frames RTSP...")
        frames_processed = 0
        
        for i in range(8):  # 8 frames
            ret, frame = cap.read()
            if ret and frame is not None:
                # Redimensionar si es necesario
                if frame.shape[0] > 1080 or frame.shape[1] > 1920:
                    frame = cv2.resize(frame, (1280, 720))
                
                metadata = {
                    'frame_id': i,
                    'source': 'rtsp',
                    'timestamp': time.time()
                }
                
                processor.add_frame(frame, metadata)
                frames_processed += 1
                print(f"   📤 RTSP Frame {i} enviado ({frame.shape})")
                time.sleep(0.3)  # Pausa entre frames
            else:
                print(f"⚠️ Fallo leyendo RTSP frame {i}")
                break
        
        time.sleep(3)  # Esperar procesamiento
        
        # Cleanup
        cap.release()
        processor.stop_processing()
        
        print(f"\n✅ Test RTSP completado:")
        print(f"   📤 Frames enviados: {frames_processed}")
        print(f"   🎯 Resultados: {len(results_received)}")
        
        return len(results_received) > 0
        
    except Exception as e:
        print(f"❌ Error en test RTSP: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🏁 TEST PROCESADOR SIMPLE COMPLETO")
    print("=" * 60)
    
    # Test 1: Procesador básico
    basic_ok = test_fixed_simple_processor()
    
    # Test 2: Con RTSP
    if basic_ok:
        rtsp_ok = test_simple_with_rtsp()
    else:
        rtsp_ok = False
    
    print("\n" + "=" * 60)
    print("🎯 RESULTADO FINAL:")
    print(f"   ⚙️  Procesador básico:  {'✅' if basic_ok else '❌'}")
    print(f"   🎥 Integración RTSP:   {'✅' if rtsp_ok else '❌'}")
    
    if basic_ok and rtsp_ok:
        print("\n🎉 ¡PROCESADOR SIMPLE FUNCIONAL!")
        print("🚀 Puedes usar este procesador en lugar del complejo")
        print("💡 Próximo paso: Integrar en tu aplicación principal")
    elif basic_ok:
        print("\n⚠️ Procesador funciona, revisa RTSP")
        print("💡 Puedes desarrollar con frames sintéticos")
    else:
        print("\n❌ Problemas con procesador básico")
        print("💡 Revisa logs de error arriba")


if __name__ == "__main__":
    main()