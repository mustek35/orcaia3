# test_processor_no_signals.py
"""
Test del procesador simple SIN usar PyQt6 signals para verificar funcionamiento
"""
import time
import threading
import queue
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from logging_utils import get_logger

logger = get_logger(__name__)

class SimpleYOLOProcessorNoSignals:
    """Procesador YOLO simple sin PyQt6 signals"""
    
    def __init__(self, model_path="yolov8n.pt", device="cuda", **kwargs):
        self.model_path = model_path
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence = kwargs.get('confidence_threshold', 0.5)
        self.imgsz = kwargs.get('input_size', kwargs.get('imgsz', 640))
        self.max_det = kwargs.get('max_det', 300)
        self.classes = kwargs.get('classes', None)
        
        self.model = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=5)
        self.results_queue = queue.Queue()  # Queue para resultados
        self.processing_thread = None
        self.results_count = 0
        
        print(f"✅ SimpleYOLOProcessorNoSignals creado en {device}")
    
    def load_model(self):
        """Cargar modelo YOLO"""
        try:
            print(f"🔧 Cargando modelo: {self.model_path}")
            self.model = YOLO(self.model_path)
            
            # Test simple
            test_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = self.model(test_img, verbose=False, imgsz=self.imgsz, conf=self.confidence)
            
            if results:
                print("✅ Modelo cargado y probado exitosamente")
                return True
            else:
                print("❌ Modelo no devuelve resultados")
                return False
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False
    
    def start_processing(self):
        """Iniciar procesamiento"""
        if self.running or not self.model:
            return
        
        self.running = True
        self.results_count = 0
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        print("✅ Procesamiento iniciado")
    
    def stop_processing(self):
        """Detener procesamiento"""
        if not self.running:
            return
        
        self.running = False
        
        # Limpiar queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        # Esperar thread
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=3.0)
        
        print(f"✅ Procesamiento detenido. Total: {self.results_count}")
    
    def add_frame(self, frame, metadata=None):
        """Añadir frame para procesamiento"""
        if not self.running:
            return
        
        try:
            frame_data = {
                'frame': frame.copy(),
                'metadata': metadata or {},
                'timestamp': time.time()
            }
            
            if not self.frame_queue.full():
                self.frame_queue.put_nowait(frame_data)
            else:
                # Descartar frame viejo
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame_data)
                except queue.Empty:
                    self.frame_queue.put_nowait(frame_data)
                    
        except Exception as e:
            print(f"❌ Error añadiendo frame: {e}")
    
    def get_results(self, timeout=0.1):
        """Obtener resultados disponibles"""
        results = []
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                result = self.results_queue.get_nowait()
                results.append(result)
            except queue.Empty:
                break
        
        return results
    
    def _processing_loop(self):
        """Loop principal de procesamiento"""
        print("🚀 Loop de procesamiento iniciado")
        
        while self.running:
            try:
                # Obtener frame
                try:
                    frame_data = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                # Procesar frame
                detections = self._process_frame(frame_data['frame'])
                self.results_count += 1
                
                # Guardar resultado en queue
                result = {
                    'detections': detections,
                    'metadata': frame_data['metadata'],
                    'frame_id': self.results_count,
                    'timestamp': time.time()
                }
                
                self.results_queue.put(result)
                
                print(f"📦 Frame {self.results_count}: {len(detections)} detecciones procesadas")
                
                # Mostrar detecciones
                for det in detections[:3]:  # Primeras 3
                    print(f"   🎯 {det['class_name']}: {det['confidence']:.2f}")
                
                self.frame_queue.task_done()
                time.sleep(0.01)
                
            except Exception as e:
                print(f"❌ Error en processing loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        print("🛑 Loop de procesamiento terminado")
    
    def _process_frame(self, frame):
        """Procesar un frame individual"""
        try:
            if frame is None or frame.size == 0:
                return []
            
            # Ejecutar inferencia
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
            
            detections = []
            
            if results and len(results) > 0:
                result = results[0]
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    if len(boxes) > 0:
                        for i in range(len(boxes)):
                            try:
                                box = boxes[i]
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = float(box.conf[0].cpu().numpy())
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
                                print(f"⚠️ Error procesando detección {i}: {e}")
                                continue
            
            return detections
            
        except Exception as e:
            print(f"❌ Error procesando frame: {e}")
            return []


def test_processor_basic():
    """Test básico del procesador"""
    print("🧪 TEST PROCESADOR BÁSICO (SIN SIGNALS)")
    print("=" * 50)
    
    try:
        # Configuración
        config = {
            'model_path': 'yolov8n.pt',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'confidence_threshold': 0.3,
            'input_size': 640,
            'max_det': 100
        }
        
        print(f"📋 Config: {config}")
        
        # Crear procesador
        processor = SimpleYOLOProcessorNoSignals(**config)
        
        # Cargar modelo
        if not processor.load_model():
            print("❌ Error cargando modelo")
            return False
        
        # Iniciar procesamiento
        processor.start_processing()
        
        # Enviar frames con objetos detectables
        print("\n📤 Enviando frames...")
        for i in range(5):
            # Frame con rectángulos para simular objetos
            frame = np.random.randint(50, 200, (640, 640, 3), dtype=np.uint8)
            
            # Añadir formas que YOLO pueda detectar como objetos
            cv2.rectangle(frame, (100, 100), (200, 300), (255, 255, 255), -1)  # Objeto blanco
            cv2.rectangle(frame, (300, 200), (450, 400), (200, 200, 200), -1)  # Objeto gris
            cv2.circle(frame, (500, 150), 50, (255, 255, 255), -1)             # Círculo
            
            metadata = {'frame_id': i, 'source': 'test'}
            processor.add_frame(frame, metadata)
            print(f"   📤 Frame {i} enviado")
            time.sleep(0.3)
        
        # Esperar y recoger resultados
        print("\n⏳ Esperando resultados...")
        time.sleep(3)
        
        # Recoger todos los resultados
        all_results = processor.get_results(timeout=1.0)
        
        # Detener
        processor.stop_processing()
        
        # Analizar resultados
        print(f"\n📊 ANÁLISIS:")
        print(f"   📤 Frames enviados: 5")
        print(f"   🎯 Resultados recibidos: {len(all_results)}")
        
        total_detections = 0
        for result in all_results:
            detections = result['detections']
            frame_id = result['metadata'].get('frame_id', '?')
            total_detections += len(detections)
            print(f"   📦 Frame {frame_id}: {len(detections)} detecciones")
            
            # Mostrar detecciones detalladas
            for det in detections[:2]:  # Primeras 2
                bbox = det['bbox']
                print(f"      🎯 {det['class_name']}: {det['confidence']:.2f} en [{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}]")
        
        print(f"   🎯 Total detecciones: {total_detections}")
        
        success = len(all_results) > 0
        print(f"   ✅ Estado: {'FUNCIONAL' if success else 'FALLO'}")
        
        return success
        
    except Exception as e:
        print(f"❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_processor_with_rtsp():
    """Test con RTSP real"""
    print("\n🎥 TEST CON RTSP REAL")
    print("=" * 40)
    
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
        
        processor = SimpleYOLOProcessorNoSignals(**config)
        
        if not processor.load_model():
            print("❌ Error cargando modelo")
            cap.release()
            return False
        
        processor.start_processing()
        
        # Procesar frames RTSP
        print("📹 Procesando frames RTSP...")
        frames_sent = 0
        
        for i in range(6):
            ret, frame = cap.read()
            if ret and frame is not None:
                # Redimensionar si es muy grande
                if frame.shape[0] > 1080:
                    frame = cv2.resize(frame, (1280, 720))
                
                metadata = {
                    'frame_id': i,
                    'source': 'rtsp',
                    'shape': frame.shape
                }
                
                processor.add_frame(frame, metadata)
                frames_sent += 1
                print(f"   📤 RTSP Frame {i} enviado {frame.shape}")
                time.sleep(0.5)
            else:
                print(f"⚠️ Fallo leyendo frame {i}")
                break
        
        time.sleep(4)  # Esperar procesamiento
        
        # Recoger resultados
        results = processor.get_results(timeout=2.0)
        
        # Cleanup
        cap.release()
        processor.stop_processing()
        
        print(f"\n📊 RESULTADOS RTSP:")
        print(f"   📤 Frames enviados: {frames_sent}")
        print(f"   🎯 Resultados: {len(results)}")
        
        person_count = car_count = other_count = 0
        
        for result in results:
            detections = result['detections']
            frame_id = result['metadata'].get('frame_id', '?')
            
            frame_persons = sum(1 for d in detections if d['class_name'] == 'person')
            frame_cars = sum(1 for d in detections if d['class_name'] in ['car', 'truck', 'bus'])
            frame_others = len(detections) - frame_persons - frame_cars
            
            person_count += frame_persons
            car_count += frame_cars
            other_count += frame_others
            
            print(f"   📦 Frame {frame_id}: {len(detections)} det. (👤{frame_persons} 🚗{frame_cars} 🔍{frame_others})")
        
        print(f"   📊 Total: 👤{person_count} personas, 🚗{car_count} vehículos, 🔍{other_count} otros")
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Error en test RTSP: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🏁 TEST PROCESADOR SIMPLE COMPLETO (SIN QT)")
    print("=" * 60)
    
    # Test básico
    basic_ok = test_processor_basic()
    
    # Test RTSP si el básico funciona
    if basic_ok:
        rtsp_ok = test_processor_with_rtsp()
    else:
        rtsp_ok = False
    
    print("\n" + "=" * 60)
    print("🎯 RESULTADO FINAL:")
    print(f"   ⚙️  Procesador básico:  {'✅' if basic_ok else '❌'}")
    print(f"   🎥 Integración RTSP:   {'✅' if rtsp_ok else '❌'}")
    
    if basic_ok and rtsp_ok:
        print("\n🎉 ¡PROCESADOR COMPLETAMENTE FUNCIONAL!")
        print("📋 El problema anterior era solo con PyQt6 signals")
        print("🚀 Sistema listo para producción")
        
        print("\n💡 PRÓXIMOS PASOS:")
        print("   1. Integrar este procesador en detector_worker.py")
        print("   2. Reemplazar CUDAPipelineProcessor problemático")
        print("   3. Usar callbacks o polling en lugar de signals")
        
    elif basic_ok:
        print("\n⚠️ Procesador funciona, pero RTSP tiene problemas")
        print("💡 Usar con frames sintéticos o revisar conexión RTSP")
    else:
        print("\n❌ Problemas básicos con el procesador")


if __name__ == "__main__":
    main()