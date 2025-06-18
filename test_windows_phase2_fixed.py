# test_windows_phase2_individual.py
import time
import numpy as np
from core.windows_native_video_reader import WindowsVideoReaderFactory
from core.cuda_pipeline_processor import cuda_pipeline_manager

def test_video_reader():
    print("🧪 Test Windows Video Reader...")
    
    # Crear reader
    reader = WindowsVideoReaderFactory.create_reader(
        rtsp_url="rtsp://root:%40Remoto753524@19.10.10.132:554/media/video1",
        camera_type="fija", 
        performance_profile="balanced"
    )
    
    frame_count = 0
    
    def on_frame(frame):
        nonlocal frame_count
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"📦 Frames recibidos: {frame_count}")
    
    def on_stats(stats):
        fps = stats.get('fps_capture', 0)
        backend = stats.get('backend_info', 'unknown')
        print(f"📊 FPS: {fps:.1f}, Backend: {backend}")
    
    reader.frame_ready.connect(on_frame)
    reader.stats_updated.connect(on_stats)
    
    # Test por 10 segundos
    reader.start()
    time.sleep(10)
    reader.stop()
    
    print(f"✅ Video test completado: {frame_count} frames")

def test_cuda_processor():
    print("🧪 Test CUDA Processor...")
    
    # Crear procesador
    config = {
        'model_path': 'yolov8n.pt',  # Modelo ligero
        'batch_size': 2,
        'confidence_threshold': 0.5,
        'half_precision': True,
    }
    
    processor = cuda_pipeline_manager.create_processor('test', config)
    processor.load_model()
    processor.start_processing()
    
    # Test con frames sintéticos
    test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    results_count = 0
    
    def on_results(detections, metadata):
        nonlocal results_count
        results_count += 1
        print(f"🎯 Resultado {results_count}: {len(detections)} detecciones")
    
    processor.results_ready.connect(on_results)
    
    # Enviar frames
    for i in range(20):
        processor.add_frame(test_frame, {'frame_id': i})
        time.sleep(0.1)
    
    time.sleep(2)  # Esperar procesamiento
    processor.stop_processing()
    cuda_pipeline_manager.stop_all_processors()
    
    print(f"✅ CUDA test completado: {results_count} resultados")

if __name__ == "__main__":
    test_video_reader()
    test_cuda_processor()