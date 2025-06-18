# test_final_complete.py
"""
Test final después de todos los patches
"""
import time
import numpy as np

def test_complete_pipeline():
    print("🧪 TEST PIPELINE COMPLETO")
    print("=" * 50)
    
    try:
        from core.cuda_pipeline_processor import cuda_pipeline_manager
        
        # Configuración ULTRA COMPLETA
        config = {
            'model_path': 'yolov8n.pt',
            'batch_size': 1,
            'confidence_threshold': 0.5,
            'half_precision': True,
            'compile_model': False,
            'optimize_for_inference': True,
            'warmup_iterations': 1,
            'max_det': 100,
            'iou_threshold': 0.45,
            'device': 'cuda:0',
            'verbose': False,
            'input_size': 640,      # ✅ CLAVE AÑADIDA
            'imgsz': 640,          # ✅ CLAVE AÑADIDA
            'classes': None,        # ✅ CLAVE AÑADIDA
            'agnostic_nms': False, # ✅ CLAVE AÑADIDA
            'memory_fraction': 0.8,
            'allow_growth': True,
            'num_workers': 1,
            'prefetch_factor': 2,
            'pin_memory': True,
            'processing_timeout': 5.0,
            'max_queue_size': 10,
            'clear_cache_interval': 100
        }
        
        print("📦 Creando procesador...")
        processor = cuda_pipeline_manager.create_processor('final_test', config)
        
        print("🔧 Cargando modelo...")
        processor.load_model()
        
        print("🚀 Iniciando procesamiento...")
        processor.start_processing()
        
        # Test con frame realista
        test_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        results_count = 0
        results_received = []
        
        def on_results(detections, metadata):
            nonlocal results_count, results_received
            results_count += 1
            results_received.append((len(detections), metadata))
            print(f"🎯 Resultado {results_count}: {len(detections)} detecciones")
        
        processor.results_ready.connect(on_results)
        
        # Enviar frames
        print("📤 Enviando frames...")
        for i in range(5):
            metadata = {
                'frame_id': i,
                'timestamp': time.time(),
                'source': 'test_final'
            }
            processor.add_frame(test_frame, metadata)
            time.sleep(0.2)
        
        # Esperar resultados
        print("⏳ Esperando resultados...")
        time.sleep(3)
        
        # Cleanup
        processor.stop_processing()
        cuda_pipeline_manager.stop_all_processors()
        
        # Resultados
        print("\n📊 RESULTADOS:")
        print(f"   📦 Frames enviados: 5")
        print(f"   🎯 Resultados recibidos: {results_count}")
        print(f"   ✅ Pipeline funcional: {'Sí' if results_count > 0 else 'No'}")
        
        if results_received:
            for i, (det_count, meta) in enumerate(results_received):
                print(f"   📋 Frame {i}: {det_count} detecciones")
        
        return results_count > 0
        
    except Exception as e:
        print(f"❌ Error en pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_rtsp_integration():
    """Test integración RTSP + CUDA"""
    print("\n🎥 TEST INTEGRACIÓN RTSP + CUDA")
    print("=" * 50)
    
    try:
        import cv2
        from core.cuda_pipeline_processor import cuda_pipeline_manager
        
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
            'batch_size': 1,
            'confidence_threshold': 0.3,
            'input_size': 640,
            'compile_model': False,
            'half_precision': True,
            'optimize_for_inference': True,
            'warmup_iterations': 1
        }
        
        processor = cuda_pipeline_manager.create_processor('rtsp_test', config)
        processor.load_model()
        processor.start_processing()
        
        results_count = 0
        
        def on_results(detections, metadata):
            nonlocal results_count
            results_count += 1
            print(f"🎯 Frame {metadata.get('frame_id', '?')}: {len(detections)} detecciones")
        
        processor.results_ready.connect(on_results)
        
        # Procesar frames reales
        print("📹 Procesando frames RTSP...")
        for i in range(10):
            ret, frame = cap.read()
            if ret:
                processor.add_frame(frame, {'frame_id': i, 'source': 'rtsp'})
                time.sleep(0.1)
            else:
                print(f"⚠️ Fallo leyendo frame {i}")
                break
        
        time.sleep(2)  # Esperar procesamiento
        
        # Cleanup
        cap.release()
        processor.stop_processing()
        cuda_pipeline_manager.stop_all_processors()
        
        print(f"\n✅ Integración RTSP+CUDA: {results_count} frames procesados")
        return results_count > 0
        
    except Exception as e:
        print(f"❌ Error integración: {e}")
        return False

def main():
    print("🏁 TEST FINAL COMPLETO")
    print("=" * 60)
    
    # Test 1: Pipeline puro
    pipeline_ok = test_complete_pipeline()
    
    # Test 2: Integración RTSP
    integration_ok = test_rtsp_integration()
    
    print("\n" + "=" * 60)
    print("🎯 RESULTADO FINAL:")
    print(f"   ⚙️  Pipeline CUDA:      {'✅' if pipeline_ok else '❌'}")
    print(f"   🎥 Integración RTSP:   {'✅' if integration_ok else '❌'}")
    
    if pipeline_ok and integration_ok:
        print("\n🎉 ¡SISTEMA COMPLETAMENTE FUNCIONAL!")
        print("🚀 Puedes proceder con el desarrollo del proyecto")
    elif pipeline_ok:
        print("\n⚠️ Pipeline funciona, pero hay problemas con RTSP")
        print("💡 Desarrolla primero con datos sintéticos")
    else:
        print("\n❌ Aún hay problemas con el pipeline CUDA")
        print("💡 Revisa los logs de error arriba")

if __name__ == "__main__":
    main()
