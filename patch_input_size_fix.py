# patch_input_size_fix.py
"""
Segundo patch para solucionar el error 'input_size' en CUDA Processor
"""
import os
import shutil
from datetime import datetime

def apply_input_size_patch():
    """Aplicar patch para el error 'input_size'"""
    
    processor_file = "core/cuda_pipeline_processor.py"
    
    if not os.path.exists(processor_file):
        print(f"❌ Archivo {processor_file} no encontrado")
        return False
    
    print(f"🔧 Aplicando patch para 'input_size'...")
    
    try:
        # Leer archivo
        with open(processor_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Crear backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{processor_file}.backup_inputsize_{timestamp}"
        shutil.copy2(processor_file, backup_file)
        print(f"💾 Backup creado: {backup_file}")
        
        # Patch 1: Fix en __init__ para añadir input_size
        init_patch_original = """def __init__(self, processor_id, model_config, device='cuda:0'):
        # Aplicar configuraciones por defecto para evitar KeyErrors
        default_config = {
            'compile_model': False,
            'optimize_for_inference': True,
            'warmup_iterations': 3,
            'half_precision': True,
            'verbose': False
        }
        self.model_config = {**default_config, **model_config}"""
        
        init_patch_new = """def __init__(self, processor_id, model_config, device='cuda:0'):
        # Aplicar configuraciones por defecto para evitar KeyErrors
        default_config = {
            'compile_model': False,
            'optimize_for_inference': True,
            'warmup_iterations': 3,
            'half_precision': True,
            'verbose': False,
            'input_size': 640,  # Tamaño de entrada por defecto
            'imgsz': 640,       # Alias para input_size
            'max_det': 300,     # Detecciones máximas
            'iou_threshold': 0.45,  # Umbral IoU
            'classes': None,    # Todas las clases
            'agnostic_nms': False,  # NMS agnóstico de clase
        }
        self.model_config = {**default_config, **model_config}"""
        
        if init_patch_original in content:
            content = content.replace(init_patch_original, init_patch_new)
            print("✅ Patch 1: Configuración por defecto expandida")
        
        # Patch 2: Fix acceso directo a input_size
        patches_direct_access = [
            {
                'find': "self.model_config['input_size']",
                'replace': "self.model_config.get('input_size', 640)",
                'desc': "Fix acceso directo input_size"
            },
            {
                'find': "self.model_config['imgsz']", 
                'replace': "self.model_config.get('imgsz', self.model_config.get('input_size', 640))",
                'desc': "Fix acceso directo imgsz"
            },
            {
                'find': "self.model_config['max_det']",
                'replace': "self.model_config.get('max_det', 300)",
                'desc': "Fix acceso directo max_det"
            },
            {
                'find': "self.model_config['iou_threshold']",
                'replace': "self.model_config.get('iou_threshold', 0.45)",
                'desc': "Fix acceso directo iou_threshold"
            },
            {
                'find': "self.model_config['classes']",
                'replace': "self.model_config.get('classes', None)",
                'desc': "Fix acceso directo classes"
            },
            {
                'find': "self.model_config['agnostic_nms']",
                'replace': "self.model_config.get('agnostic_nms', False)",
                'desc': "Fix acceso directo agnostic_nms"
            }
        ]
        
        patches_applied = 0
        for patch in patches_direct_access:
            if patch['find'] in content:
                content = content.replace(patch['find'], patch['replace'])
                patches_applied += 1
                print(f"✅ {patch['desc']}")
        
        # Patch 3: Fix en preprocessing específico
        preprocessing_patch_find = """def _preprocess_frame(self, frame):
        """
        
        preprocessing_patch_replace = """def _preprocess_frame(self, frame):
        try:
            # Obtener tamaño de entrada de forma segura
            input_size = self.model_config.get('input_size', 
                        self.model_config.get('imgsz', 640))
            
            if isinstance(input_size, (list, tuple)):
                target_size = input_size[:2]  # (height, width)
            else:
                target_size = (input_size, input_size)  # cuadrado
        """
        
        if preprocessing_patch_find in content and preprocessing_patch_replace not in content:
            content = content.replace(preprocessing_patch_find, preprocessing_patch_replace)
            print("✅ Patch preprocessing function")
            patches_applied += 1
        
        # Escribir archivo modificado
        if patches_applied > 0:
            with open(processor_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Segundo patch aplicado ({patches_applied} modificaciones)")
            return True
        else:
            print("⚠️ No se encontraron patrones para patchear")
            return True
            
    except Exception as e:
        print(f"❌ Error aplicando patch: {e}")
        return False

def create_complete_test():
    """Crear test final completo"""
    
    test_content = '''# test_final_complete.py
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
        print("\\n📊 RESULTADOS:")
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
    print("\\n🎥 TEST INTEGRACIÓN RTSP + CUDA")
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
        
        print(f"\\n✅ Integración RTSP+CUDA: {results_count} frames procesados")
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
    
    print("\\n" + "=" * 60)
    print("🎯 RESULTADO FINAL:")
    print(f"   ⚙️  Pipeline CUDA:      {'✅' if pipeline_ok else '❌'}")
    print(f"   🎥 Integración RTSP:   {'✅' if integration_ok else '❌'}")
    
    if pipeline_ok and integration_ok:
        print("\\n🎉 ¡SISTEMA COMPLETAMENTE FUNCIONAL!")
        print("🚀 Puedes proceder con el desarrollo del proyecto")
    elif pipeline_ok:
        print("\\n⚠️ Pipeline funciona, pero hay problemas con RTSP")
        print("💡 Desarrolla primero con datos sintéticos")
    else:
        print("\\n❌ Aún hay problemas con el pipeline CUDA")
        print("💡 Revisa los logs de error arriba")

if __name__ == "__main__":
    main()
'''
    
    with open('test_final_complete.py', 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ test_final_complete.py creado")

def main():
    print("🔧 SEGUNDO PATCH - FIX INPUT_SIZE")
    print("=" * 50)
    
    print("Este patch solucionará:")
    print("   ❌ Error 'input_size' en preprocessing")
    print("   ❌ Configuraciones faltantes en CUDA processor")
    
    if apply_input_size_patch():
        print("\n✅ Segundo patch aplicado exitosamente")
        
        create_complete_test()
        
        print("\n🧪 EJECUTA EL TEST FINAL:")
        print("   python test_final_complete.py")
        
        print("\n💡 Si aún hay errores:")
        print("   1. Reinicia Python completamente")
        print("   2. Ejecuta el test final")
        print("   3. Revisa los logs detallados")
    else:
        print("\n❌ Error aplicando segundo patch")

if __name__ == "__main__":
    main()