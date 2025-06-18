# test_standalone_complete.py
"""
Test completamente independiente que no depende de los módulos con errores
"""
import time
import numpy as np
import cv2
import torch
import sys
import os
from pathlib import Path

def check_ffmpeg():
    """Verificar si FFmpeg está disponible"""
    print("🔍 Verificando FFmpeg...")
    
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ {version}")
            return True
        else:
            print("❌ FFmpeg no funciona")
            return False
    except FileNotFoundError:
        print("❌ FFmpeg no está instalado")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_opencv_backends():
    """Test todos los backends de OpenCV disponibles"""
    print("🧪 Testing backends OpenCV...")
    
    # Lista de backends a probar
    backends_to_test = [
        ('CAP_FFMPEG', getattr(cv2, 'CAP_FFMPEG', None)),
        ('CAP_GSTREAMER', getattr(cv2, 'CAP_GSTREAMER', None)),
        ('CAP_DSHOW', getattr(cv2, 'CAP_DSHOW', None)),
        ('CAP_MSMF', getattr(cv2, 'CAP_MSMF', None)),
    ]
    
    available_backends = [(name, backend) for name, backend in backends_to_test if backend is not None]
    print(f"📋 Backends disponibles: {[name for name, _ in available_backends]}")
    
    return available_backends

def test_rtsp_connection():
    """Test directo de conexión RTSP"""
    print("🧪 Test conexión RTSP...")
    
    rtsp_url = "rtsp://root:%40Remoto753524@19.10.10.132:554/media/video1"
    rtsp_tcp = rtsp_url + "?tcp"
    
    backends = test_opencv_backends()
    
    for backend_name, backend_id in backends:
        print(f"\n🔧 Probando {backend_name}...")
        
        cap = cv2.VideoCapture()
        
        # Configuraciones importantes
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Probar URL normal
        print(f"   📡 Intentando: {rtsp_url}")
        success = cap.open(rtsp_url, backend_id)
        
        if not success:
            print(f"   📡 Intentando TCP: {rtsp_tcp}")
            success = cap.open(rtsp_tcp, backend_id)
        
        if success:
            print(f"   ✅ {backend_name}: Conexión exitosa")
            
            # Intentar leer frames
            frames_ok = 0
            for i in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames_ok += 1
                    if i == 0:
                        print(f"   📐 Frame: {frame.shape}")
                else:
                    break
            
            print(f"   📊 Frames: {frames_ok}/5")
            cap.release()
            
            if frames_ok > 0:
                return backend_name, frames_ok
        else:
            print(f"   ❌ {backend_name}: Fallo")
        
        cap.release()
    
    return None, 0

def test_cuda_basic():
    """Test básico de CUDA"""
    print("🧪 Test CUDA básico...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name}")
        print(f"📊 Memoria: {memory:.1f} GB")
        
        # Test tensor básico
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print(f"✅ Operación CUDA exitosa: {y.shape}")
            return True
        except Exception as e:
            print(f"❌ Error operación CUDA: {e}")
            return False
    else:
        print("❌ CUDA no disponible")
        return False

def test_yolo_basic():
    """Test básico de YOLO sin usar el procesador problemático"""
    print("🧪 Test YOLO básico...")
    
    try:
        from ultralytics import YOLO
        
        # Verificar si existe el modelo
        model_paths = ['yolov8n.pt', 'models/yolov8n.pt', '../models/yolov8n.pt']
        model_path = None
        
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("⚠️ Modelo yolov8n.pt no encontrado, descargando...")
            model_path = 'yolov8n.pt'
        
        print(f"📦 Cargando modelo: {model_path}")
        model = YOLO(model_path)
        
        # Test con imagen sintética
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        print("🔍 Ejecutando detección...")
        results = model(test_image, verbose=False)
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            detections = len(boxes) if boxes is not None else 0
            print(f"✅ YOLO funcionando: {detections} detecciones")
            return True
        else:
            print("⚠️ YOLO ejecutó pero sin resultados")
            return True
    
    except Exception as e:
        print(f"❌ Error YOLO: {e}")
        return False

def test_connectivity():
    """Test conectividad básica"""
    print("🧪 Test conectividad...")
    
    import socket
    
    ip = "19.10.10.132"
    port = 554
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((ip, port))
        sock.close()
        
        if result == 0:
            print(f"✅ {ip}:{port} accesible")
            return True
        else:
            print(f"❌ {ip}:{port} no accesible")
            return False
    except Exception as e:
        print(f"❌ Error conectividad: {e}")
        return False

def patch_cuda_processor():
    """Intentar patchear el archivo cuda_pipeline_processor.py"""
    print("🔧 Intentando patchear cuda_pipeline_processor.py...")
    
    processor_file = "core/cuda_pipeline_processor.py"
    
    if not os.path.exists(processor_file):
        print(f"❌ Archivo {processor_file} no encontrado")
        return False
    
    try:
        # Leer el archivo
        with open(processor_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar la línea problemática
        problematic_line = "if self.model_config['compile_model'] and hasattr(torch, 'compile'):"
        
        if problematic_line in content:
            # Reemplazar con versión segura
            safe_line = "if self.model_config.get('compile_model', False) and hasattr(torch, 'compile'):"
            content = content.replace(problematic_line, safe_line)
            
            # Crear backup
            backup_file = processor_file + ".backup"
            if not os.path.exists(backup_file):
                with open(backup_file, 'w', encoding='utf-8') as f:
                    f.write(content.replace(safe_line, problematic_line))  # backup original
            
            # Escribir archivo patcheado
            with open(processor_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"✅ Archivo patcheado: {processor_file}")
            print(f"💾 Backup creado: {backup_file}")
            return True
        else:
            print("⚠️ Línea problemática no encontrada (posiblemente ya patcheado)")
            return True
    
    except Exception as e:
        print(f"❌ Error patcheando: {e}")
        return False

def test_patched_cuda_processor():
    """Test del procesador CUDA después del patch"""
    print("🧪 Test CUDA Processor patcheado...")
    
    try:
        from core.cuda_pipeline_processor import cuda_pipeline_manager
        
        # Configuración completa
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
            'verbose': False
        }
        
        processor = cuda_pipeline_manager.create_processor('test_patched', config)
        processor.load_model()
        processor.start_processing()
        
        # Test rápido
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        results_count = 0
        
        def on_results(detections, metadata):
            nonlocal results_count
            results_count += 1
            print(f"🎯 Resultado: {len(detections)} detecciones")
        
        processor.results_ready.connect(on_results)
        
        # Enviar algunos frames
        for i in range(3):
            processor.add_frame(test_frame, {'frame_id': i})
            time.sleep(0.1)
        
        time.sleep(1)  # Esperar procesamiento
        
        processor.stop_processing()
        cuda_pipeline_manager.stop_all_processors()
        
        print(f"✅ Procesador funcionando: {results_count} resultados")
        return results_count > 0
        
    except Exception as e:
        print(f"❌ Error procesador: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 TEST STANDALONE COMPLETO")
    print("=" * 60)
    
    results = {}
    
    # 1. Verificar FFmpeg
    print("\n1️⃣ FFmpeg:")
    results['ffmpeg'] = check_ffmpeg()
    
    # 2. Test conectividad
    print("\n2️⃣ Conectividad:")
    results['connectivity'] = test_connectivity()
    
    # 3. Test RTSP
    print("\n3️⃣ RTSP:")
    backend, frames = test_rtsp_connection()
    results['rtsp'] = frames > 0
    results['rtsp_backend'] = backend
    
    # 4. Test CUDA básico
    print("\n4️⃣ CUDA:")
    results['cuda'] = test_cuda_basic()
    
    # 5. Test YOLO básico
    print("\n5️⃣ YOLO:")
    results['yolo'] = test_yolo_basic()
    
    # 6. Patchear procesador
    print("\n6️⃣ Patch CUDA Processor:")
    results['patch'] = patch_cuda_processor()
    
    # 7. Test procesador patcheado
    if results['patch']:
        print("\n7️⃣ Test Procesador Patcheado:")
        results['processor'] = test_patched_cuda_processor()
    else:
        results['processor'] = False
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN COMPLETO:")
    print(f"   🎬 FFmpeg:           {'✅' if results['ffmpeg'] else '❌'}")
    print(f"   🌐 Conectividad:     {'✅' if results['connectivity'] else '❌'}")
    print(f"   📡 RTSP:             {'✅' if results['rtsp'] else '❌'}")
    if results['rtsp_backend']:
        print(f"   📺 Backend exitoso:  {results['rtsp_backend']}")
    print(f"   🎯 CUDA:             {'✅' if results['cuda'] else '❌'}")
    print(f"   🤖 YOLO:             {'✅' if results['yolo'] else '❌'}")
    print(f"   🔧 Patch:            {'✅' if results['patch'] else '❌'}")
    print(f"   ⚙️  Procesador:       {'✅' if results['processor'] else '❌'}")
    
    # Análisis
    critical_working = results['cuda'] and results['yolo']
    rtsp_working = results['ffmpeg'] and results['connectivity'] and results['rtsp']
    
    print("\n🎯 ANÁLISIS:")
    if critical_working and rtsp_working and results['processor']:
        print("   🎉 ¡TODO FUNCIONANDO! Sistema listo para desarrollo")
    elif critical_working and results['processor']:
        print("   ⚠️ CUDA/YOLO OK, pero problemas con RTSP")
        print("   💡 Puedes desarrollar con datos sintéticos")
    elif rtsp_working:
        print("   ⚠️ RTSP OK, pero problemas con IA")
        print("   💡 Revisa instalación de PyTorch/CUDA")
    else:
        print("   ❌ Problemas críticos detectados")
    
    print("\n💡 PRÓXIMOS PASOS:")
    if not results['ffmpeg']:
        print("   • Instala FFmpeg: https://ffmpeg.org/download.html")
    if not results['connectivity']:
        print("   • Verifica conexión de red a la cámara")
    if not results['cuda']:
        print("   • Reinstala PyTorch con CUDA")
    if results['patch'] and not results['processor']:
        print("   • Reinicia Python y vuelve a probar")
    
    if critical_working and rtsp_working and results['processor']:
        print("   🚀 ¡Puedes continuar con el desarrollo del proyecto!")

if __name__ == "__main__":
    main()