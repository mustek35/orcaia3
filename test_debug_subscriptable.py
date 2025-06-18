# test_debug_subscriptable.py
"""
Test específico para debuggear el error subscriptable
"""
import numpy as np
import torch

def test_preprocessing_isolated():
    """Test aislado de preprocessing"""
    print("🧪 Test preprocessing aislado...")
    
    # Simular diferentes tipos de input_size
    test_cases = [
        640,           # int
        [640],         # lista con 1 elemento
        [640, 640],    # lista con 2 elementos
        (480, 640),    # tupla
        "640",         # string (caso edge)
    ]
    
    for i, input_size in enumerate(test_cases):
        print(f"\n📋 Caso {i+1}: {input_size} (tipo: {type(input_size)})")
        
        try:
            # Lógica similar al preprocessing
            if isinstance(input_size, (list, tuple)):
                if len(input_size) >= 2:
                    target_height, target_width = input_size[0], input_size[1]
                else:
                    target_height = target_width = input_size[0]
            else:
                target_height = target_width = int(input_size)
            
            # Garantizar valores enteros positivos
            target_height = max(32, int(target_height))
            target_width = max(32, int(target_width))
            
            print(f"   ✅ Resultado: {target_height}x{target_width}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

def test_model_config_access():
    """Test acceso a configuración del modelo"""
    print("\n🧪 Test acceso model_config...")
    
    # Configuración de prueba
    test_config = {
        'input_size': 640,
        'imgsz': 416,
        'batch_size': 1,
        'confidence_threshold': 0.5
    }
    
    # Tests de acceso seguro
    access_tests = [
        ('input_size', 640),
        ('imgsz', 416), 
        ('missing_key', None),
        ('batch_size', 1)
    ]
    
    for key, expected in access_tests:
        # Acceso seguro con .get()
        value = test_config.get(key, 'DEFAULT')
        print(f"   {key}: {value} ({'✅' if value != 'DEFAULT' else '⚠️'})")

def test_tensor_creation():
    """Test creación de tensores"""
    print("\n🧪 Test creación tensores...")
    
    try:
        # Test básico CUDA
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            tensor = torch.randn(1, 3, 640, 640, device=device)
            print(f"   ✅ Tensor CUDA: {tensor.shape}")
        else:
            print("   ⚠️ CUDA no disponible")
            
        # Test CPU fallback
        tensor_cpu = torch.randn(1, 3, 480, 640)
        print(f"   ✅ Tensor CPU: {tensor_cpu.shape}")
        
    except Exception as e:
        print(f"   ❌ Error tensores: {e}")

def test_simple_yolo():
    """Test YOLO simple sin procesador complejo"""
    print("\n🧪 Test YOLO simple...")
    
    try:
        from ultralytics import YOLO
        
        model = YOLO('yolov8n.pt')
        
        # Imagen de prueba
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Inferencia simple
        results = model(test_image, verbose=False, imgsz=640)
        
        if results:
            print(f"   ✅ YOLO simple funciona: {len(results)} resultados")
            
            if hasattr(results[0], 'boxes') and results[0].boxes is not None:
                print(f"   📦 Detecciones: {len(results[0].boxes)}")
            else:
                print("   📦 Sin detecciones")
        else:
            print("   ⚠️ YOLO sin resultados")
            
    except Exception as e:
        print(f"   ❌ Error YOLO simple: {e}")

def main():
    print("🔍 DEBUG SUBSCRIPTABLE ERROR")
    print("=" * 50)
    
    test_preprocessing_isolated()
    test_model_config_access()
    test_tensor_creation()
    test_simple_yolo()
    
    print("\n💡 DIAGNÓSTICO:")
    print("   Si todos los tests pasan, el problema está en")
    print("   el código específico del CUDAPipelineProcessor")

if __name__ == "__main__":
    main()
