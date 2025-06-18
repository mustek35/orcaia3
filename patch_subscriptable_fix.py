# patch_subscriptable_fix.py
"""
Tercer patch para solucionar 'int' object is not subscriptable
"""
import os
import shutil
import re
from datetime import datetime

def find_problematic_lines():
    """Encontrar líneas problemáticas en cuda_pipeline_processor.py"""
    
    processor_file = "core/cuda_pipeline_processor.py"
    
    if not os.path.exists(processor_file):
        return []
    
    problematic_patterns = [
        r"input_size\[\d+\]",  # input_size[0], input_size[1], etc.
        r"self\.model_config\[['\"]\w+['\"]]\[\d+\]",  # config['key'][0]
        r"target_size\[\d+\]",  # target_size[0]
        r"imgsz\[\d+\]",  # imgsz[0]
    ]
    
    problems = []
    
    try:
        with open(processor_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line_num, line in enumerate(lines, 1):
            for pattern in problematic_patterns:
                if re.search(pattern, line):
                    problems.append({
                        'line_num': line_num,
                        'line': line.strip(),
                        'pattern': pattern
                    })
    except:
        pass
    
    return problems

def apply_subscriptable_patch():
    """Aplicar patch para 'int' object is not subscriptable"""
    
    processor_file = "core/cuda_pipeline_processor.py"
    
    if not os.path.exists(processor_file):
        print(f"❌ Archivo {processor_file} no encontrado")
        return False
    
    print("🔍 Analizando problemas de subscriptable...")
    problems = find_problematic_lines()
    
    if problems:
        print(f"📋 Encontrados {len(problems)} patrones problemáticos:")
        for p in problems[:5]:  # Mostrar solo los primeros 5
            print(f"   Línea {p['line_num']}: {p['line']}")
    
    print(f"🔧 Aplicando patch para subscriptable...")
    
    try:
        # Leer archivo
        with open(processor_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Crear backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = f"{processor_file}.backup_subscript_{timestamp}"
        shutil.copy2(processor_file, backup_file)
        print(f"💾 Backup creado: {backup_file}")
        
        # Lista de patches específicos para subscriptable
        patches = [
            # Patch 1: Fix input_size[0] y input_size[1]
            {
                'find': r"input_size\[0\]",
                'replace': "(input_size if isinstance(input_size, (int, float)) else input_size[0])",
                'desc': "Fix input_size[0]"
            },
            {
                'find': r"input_size\[1\]", 
                'replace': "(input_size if isinstance(input_size, (int, float)) else input_size[1] if len(input_size) > 1 else input_size[0])",
                'desc': "Fix input_size[1]"
            },
            # Patch 2: Fix target_size accesos
            {
                'find': r"target_size\[0\]",
                'replace': "(target_size if isinstance(target_size, (int, float)) else target_size[0])",
                'desc': "Fix target_size[0]"
            },
            {
                'find': r"target_size\[1\]",
                'replace': "(target_size if isinstance(target_size, (int, float)) else target_size[1] if len(target_size) > 1 else target_size[0])",
                'desc': "Fix target_size[1]"
            },
            # Patch 3: Fix imgsz accesos
            {
                'find': r"imgsz\[0\]",
                'replace': "(imgsz if isinstance(imgsz, (int, float)) else imgsz[0])",
                'desc': "Fix imgsz[0]"
            },
            {
                'find': r"imgsz\[1\]",
                'replace': "(imgsz if isinstance(imgsz, (int, float)) else imgsz[1] if len(imgsz) > 1 else imgsz[0])",
                'desc': "Fix imgsz[1]"
            }
        ]
        
        patches_applied = 0
        
        # Aplicar patches con regex
        for patch in patches:
            if re.search(patch['find'], content):
                content = re.sub(patch['find'], patch['replace'], content)
                patches_applied += 1
                print(f"✅ {patch['desc']}")
        
        # Patch general para preprocessing function
        preprocessing_old = """def _preprocess_frame(self, frame):
        try:
            # Obtener tamaño de entrada de forma segura
            input_size = self.model_config.get('input_size', 
                        self.model_config.get('imgsz', 640))
            
            if isinstance(input_size, (list, tuple)):
                target_size = input_size[:2]  # (height, width)
            else:
                target_size = (input_size, input_size)  # cuadrado"""
        
        preprocessing_new = """def _preprocess_frame(self, frame):
        try:
            # Obtener tamaño de entrada de forma segura
            input_size = self.model_config.get('input_size', 
                        self.model_config.get('imgsz', 640))
            
            # Asegurar que input_size sea manejable
            if isinstance(input_size, (list, tuple)):
                if len(input_size) >= 2:
                    target_height, target_width = input_size[0], input_size[1]
                else:
                    target_height = target_width = input_size[0]
            else:
                target_height = target_width = int(input_size)
            
            # Garantizar valores enteros positivos
            target_height = max(32, int(target_height))
            target_width = max(32, int(target_width))"""
        
        if preprocessing_old in content:
            content = content.replace(preprocessing_old, preprocessing_new)
            print("✅ Preprocessing function reescrita")
            patches_applied += 1
        
        # Patch para warmup function
        warmup_pattern = r"def _warmup_model\(self\):"
        if re.search(warmup_pattern, content):
            warmup_replacement = """def _warmup_model(self):
        try:
            # Obtener dimensiones de forma segura
            input_size = self.model_config.get('input_size', 640)
            
            if isinstance(input_size, (list, tuple)):
                height = int(input_size[0]) if len(input_size) > 0 else 640
                width = int(input_size[1]) if len(input_size) > 1 else height
            else:
                height = width = int(input_size)
            
            # Crear tensor de warmup
            warmup_tensor = torch.randn(1, 3, height, width, device=self.device)
            
            for i in range(self.model_config.get('warmup_iterations', 3)):
                with torch.no_grad():
                    _ = self.model(warmup_tensor)
            
            logger.info("Warmup del modelo completado")
            
        except Exception as e:
            logger.warning(f"Error en warmup: {e}")"""
            
            # Encontrar la función warmup actual y reemplazarla
            warmup_start = content.find("def _warmup_model(self):")
            if warmup_start != -1:
                # Encontrar el final de la función (próxima función def o final del archivo)
                next_def = content.find("\n    def ", warmup_start + 1)
                if next_def == -1:
                    next_def = len(content)
                
                # Reemplazar toda la función
                content = content[:warmup_start] + warmup_replacement + content[next_def:]
                print("✅ Warmup function reescrita")
                patches_applied += 1
        
        # Escribir archivo modificado
        if patches_applied > 0:
            with open(processor_file, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Tercer patch aplicado ({patches_applied} modificaciones)")
            return True
        else:
            print("⚠️ No se encontraron patrones para patchear")
            return True
            
    except Exception as e:
        print(f"❌ Error aplicando patch: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_debug_test():
    """Crear test de debug específico"""
    
    test_content = '''# test_debug_subscriptable.py
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
        print(f"\\n📋 Caso {i+1}: {input_size} (tipo: {type(input_size)})")
        
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
    print("\\n🧪 Test acceso model_config...")
    
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
    print("\\n🧪 Test creación tensores...")
    
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
    print("\\n🧪 Test YOLO simple...")
    
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
    
    print("\\n💡 DIAGNÓSTICO:")
    print("   Si todos los tests pasan, el problema está en")
    print("   el código específico del CUDAPipelineProcessor")

if __name__ == "__main__":
    main()
'''
    
    with open('test_debug_subscriptable.py', 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    print("✅ test_debug_subscriptable.py creado")

def main():
    print("🔧 TERCER PATCH - FIX SUBSCRIPTABLE")
    print("=" * 50)
    
    print("Este patch solucionará:")
    print("   ❌ Error 'int' object is not subscriptable")
    print("   ❌ Problemas con acceso a índices en enteros")
    
    # Encontrar problemas primero
    problems = find_problematic_lines()
    if problems:
        print(f"\\n📋 Problemas encontrados: {len(problems)}")
    
    if apply_subscriptable_patch():
        print("\\n✅ Tercer patch aplicado exitosamente")
        
        create_debug_test()
        
        print("\\n🧪 EJECUTA LOS TESTS:")
        print("   1. python test_debug_subscriptable.py")
        print("   2. python test_final_complete.py")
        
        print("\\n💡 Si persisten errores:")
        print("   - El archivo puede tener problemas más profundos")
        print("   - Considera restaurar desde backup y reescribir")
    else:
        print("\\n❌ Error aplicando tercer patch")

if __name__ == "__main__":
    main()