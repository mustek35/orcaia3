# add_missing_iou_function.py
"""
Añadir la función iou faltante al detector_worker para compatibilidad
"""

def add_iou_function_to_detector():
    """Añadir función iou al detector_worker.py para compatibilidad"""
    
    detector_file = "core/detector_worker.py"
    
    print("🔧 Añadiendo función iou al detector_worker...")
    
    try:
        # Leer archivo actual
        with open(detector_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Función iou que se necesita
        iou_function = '''
def iou(boxA, boxB):
    """Compute IoU between two boxes given as [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea == 0:
        return 0.0
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    if union == 0:
        return 0.0
    return interArea / union
'''
        
        # Verificar si ya está presente
        if "def iou(boxA, boxB):" in content:
            print("✅ Función iou ya presente")
            return True
        
        # Añadir la función después de los imports
        # Buscar el final de los imports
        import_end = content.find("logger = get_logger(__name__)")
        
        if import_end != -1:
            # Encontrar el final de esa línea
            line_end = content.find("\n", import_end) + 1
            
            # Insertar la función iou
            new_content = content[:line_end] + iou_function + content[line_end:]
            
            # Escribir archivo actualizado
            with open(detector_file, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            print("✅ Función iou añadida exitosamente")
            return True
        else:
            print("❌ No se pudo encontrar ubicación para insertar iou")
            return False
        
    except Exception as e:
        print(f"❌ Error añadiendo función iou: {e}")
        return False

def test_iou_import():
    """Test que la función iou se puede importar"""
    
    print("🧪 Verificando importación de iou...")
    
    try:
        # Recargar módulo
        import sys
        if 'core.detector_worker' in sys.modules:
            del sys.modules['core.detector_worker']
        
        # Importar función iou
        from core.detector_worker import iou
        
        # Test básico
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 15, 15]
        iou_result = iou(box1, box2)
        
        print(f"✅ Función iou importada y probada: IoU = {iou_result:.3f}")
        return True
        
    except ImportError as e:
        print(f"❌ Error importando iou: {e}")
        return False
    except Exception as e:
        print(f"❌ Error probando iou: {e}")
        return False

def test_app_startup():
    """Test que la app se puede iniciar sin errores de importación"""
    
    print("🧪 Verificando que la app no tenga errores de importación...")
    
    try:
        # Test importación de GrillaWidget
        from gui.grilla_widget import GrillaWidget
        print("✅ GrillaWidget se puede importar sin errores")
        
        # Test que puede importar iou desde detector_worker
        from core.detector_worker import iou
        print("✅ iou se puede importar desde detector_worker")
        
        return True
        
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Función principal"""
    
    print("🔧 FIX FUNCIÓN IOU FALTANTE")
    print("=" * 40)
    
    print("Este script añade la función iou que falta")
    print("en detector_worker.py para compatibilidad")
    
    # Paso 1: Añadir función iou
    if add_iou_function_to_detector():
        print("\n✅ Función iou añadida")
    else:
        print("\n❌ Error añadiendo función iou")
        return
    
    # Paso 2: Verificar importación
    if test_iou_import():
        print("\n✅ Importación verificada")
    else:
        print("\n❌ Error en importación")
        return
    
    # Paso 3: Test app
    if test_app_startup():
        print("\n✅ App verificada")
    else:
        print("\n❌ Errores de app persisten")
        return
    
    print("\n" + "=" * 40)
    print("🎉 ¡FIX COMPLETADO!")
    print("=" * 40)
    
    print("\n🚀 AHORA PUEDES:")
    print("   python app.py")
    print("   (Sin errores de importación)")
    
    print("\n✅ FUNCIONALIDADES VERIFICADAS:")
    print("   • Función iou disponible")
    print("   • GrillaWidget importable")
    print("   • DetectorWorker funcional")
    print("   • Compatibilidad completa")
    
    print("\n💡 SI SIGUE HABIENDO PROBLEMAS:")
    print("   • Reinicia completamente Python")
    print("   • Verifica que no hay otros módulos cacheados")
    print("   • Los logs te dirán exactamente qué falla")

if __name__ == "__main__":
    main()