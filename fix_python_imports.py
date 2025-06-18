#!/usr/bin/env python3
"""
Script para Solucionar Imports del Sistema PTZ
Crea y configura correctamente todos los archivos __init__.py
"""

import os
import sys
from pathlib import Path

def create_init_files():
    """Crear todos los archivos __init__.py necesarios"""
    print("📝 Creando archivos __init__.py...")
    
    init_files = {
        "__init__.py": '''# Proyecto PTZ Tracker
__version__ = "1.0.0"
''',
        
        "core/__init__.py": '''"""
Núcleo del Sistema PTZ
Componentes principales de seguimiento y control
"""

# Importaciones principales
try:
    from .ptz_tracking_system import PTZTrackingManager, PTZTrackingThread, DetectionEvent
    from .light_api import LightAPI, PTZDirection, ZoomDirection, PresetInfo
    from .ptz_integration import PTZSystemIntegration, PTZControlInterface
    from .grid_utils import GridUtils, GridCell, GridZone
    from .detection_ptz_bridge import DetectionPTZBridge, detection_ptz_bridge
    
    __all__ = [
        'PTZTrackingManager', 'PTZTrackingThread', 'DetectionEvent',
        'LightAPI', 'PTZDirection', 'ZoomDirection', 'PresetInfo',
        'PTZSystemIntegration', 'PTZControlInterface', 
        'GridUtils', 'GridCell', 'GridZone',
        'DetectionPTZBridge', 'detection_ptz_bridge'
    ]
    
except ImportError as e:
    print(f"⚠️ Warning: Some PTZ modules not available: {e}")
    __all__ = []

__version__ = "1.0.0"
''',
        
        "gui/__init__.py": '''"""
Interfaces Gráficas del Sistema PTZ
Widgets y componentes de UI
"""

# Importaciones PTZ
try:
    from .ptz_config_widget import PTZConfigWidget, PTZControlThread
    PTZ_GUI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: PTZ GUI components not available: {e}")
    PTZ_GUI_AVAILABLE = False

# Importaciones existentes
try:
    from .grilla_widget import GrillaWidget
    GRILLA_AVAILABLE = True
except ImportError:
    GRILLA_AVAILABLE = False

try:
    from .components import MainGUI
    MAIN_GUI_AVAILABLE = True
except ImportError:
    MAIN_GUI_AVAILABLE = False

# Construir __all__ dinámicamente
__all__ = []
if PTZ_GUI_AVAILABLE:
    __all__.extend(['PTZConfigWidget', 'PTZControlThread'])
if GRILLA_AVAILABLE:
    __all__.append('GrillaWidget')
if MAIN_GUI_AVAILABLE:
    __all__.append('MainGUI')

__version__ = "1.0.0"
''',
        
        "ui/__init__.py": '''"""
Interfaces de Usuario
Diálogos y componentes de UI existentes
"""

__version__ = "1.0.0"
''',
        
        "scripts/__init__.py": '''"""
Scripts del Sistema PTZ
Scripts de automatización y mantenimiento
"""

__version__ = "1.0.0"
''',
        
        "utils/__init__.py": '''"""
Utilidades del Sistema PTZ
Funciones auxiliares y helpers
"""

__version__ = "1.0.0"
''',
        
        "tests/__init__.py": '''"""
Tests del Sistema PTZ
Pruebas unitarias y de integración
"""

__version__ = "1.0.0"
'''
    }
    
    created_files = []
    for file_path, content in init_files.items():
        try:
            # Crear directorio padre si no existe
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Escribir archivo
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            created_files.append(file_path)
            print(f"  ✅ {file_path}")
                
        except Exception as e:
            print(f"  ❌ Error creando {file_path}: {e}")
    
    return created_files

def fix_python_path():
    """Agregar directorio actual al PYTHONPATH"""
    print("\n🔧 Configurando PYTHONPATH...")
    
    current_dir = os.getcwd()
    
    # Agregar al sys.path
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
        print(f"  ✅ Agregado al sys.path: {current_dir}")
    
    # Crear archivo .pth para persistencia (opcional)
    try:
        import site
        site_packages = site.getsitepackages()[0] if site.getsitepackages() else None
        
        if site_packages:
            pth_file = os.path.join(site_packages, "ptz_tracker.pth")
            with open(pth_file, 'w') as f:
                f.write(current_dir)
            print(f"  ✅ Archivo .pth creado: {pth_file}")
        
    except Exception as e:
        print(f"  ⚠️ No se pudo crear archivo .pth: {e}")

def test_imports():
    """Probar imports después de la corrección"""
    print("\n🧪 Probando imports corregidos...")
    
    # Agregar directorio actual al path
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    tests = [
        ("core.ptz_integration", "PTZSystemIntegration"),
        ("core.light_api", "LightAPI"), 
        ("gui.ptz_config_widget", "PTZConfigWidget"),
        ("core.grid_utils", "GridUtils"),
        ("core.detection_ptz_bridge", "DetectionPTZBridge")
    ]
    
    success_count = 0
    for module_name, class_name in tests:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print(f"  ✅ {module_name}.{class_name}")
            success_count += 1
        except ImportError as e:
            print(f"  ❌ {module_name}.{class_name}: {e}")
        except AttributeError as e:
            print(f"  ❌ {module_name}.{class_name}: Clase no encontrada")
        except Exception as e:
            print(f"  ❌ {module_name}.{class_name}: {e}")
    
    print(f"\n📊 Resultado: {success_count}/{len(tests)} imports exitosos")
    return success_count == len(tests)

def create_setup_py():
    """Crear setup.py para instalación local"""
    print("\n📦 Creando setup.py...")
    
    setup_content = '''#!/usr/bin/env python3
"""
Setup script para PTZ Tracker
Permite instalación local del paquete
"""

from setuptools import setup, find_packages

setup(
    name="ptz_tracker",
    version="1.0.0",
    description="Sistema PTZ Profesional con Seguimiento Automático",
    packages=find_packages(),
    install_requires=[
        "PyQt6>=6.0.0",
        "requests>=2.25.0",
        "numpy>=1.20.0",
        "Pillow>=8.0.0"
    ],
    python_requires=">=3.8",
    author="PTZ Tracker Team",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
'''
    
    try:
        with open("setup.py", 'w', encoding='utf-8') as f:
            f.write(setup_content)
        print("  ✅ setup.py creado")
        return True
    except Exception as e:
        print(f"  ❌ Error creando setup.py: {e}")
        return False

def install_package_locally():
    """Instalar paquete localmente en modo desarrollo"""
    print("\n📦 Instalando paquete localmente...")
    
    try:
        import subprocess
        
        # Instalar en modo desarrollo
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "."
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("  ✅ Paquete instalado en modo desarrollo")
            print("  💡 Ahora los imports deberían funcionar desde cualquier directorio")
            return True
        else:
            print(f"  ❌ Error instalando paquete: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"  ❌ Error en instalación: {e}")
        return False

def create_run_script():
    """Crear script de ejecución que maneja el PATH"""
    print("\n🚀 Creando script de ejecución...")
    
    run_script_content = '''#!/usr/bin/env python3
"""
Script de ejecución para PTZ Tracker
Configura automáticamente el PATH de Python
"""

import os
import sys

# Agregar directorio actual al PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Importar y ejecutar aplicación principal
try:
    from main_window import MainGUI
    from PyQt6.QtWidgets import QApplication
    
    def main():
        app = QApplication(sys.argv)
        gui = MainGUI()
        gui.show()
        return app.exec()
    
    if __name__ == "__main__":
        sys.exit(main())
        
except ImportError as e:
    print(f"❌ Error importando aplicación principal: {e}")
    print("💡 Asegúrese de que todos los archivos PTZ estén en su lugar")
    sys.exit(1)
'''
    
    try:
        with open("run_ptz_tracker.py", 'w', encoding='utf-8') as f:
            f.write(run_script_content)
        
        # Hacer ejecutable en sistemas Unix
        if os.name != 'nt':
            os.chmod("run_ptz_tracker.py", 0o755)
        
        print("  ✅ run_ptz_tracker.py creado")
        return True
        
    except Exception as e:
        print(f"  ❌ Error creando script: {e}")
        return False

def main():
    """Función principal de corrección"""
    print("🔧 CORRECCIÓN DE IMPORTS DEL SISTEMA PTZ")
    print("=" * 50)
    
    try:
        # Paso 1: Crear archivos __init__.py
        created_files = create_init_files()
        print(f"✅ {len(created_files)} archivos __init__.py creados")
        
        # Paso 2: Configurar PYTHONPATH
        fix_python_path()
        
        # Paso 3: Crear setup.py
        create_setup_py()
        
        # Paso 4: Crear script de ejecución
        create_run_script()
        
        # Paso 5: Probar imports
        imports_ok = test_imports()
        
        print("\n" + "=" * 50)
        print("🎉 CORRECCIÓN COMPLETADA")
        print("=" * 50)
        
        if imports_ok:
            print("\n✅ TODOS LOS IMPORTS FUNCIONAN CORRECTAMENTE")
            print("\n🚀 OPCIONES PARA EJECUTAR:")
            print("1. python run_ptz_tracker.py")
            print("2. python main_window.py")
            print("3. python scripts/ptz_diagnostics.py")
        else:
            print("\n⚠️ ALGUNOS IMPORTS FALLAN")
            print("\n🔧 OPCIONES DE SOLUCIÓN:")
            print("1. Instalar en modo desarrollo:")
            print("   pip install -e .")
            print("2. Usar script de ejecución:")
            print("   python run_ptz_tracker.py")
            print("3. Verificar que todos los archivos PTZ estén presentes")
        
        print("\n💡 COMANDOS ÚTILES:")
        print("🔍 Diagnóstico: python scripts/ptz_diagnostics.py")
        print("🛠️ Re-ejecutar fix: python fix_python_imports.py")
        print("📦 Instalar paquete: pip install -e .")
        
        return imports_ok
        
    except Exception as e:
        print(f"\n❌ ERROR FATAL: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)