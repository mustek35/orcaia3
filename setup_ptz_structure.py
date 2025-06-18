#!/usr/bin/env python3
"""
Script de Setup Automático para Sistema PTZ
Crea la estructura completa de carpetas y organiza archivos
"""

import os
import sys
import shutil
import json
from datetime import datetime
from pathlib import Path

def create_folder_structure():
    """Crear estructura completa de carpetas"""
    print("📁 Creando estructura de carpetas PTZ...")
    
    folders = [
        "config",
        "logs", 
        "backups",
        "backups/config_backups",
        "scripts",
        "utils",
        "tests",
        "docs"
    ]
    
    created_folders = []
    for folder in folders:
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            created_folders.append(folder)
            print(f"  ✅ {folder}/")
        except Exception as e:
            print(f"  ❌ Error creando {folder}: {e}")
    
    return created_folders

def create_init_files():
    """Crear archivos __init__.py necesarios"""
    print("\n📝 Creando archivos __init__.py...")
    
    init_files = {
        "core/__init__.py": '''"""
Núcleo del Sistema PTZ
Componentes principales de seguimiento y control
"""

from .ptz_tracking_system import PTZTrackingManager, PTZTrackingThread
from .light_api import LightAPI
from .ptz_integration import PTZSystemIntegration
from .grid_utils import GridUtils
from .detection_ptz_bridge import DetectionPTZBridge

__version__ = "1.0.0"
__all__ = [
    'PTZTrackingManager', 'PTZTrackingThread', 'LightAPI',
    'PTZSystemIntegration', 'GridUtils', 'DetectionPTZBridge'
]
''',
        
        "gui/__init__.py": '''"""
Interfaces Gráficas del Sistema PTZ
Widgets y componentes de UI
"""

from .ptz_config_widget import PTZConfigWidget
from .grilla_widget import GrillaWidget

# Importar MainGUI si existe
try:
    from .components import MainGUI
    __all__ = ['PTZConfigWidget', 'MainGUI', 'GrillaWidget']
except ImportError:
    __all__ = ['PTZConfigWidget', 'GrillaWidget']
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
            
            # Escribir archivo solo si no existe
            if not Path(file_path).exists():
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                created_files.append(file_path)
                print(f"  ✅ {file_path}")
            else:
                print(f"  ℹ️ {file_path} (ya existe)")
                
        except Exception as e:
            print(f"  ❌ Error creando {file_path}: {e}")
    
    return created_files

def organize_ptz_files():
    """Organizar archivos PTZ existentes en estructura correcta"""
    print("\n📂 Organizando archivos PTZ en estructura...")
    
    file_moves = {
        # Archivos de core/
        "ptz_tracking_system.py": "core/ptz_tracking_system.py",
        "light_api.py": "core/light_api.py", 
        "ptz_integration.py": "core/ptz_integration.py",
        "grid_utils.py": "core/grid_utils.py",
        "detection_ptz_bridge.py": "core/detection_ptz_bridge.py",
        
        # Archivos de gui/
        "ptz_config_widget.py": "gui/ptz_config_widget.py",
        
        # Archivos de scripts/
        "integrate_ptz_with_existing_app.py": "scripts/integrate_ptz_with_existing_app.py",
        "main_ptz_integration.py": "scripts/main_ptz_integration.py",
        
        # Archivos de configuración
        "ptz_tracking_config.json": "config/ptz_tracking_config.json",
        "ptz_integration_status.json": "config/ptz_integration_status.json",
    }
    
    moved_files = []
    for source, destination in file_moves.items():
        try:
            if os.path.exists(source):
                # Crear directorio destino si no existe
                dest_dir = os.path.dirname(destination)
                if dest_dir:
                    Path(dest_dir).mkdir(parents=True, exist_ok=True)
                
                # Mover archivo
                shutil.move(source, destination)
                moved_files.append((source, destination))
                print(f"  ✅ {source} → {destination}")
            else:
                print(f"  ⚠️ {source} (no encontrado)")
                
        except Exception as e:
            print(f"  ❌ Error moviendo {source}: {e}")
    
    return moved_files

def create_config_files():
    """Crear archivos de configuración por defecto"""
    print("\n⚙️ Creando archivos de configuración...")
    
    # Configuración principal PTZ
    ptz_config = {
        "ptz_cameras": [],
        "global_settings": {
            "auto_start_tracking": False,
            "log_level": "INFO",
            "max_tracking_distance": 0.3,
            "tracking_timeout": 10.0,
            "grid_dimensions": {
                "rows": 12,
                "columns": 16
            },
            "detection_filters": {
                "min_confidence": 0.3,
                "allowed_object_types": ["person", "vehicle", "boat", "ship"],
                "ignore_small_objects": True,
                "min_object_size": 50
            }
        },
        "version": "1.0",
        "created_date": datetime.now().isoformat()
    }
    
    # Estado de integración
    integration_status = {
        "integration_date": datetime.now().isoformat(),
        "ptz_system_version": "1.0",
        "integrated_components": [
            "ptz_tracking_system",
            "light_api", 
            "ptz_config_widget",
            "detection_ptz_bridge"
        ],
        "integration_status": "setup_completed",
        "folder_structure_created": True,
        "backup_enabled": True
    }
    
    config_files = {
        "config/ptz_tracking_config.json": ptz_config,
        "config/ptz_integration_status.json": integration_status,
        "config/ptz_presets.json": {"presets": [], "version": "1.0"},
        "config/ptz_zones.json": {"zones": [], "version": "1.0"}
    }
    
    created_configs = []
    for file_path, content in config_files.items():
        try:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(content, f, indent=4, ensure_ascii=False)
                created_configs.append(file_path)
                print(f"  ✅ {file_path}")
            else:
                print(f"  ℹ️ {file_path} (ya existe)")
                
        except Exception as e:
            print(f"  ❌ Error creando {file_path}: {e}")
    
    return created_configs

def create_documentation():
    """Crear documentación básica"""
    print("\n📚 Creando documentación...")
    
    readme_ptz = '''# Sistema PTZ Profesional

## 🎯 Descripción
Sistema completo de seguimiento PTZ integrado con detección de objetos YOLO.

## 🚀 Características
- Seguimiento automático basado en detecciones
- Control manual de cámaras PTZ
- Configuración de zonas de seguimiento
- Interfaz gráfica profesional
- API completa para cámaras PTZ
- Logs y estadísticas detalladas

## 📁 Estructura de Archivos
```
core/          # Núcleo del sistema PTZ
gui/           # Interfaces gráficas
config/        # Archivos de configuración
logs/          # Logs del sistema
scripts/       # Scripts de automatización
utils/         # Utilidades auxiliares
tests/         # Pruebas del sistema
docs/          # Documentación
```

## 🔧 Instalación
1. Ejecutar: `python scripts/setup_ptz_structure.py`
2. Configurar cámaras PTZ en `config/ptz_tracking_config.json`
3. Ejecutar aplicación principal

## 📖 Uso
- Panel PTZ: Configuración y control manual
- Seguimiento automático: Se activa con detecciones
- Parada de emergencia: Botón STOP en interfaz

## 🆘 Soporte
Ver `TROUBLESHOOTING.md` para solución de problemas.
'''
    
    installation_guide = '''# Guía de Instalación Sistema PTZ

## 📋 Prerrequisitos
- Python 3.8+
- PyQt6
- requests
- numpy

## 🔧 Instalación Paso a Paso

### 1. Crear Estructura
```bash
python setup_ptz_structure.py
```

### 2. Instalar Dependencias
```bash
pip install PyQt6 requests numpy
```

### 3. Configurar Cámaras PTZ
Editar `config/ptz_tracking_config.json`:
```json
{
  "ptz_cameras": [
    {
      "ip": "192.168.1.100",
      "username": "admin", 
      "password": "password123",
      "tracking_enabled": true
    }
  ]
}
```

### 4. Ejecutar Integración
```bash
python scripts/integrate_ptz_with_existing_app.py
```

### 5. Iniciar Aplicación
```bash
python main_gui.py
```

## ✅ Verificación
- Verificar que aparecen botones PTZ en interfaz
- Probar conexión con cámara PTZ
- Configurar zona de seguimiento
'''
    
    troubleshooting = '''# Solución de Problemas Sistema PTZ

## 🔍 Problemas Comunes

### Error: "No se puede conectar a cámara PTZ"
**Solución:**
1. Verificar IP, usuario y contraseña
2. Comprobar conectividad de red
3. Verificar que cámara soporta LightAPI

### Error: "Imports PTZ no encontrados"
**Solución:**
1. Verificar estructura de carpetas
2. Ejecutar `python setup_ptz_structure.py`
3. Verificar archivos __init__.py

### Seguimiento no funciona
**Solución:**
1. Verificar que cámara está configurada como tipo "PTZ"
2. Habilitar seguimiento en Panel PTZ
3. Configurar zona de seguimiento
4. Verificar detecciones YOLO

### Performance lento
**Solución:**
1. Reducir sensibilidad de seguimiento
2. Configurar zona más pequeña
3. Ajustar FPS de detección

## 📞 Contacto
Para soporte adicional, revisar logs en `logs/ptz_tracking.log`
'''
    
    docs = {
        "docs/README_PTZ.md": readme_ptz,
        "docs/INSTALLATION.md": installation_guide,
        "docs/TROUBLESHOOTING.md": troubleshooting
    }
    
    created_docs = []
    for file_path, content in docs.items():
        try:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                created_docs.append(file_path)
                print(f"  ✅ {file_path}")
            else:
                print(f"  ℹ️ {file_path} (ya existe)")
                
        except Exception as e:
            print(f"  ❌ Error creando {file_path}: {e}")
    
    return created_docs

def create_utility_scripts():
    """Crear scripts de utilidad"""
    print("\n🔧 Creando scripts de utilidad...")
    
    # Script de diagnósticos
    diagnostics_script = '''#!/usr/bin/env python3
"""
Script de Diagnósticos del Sistema PTZ
Verifica configuración y conectividad
"""

import os
import sys
import json
import importlib.util

def check_file_structure():
    """Verificar estructura de archivos"""
    print("📁 Verificando estructura de archivos...")
    
    required_folders = ["core", "gui", "config", "logs", "scripts", "utils", "tests", "docs"]
    required_files = [
        "core/ptz_tracking_system.py",
        "core/light_api.py",
        "gui/ptz_config_widget.py",
        "config/ptz_tracking_config.json"
    ]
    
    for folder in required_folders:
        if os.path.exists(folder):
            print(f"  ✅ {folder}/")
        else:
            print(f"  ❌ {folder}/ (faltante)")
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} (faltante)")

def check_imports():
    """Verificar imports del sistema PTZ"""
    print("\\n📦 Verificando imports...")
    
    try:
        from core.ptz_integration import PTZSystemIntegration
        print("  ✅ PTZSystemIntegration")
    except ImportError as e:
        print(f"  ❌ PTZSystemIntegration: {e}")
    
    try:
        from core.light_api import LightAPI
        print("  ✅ LightAPI")
    except ImportError as e:
        print(f"  ❌ LightAPI: {e}")
    
    try:
        from gui.ptz_config_widget import PTZConfigWidget
        print("  ✅ PTZConfigWidget")
    except ImportError as e:
        print(f"  ❌ PTZConfigWidget: {e}")

def check_configuration():
    """Verificar archivos de configuración"""
    print("\\n⚙️ Verificando configuración...")
    
    config_file = "config/ptz_tracking_config.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            print(f"  ✅ {config_file}")
            print(f"    📹 Cámaras PTZ configuradas: {len(config.get('ptz_cameras', []))}")
            print(f"    🎯 Seguimiento automático: {config.get('global_settings', {}).get('auto_start_tracking', False)}")
            
        except json.JSONDecodeError:
            print(f"  ❌ {config_file} (JSON inválido)")
        except Exception as e:
            print(f"  ❌ {config_file}: {e}")
    else:
        print(f"  ❌ {config_file} (no encontrado)")

if __name__ == "__main__":
    print("🔍 DIAGNÓSTICOS DEL SISTEMA PTZ")
    print("=" * 40)
    
    check_file_structure()
    check_imports()
    check_configuration()
    
    print("\\n✅ Diagnósticos completados")
'''
    
    # Script de mantenimiento
    maintenance_script = '''#!/usr/bin/env python3
"""
Script de Mantenimiento del Sistema PTZ
Limpieza de logs, backups y optimización
"""

import os
import sys
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path

def clean_old_logs():
    """Limpiar logs antiguos"""
    print("🧹 Limpiando logs antiguos...")
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("  ℹ️ Directorio logs no existe")
        return
    
    # Eliminar logs más antiguos de 30 días
    cutoff_date = datetime.now() - timedelta(days=30)
    cleaned_count = 0
    
    for log_file in logs_dir.glob("*.log"):
        try:
            file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
            if file_time < cutoff_date:
                log_file.unlink()
                cleaned_count += 1
                print(f"  🗑️ Eliminado: {log_file.name}")
        except Exception as e:
            print(f"  ❌ Error procesando {log_file.name}: {e}")
    
    print(f"  ✅ {cleaned_count} logs antiguos eliminados")

def clean_old_backups():
    """Limpiar backups antiguos"""
    print("\\n🗂️ Limpiando backups antiguos...")
    
    backups_dir = Path("backups")
    if not backups_dir.exists():
        print("  ℹ️ Directorio backups no existe")
        return
    
    # Mantener solo los 10 backups más recientes
    backup_dirs = sorted([d for d in backups_dir.iterdir() if d.is_dir() and d.name.startswith("backup_original_")])
    
    if len(backup_dirs) > 10:
        for old_backup in backup_dirs[:-10]:
            try:
                shutil.rmtree(old_backup)
                print(f"  🗑️ Eliminado: {old_backup.name}")
            except Exception as e:
                print(f"  ❌ Error eliminando {old_backup.name}: {e}")
        
        print(f"  ✅ {len(backup_dirs) - 10} backups antiguos eliminados")
    else:
        print("  ℹ️ No hay backups antiguos para eliminar")

def optimize_config():
    """Optimizar archivos de configuración"""
    print("\\n⚙️ Optimizando configuración...")
    
    config_file = Path("config/ptz_tracking_config.json")
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Limpiar configuraciones inválidas
            valid_cameras = []
            for camera in config.get("ptz_cameras", []):
                if camera.get("ip") and camera.get("username"):
                    valid_cameras.append(camera)
            
            config["ptz_cameras"] = valid_cameras
            config["last_maintenance"] = datetime.now().isoformat()
            
            # Guardar configuración optimizada
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=4)
            
            print(f"  ✅ Configuración optimizada ({len(valid_cameras)} cámaras válidas)")
            
        except Exception as e:
            print(f"  ❌ Error optimizando configuración: {e}")
    else:
        print("  ℹ️ Archivo de configuración no encontrado")

def generate_maintenance_report():
    """Generar reporte de mantenimiento"""
    print("\\n📊 Generando reporte de mantenimiento...")
    
    report = {
        "maintenance_date": datetime.now().isoformat(),
        "system_status": "healthy",
        "files_cleaned": True,
        "backups_optimized": True,
        "config_optimized": True,
        "next_maintenance": (datetime.now() + timedelta(days=7)).isoformat()
    }
    
    try:
        with open("logs/maintenance_report.json", 'w') as f:
            json.dump(report, f, indent=4)
        
        print("  ✅ Reporte guardado en logs/maintenance_report.json")
        
    except Exception as e:
        print(f"  ❌ Error generando reporte: {e}")

if __name__ == "__main__":
    print("🛠️ MANTENIMIENTO DEL SISTEMA PTZ")
    print("=" * 40)
    
    clean_old_logs()
    clean_old_backups()
    optimize_config()
    generate_maintenance_report()
    
    print("\\n✅ Mantenimiento completado")
'''
    
    scripts = {
        "scripts/ptz_diagnostics.py": diagnostics_script,
        "scripts/ptz_maintenance.py": maintenance_script
    }
    
    created_scripts = []
    for file_path, content in scripts.items():
        try:
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                # Hacer ejecutable en sistemas Unix
                if os.name != 'nt':  # No Windows
                    os.chmod(file_path, 0o755)
                created_scripts.append(file_path)
                print(f"  ✅ {file_path}")
            else:
                print(f"  ℹ️ {file_path} (ya existe)")
                
        except Exception as e:
            print(f"  ❌ Error creando {file_path}: {e}")
    
    return created_scripts

def update_requirements():
    """Actualizar archivo requirements.txt"""
    print("\n📦 Actualizando requirements.txt...")
    
    ptz_requirements = [
        "PyQt6>=6.0.0",
        "requests>=2.25.0", 
        "numpy>=1.20.0",
        "Pillow>=8.0.0"
    ]
    
    requirements_file = "requirements.txt"
    existing_requirements = []
    
    # Leer requirements existentes
    if os.path.exists(requirements_file):
        with open(requirements_file, 'r') as f:
            existing_requirements = [line.strip() for line in f.readlines() if line.strip()]
    
    # Agregar nuevos requirements si no existen
    updated = False
    for req in ptz_requirements:
        package_name = req.split('>=')[0].split('==')[0]
        
        # Verificar si el paquete ya está listado
        package_exists = any(existing_req.startswith(package_name) for existing_req in existing_requirements)
        
        if not package_exists:
            existing_requirements.append(req)
            updated = True
            print(f"  ➕ Agregado: {req}")
    
    # Guardar requirements actualizados
    if updated:
        try:
            with open(requirements_file, 'w') as f:
                f.write('\\n'.join(sorted(existing_requirements)))
                f.write('\\n')
            print(f"  ✅ {requirements_file} actualizado")
        except Exception as e:
            print(f"  ❌ Error actualizando requirements: {e}")
    else:
        print("  ℹ️ requirements.txt ya está actualizado")

def create_gitignore():
    """Crear .gitignore para archivos PTZ"""
    print("\\n📝 Actualizando .gitignore...")
    
    ptz_ignores = [
        "# Sistema PTZ",
        "logs/*.log",
        "backups/",
        "config/ptz_tracking_config.json",
        "*.prof",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        ".pytest_cache/",
        "*.egg-info/"
    ]
    
    gitignore_file = ".gitignore"
    existing_ignores = []
    
    # Leer .gitignore existente
    if os.path.exists(gitignore_file):
        with open(gitignore_file, 'r') as f:
            existing_ignores = [line.strip() for line in f.readlines()]
    
    # Agregar nuevas entradas
    updated = False
    for ignore in ptz_ignores:
        if ignore not in existing_ignores:
            existing_ignores.append(ignore)
            updated = True
    
    # Guardar .gitignore actualizado
    if updated:
        try:
            with open(gitignore_file, 'w') as f:
                f.write('\\n'.join(existing_ignores))
                f.write('\\n')
            print("  ✅ .gitignore actualizado")
        except Exception as e:
            print(f"  ❌ Error actualizando .gitignore: {e}")
    else:
        print("  ℹ️ .gitignore ya está actualizado")

def main():
    """Función principal de setup"""
    print("🚀 SETUP AUTOMÁTICO DEL SISTEMA PTZ")
    print("=" * 50)
    
    try:
        # Paso 1: Crear estructura de carpetas
        created_folders = create_folder_structure()
        
        # Paso 2: Crear archivos __init__.py
        created_inits = create_init_files()
        
        # Paso 3: Organizar archivos PTZ existentes
        moved_files = organize_ptz_files()
        
        # Paso 4: Crear archivos de configuración
        created_configs = create_config_files()
        
        # Paso 5: Crear documentación
        created_docs = create_documentation()
        
        # Paso 6: Crear scripts de utilidad
        created_scripts = create_utility_scripts()
        
        # Paso 7: Actualizar requirements
        update_requirements()
        
        # Paso 8: Actualizar .gitignore
        create_gitignore()
        
        # Resumen final
        print("\\n" + "=" * 50)
        print("🎉 SETUP COMPLETADO EXITOSAMENTE")
        print("=" * 50)
        
        print(f"\\n📊 RESUMEN:")
        print(f"  📁 Carpetas creadas: {len(created_folders)}")
        print(f"  📝 Archivos __init__.py: {len(created_inits)}")
        print(f"  📂 Archivos movidos: {len(moved_files)}")
        print(f"  ⚙️ Configuraciones creadas: {len(created_configs)}")
        print(f"  📚 Documentación creada: {len(created_docs)}")
        print(f"  🔧 Scripts creados: {len(created_scripts)}")
        
        print(f"\\n📋 ESTRUCTURA FINAL:")
        print("  📁 core/          # Núcleo del sistema PTZ")
        print("  📁 gui/           # Interfaces gráficas")
        print("  📁 config/        # Configuraciones")
        print("  📁 logs/          # Logs del sistema")
        print("  📁 scripts/       # Scripts de automatización")
        print("  📁 utils/         # Utilidades auxiliares")
        print("  📁 tests/         # Pruebas del sistema")
        print("  📁 docs/          # Documentación")
        print("  📁 backups/       # Backups automáticos")
        
        print(f"\\n🚀 PRÓXIMOS PASOS:")
        print("  1. Verificar estructura: python scripts/ptz_diagnostics.py")
        print("  2. Integrar con app: python scripts/integrate_ptz_with_existing_app.py")
        print("  3. Configurar cámaras: editar config/ptz_tracking_config.json")
        print("  4. Ejecutar aplicación: python main_gui.py")
        
        print(f"\\n💡 COMANDOS ÚTILES:")
        print("  🔍 Diagnósticos: python scripts/ptz_diagnostics.py")
        print("  🛠️ Mantenimiento: python scripts/ptz_maintenance.py")
        print("  📚 Documentación: cat docs/README_PTZ.md")
        
        return True
        
    except Exception as e:
        print(f"\\n❌ ERROR FATAL EN SETUP: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)