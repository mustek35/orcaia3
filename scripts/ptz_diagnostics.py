#!/usr/bin/env python3
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
    print("\n📦 Verificando imports...")
    
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
    print("\n⚙️ Verificando configuración...")
    
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
    
    print("\n✅ Diagnósticos completados")
