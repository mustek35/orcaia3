#!/usr/bin/env python3
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
    print("\n🗂️ Limpiando backups antiguos...")
    
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
    print("\n⚙️ Optimizando configuración...")
    
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
    print("\n📊 Generando reporte de mantenimiento...")
    
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
    
    print("\n✅ Mantenimiento completado")
