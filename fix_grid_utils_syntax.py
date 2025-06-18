#!/usr/bin/env python3
"""
Corrección manual específica para grid_utils.py
"""

import os
import shutil
from datetime import datetime

def show_problematic_lines():
    """Mostrar líneas problemáticas para diagnóstico"""
    print("🔍 Analizando líneas problemáticas...")
    
    file_path = "core/grid_utils.py"
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Mostrar líneas alrededor de los errores
        problem_lines = [658, 662, 699, 705]
        
        for line_num in problem_lines:
            if line_num <= len(lines):
                print(f"\n📍 Línea {line_num}:")
                
                # Mostrar contexto (2 líneas antes y después)
                start = max(0, line_num - 3)
                end = min(len(lines), line_num + 2)
                
                for i in range(start, end):
                    marker = ">>>" if i == line_num - 1 else "   "
                    print(f"{marker} {i+1:3d}: {lines[i].rstrip()}")
        
        return lines
        
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return None

def fix_grid_utils_manual():
    """Corrección manual completa de grid_utils.py"""
    print("🔧 Aplicando corrección manual completa...")
    
    file_path = "core/grid_utils.py"
    
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return False
    
    # Crear backup
    backup_path = f"{file_path}.manual_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(file_path, backup_path)
        print(f"📦 Backup creado: {backup_path}")
    except Exception as e:
        print(f"⚠️ No se pudo crear backup: {e}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Correcciones específicas línea por línea
        fixes = [
            # Corregir patrones problemáticos
            ("for zone_id_str, zone_data in zones_data.items():", 
             "            for zone_id_str, zone_data in zones_data.items():"),
            
            # Corregir bucles mal formateados
            ("                    for zone_id, zone in self.zones.items():", 
             "            for zone_id, zone in self.zones.items():"),
            
            # Corregir indentación problemática
            ("            zones_data = config.get(\"zones\", {})", 
             "            zones_data = config.get(\"zones\", {})"),
            
            # Asegurar que las líneas terminen correctamente
            ("zones_", "zones_data"),
        ]
        
        fixed_content = content
        applied_fixes = []
        
        for old_pattern, new_pattern in fixes:
            if old_pattern in fixed_content:
                fixed_content = fixed_content.replace(old_pattern, new_pattern)
                applied_fixes.append(f"'{old_pattern}' → '{new_pattern}'")
        
        # Corrección específica para el problema de bucle incompleto
        # Buscar líneas que terminan con 'zones_' y corregirlas
        lines = fixed_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().endswith('zones_'):
                if 'for' in line and 'in' in line:
                    # Es un bucle for incompleto
                    lines[i] = line.replace('zones_', 'zones_data.items():')
                    applied_fixes.append(f"Línea {i+1}: Bucle for corregido")
                else:
                    # Es una variable incompleta
                    lines[i] = line.replace('zones_', 'zones_data')
                    applied_fixes.append(f"Línea {i+1}: Variable corregida")
        
        fixed_content = '\n'.join(lines)
        
        # Escribir contenido corregido
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        if applied_fixes:
            print("\n✅ Correcciones aplicadas:")
            for fix in applied_fixes:
                print(f"  • {fix}")
        else:
            print("ℹ️ No se encontraron patrones específicos para corregir")
        
        return True
        
    except Exception as e:
        print(f"❌ Error aplicando correcciones: {e}")
        return False

def create_clean_grid_utils():
    """Crear una versión limpia de grid_utils.py desde cero"""
    print("🆕 Creando versión limpia de grid_utils.py...")
    
    clean_content = '''"""
Utilidades para manejo de grilla de seguimiento PTZ
Funciones auxiliares para cálculos de posición y mapeo de coordenadas
"""

import math
import json
import time
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class GridCell:
    """Representa una celda de la grilla"""
    row: int
    col: int
    x_start: int
    y_start: int
    width: int
    height: int
    center_x: int
    center_y: int
    is_active: bool = False
    detection_count: int = 0

@dataclass
class GridZone:
    """Representa una zona de la grilla"""
    id: int
    name: str
    cells: List[Tuple[int, int]]
    priority: int = 1
    enabled: bool = True
    tracking_sensitivity: float = 0.005

class GridUtils:
    """Utilidades para manejo de grilla de seguimiento"""
    
    def __init__(self, grid_rows: int = 12, grid_cols: int = 16):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.grid_cells: Dict[Tuple[int, int], GridCell] = {}
        self.zones: Dict[int, GridZone] = {}
        
    def initialize_grid(self, frame_width: int, frame_height: int):
        """Inicializar la grilla con las dimensiones del frame"""
        try:
            cell_width = frame_width // self.grid_cols
            cell_height = frame_height // self.grid_rows
            
            self.grid_cells.clear()
            
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    x_start = col * cell_width
                    y_start = row * cell_height
                    
                    width = cell_width
                    height = cell_height
                    
                    if col == self.grid_cols - 1:
                        width = frame_width - x_start
                    if row == self.grid_rows - 1:
                        height = frame_height - y_start
                    
                    center_x = x_start + width // 2
                    center_y = y_start + height // 2
                    
                    cell = GridCell(
                        row=row,
                        col=col,
                        x_start=x_start,
                        y_start=y_start,
                        width=width,
                        height=height,
                        center_x=center_x,
                        center_y=center_y
                    )
                    
                    self.grid_cells[(row, col)] = cell
            
            logger.info(f"Grilla inicializada: {self.grid_rows}x{self.grid_cols}")
                       
        except Exception as e:
            logger.error(f"Error inicializando grilla: {e}")
    
    def pixel_to_grid_cell(self, x: int, y: int, frame_width: int, frame_height: int) -> Tuple[int, int]:
        """Convertir coordenadas de píxel a celda de grilla"""
        try:
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            
            col = min(int(x * self.grid_cols / frame_width), self.grid_cols - 1)
            row = min(int(y * self.grid_rows / frame_height), self.grid_rows - 1)
            
            return (row, col)
            
        except Exception as e:
            logger.error(f"Error convirtiendo píxel a celda: {e}")
            return (0, 0)
    
    def grid_cell_to_pixel(self, row: int, col: int, frame_width: int, frame_height: int) -> Tuple[int, int]:
        """Convertir celda de grilla a coordenadas centrales de píxel"""
        try:
            row = max(0, min(row, self.grid_rows - 1))
            col = max(0, min(col, self.grid_cols - 1))
            
            cell_width = frame_width / self.grid_cols
            cell_height = frame_height / self.grid_rows
            
            center_x = int((col + 0.5) * cell_width)
            center_y = int((row + 0.5) * cell_height)
            
            return (center_x, center_y)
            
        except Exception as e:
            logger.error(f"Error convirtiendo celda a píxel: {e}")
            return (frame_width // 2, frame_height // 2)
    
    def get_active_cells(self) -> List[Tuple[int, int]]:
        """Obtener lista de celdas activas"""
        try:
            active_cells = []
            for (row, col), cell in self.grid_cells.items():
                if cell.is_active:
                    active_cells.append((row, col))
            return active_cells
        except Exception as e:
            logger.error(f"Error obteniendo celdas activas: {e}")
            return []
    
    def activate_cells(self, cells: List[Tuple[int, int]]):
        """Activar celdas específicas para seguimiento"""
        try:
            for row, col in cells:
                if (row, col) in self.grid_cells:
                    self.grid_cells[(row, col)].is_active = True
            logger.info(f"Activadas {len(cells)} celdas para seguimiento")
        except Exception as e:
            logger.error(f"Error activando celdas: {e}")
    
    def increment_detection_count(self, row: int, col: int):
        """Incrementar contador de detecciones para una celda"""
        try:
            if (row, col) in self.grid_cells:
                self.grid_cells[(row, col)].detection_count += 1
        except Exception as e:
            logger.error(f"Error incrementando contador: {e}")
    
    def get_detection_heatmap(self) -> Dict[Tuple[int, int], int]:
        """Obtener mapa de calor de detecciones"""
        try:
            heatmap = {}
            for (row, col), cell in self.grid_cells.items():
                if cell.detection_count > 0:
                    heatmap[(row, col)] = cell.detection_count
            return heatmap
        except Exception as e:
            logger.error(f"Error creando mapa de calor: {e}")
            return {}
    
    def save_configuration(self, filename: str) -> bool:
        """Guardar configuración de grilla en archivo"""
        try:
            config = {
                "grid_dimensions": {
                    "rows": self.grid_rows,
                    "cols": self.grid_cols
                },
                "active_cells": self.get_active_cells(),
                "zones": {},
                "detection_counts": {}
            }
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=4)
            
            logger.info(f"Configuración guardada en {filename}")
            return True
        except Exception as e:
            logger.error(f"Error guardando configuración: {e}")
            return False
    
    def load_configuration(self, filename: str) -> bool:
        """Cargar configuración de grilla desde archivo"""
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            
            dimensions = config.get("grid_dimensions", {})
            self.grid_rows = dimensions.get("rows", self.grid_rows)
            self.grid_cols = dimensions.get("cols", self.grid_cols)
            
            active_cells = config.get("active_cells", [])
            self.activate_cells([tuple(cell) for cell in active_cells])
            
            logger.info(f"Configuración cargada desde {filename}")
            return True
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            return False

# Instancia global por defecto
default_grid_utils = GridUtils()

def get_default_grid_utils() -> GridUtils:
    """Obtener instancia por defecto de GridUtils"""
    return default_grid_utils
'''
    
    try:
        with open("core/grid_utils.py", 'w', encoding='utf-8') as f:
            f.write(clean_content)
        
        print("✅ Versión limpia de grid_utils.py creada")
        return True
        
    except Exception as e:
        print(f"❌ Error creando versión limpia: {e}")
        return False

def validate_syntax_and_test():
    """Validar sintaxis y probar imports"""
    print("\n🧪 Validando sintaxis y probando imports...")
    
    try:
        # Validar sintaxis
        with open("core/grid_utils.py", 'r', encoding='utf-8') as f:
            content = f.read()
        
        compile(content, "core/grid_utils.py", 'exec')
        print("✅ Sintaxis válida")
        
        # Probar import
        import sys
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Remover módulo del cache si existe
        if 'core.grid_utils' in sys.modules:
            del sys.modules['core.grid_utils']
        
        from core.grid_utils import GridUtils
        print("✅ Import exitoso")
        
        # Probar funcionalidad básica
        grid = GridUtils()
        grid.initialize_grid(640, 480)
        print("✅ Funcionalidad básica OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Función principal"""
    print("🔧 CORRECCIÓN MANUAL DE GRID_UTILS.PY")
    print("=" * 45)
    
    # Mostrar líneas problemáticas
    lines = show_problematic_lines()
    
    if lines:
        print("\n🤔 ¿Qué quieres hacer?")
        print("1. Aplicar corrección automática")
        print("2. Crear versión limpia desde cero")
        print("3. Solo validar sintaxis actual")
        
        try:
            choice = input("\nElige opción (1/2/3): ").strip()
        except:
            choice = "2"  # Default a versión limpia
        
        if choice == "1":
            if fix_grid_utils_manual():
                print("✅ Corrección automática aplicada")
            else:
                print("❌ Corrección automática falló")
                choice = "2"  # Fallback a versión limpia
        
        if choice == "2":
            if create_clean_grid_utils():
                print("✅ Versión limpia creada")
            else:
                print("❌ Error creando versión limpia")
                return False
    
    # Validar resultado final
    if validate_syntax_and_test():
        print("\n🎉 ¡GRID_UTILS.PY CORREGIDO EXITOSAMENTE!")
        print("\n🚀 Ahora ejecuta:")
        print("  python scripts/ptz_diagnostics.py")
        return True
    else:
        print("\n❌ Aún hay problemas con grid_utils.py")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)