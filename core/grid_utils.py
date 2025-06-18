"""
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
