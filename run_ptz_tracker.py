#!/usr/bin/env python3
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
