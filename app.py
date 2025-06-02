import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainGUI # Asegúrate que esta es la GUI principal correcta

# Cualquier otro import que sea REALMENTE necesario para este archivo mínimo
# En este caso, no parece haber otros necesarios después de la limpieza.

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MainGUI() # Asumiendo que MainGUI es la clase de ui.main_window
    gui.show()
    sys.exit(app.exec())
