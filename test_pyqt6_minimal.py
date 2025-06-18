#!/usr/bin/env python3
"""
Test mínimo de PyQt6 para verificar funcionamiento
"""

import sys
import os

print("🧪 Test mínimo de PyQt6")
print("=" * 30)

try:
    print("1. Importando PyQt6.QtCore...")
    from PyQt6.QtCore import Qt, QCoreApplication
    print("   ✅ Éxito")
    
    print("2. Importando PyQt6.QtWidgets...")
    from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
    print("   ✅ Éxito")
    
    print("3. Creando aplicación...")
    app = QApplication(sys.argv)
    print("   ✅ QApplication creado")
    
    print("4. Creando ventana...")
    window = QMainWindow()
    window.setWindowTitle("✅ PyQt6 Funciona!")
    window.setGeometry(300, 300, 400, 200)
    
    central_widget = QWidget()
    window.setCentralWidget(central_widget)
    
    layout = QVBoxLayout()
    central_widget.setLayout(layout)
    
    label = QLabel("🎉 ¡PyQt6 funciona correctamente!\n\n🎯 El sistema PTZ está listo")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    
    print("   ✅ Ventana creada")
    
    print("5. Mostrando ventana...")
    window.show()
    print("   ✅ Ventana mostrada")
    
    print("\n🎉 ¡ÉXITO! PyQt6 funciona perfectamente")
    print("💡 Cierra la ventana para continuar")
    
    return app.exec()
    
except ImportError as e:
    print(f"❌ Error de import: {e}")
    if "DLL load failed" in str(e):
        print("\n💡 SOLUCIÓN:")
        print("   1. Instalar Visual C++ Redistributable 2015-2022 x64")
        print("   2. Reiniciar el sistema")
        print("   3. Volver a probar")
    return 1
    
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
