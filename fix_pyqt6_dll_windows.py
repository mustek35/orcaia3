#!/usr/bin/env python3
"""
Script para solucionar problemas de DLL en PyQt6 en Windows
Mantiene PyQt6 que es necesario para la analítica
"""

import subprocess
import sys
import os
import platform

def run_command(command, description):
    """Ejecutar comando y mostrar resultado"""
    try:
        print(f"🔧 {description}...")
        print(f"   Comando: {command}")
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ {description} - Éxito")
            if result.stdout.strip():
                # Solo mostrar líneas importantes
                lines = result.stdout.strip().split('\n')
                important_lines = [line for line in lines if any(keyword in line.lower() 
                    for keyword in ['successfully', 'installed', 'upgraded', 'pyqt6'])]
                if important_lines:
                    for line in important_lines[:3]:  # Solo primeras 3 líneas importantes
                        print(f"   {line}")
        else:
            print(f"⚠️ {description} - Advertencia")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error ejecutando {command}: {e}")
        return False

def check_system_info():
    """Verificar información del sistema"""
    print("🖥️ Información del sistema:")
    print(f"   OS: {platform.system()} {platform.release()}")
    print(f"   Arquitectura: {platform.architecture()[0]}")
    print(f"   Python: {sys.version}")
    print(f"   Python ejecutable: {sys.executable}")
    
    # Verificar si es 64-bit
    is_64bit = platform.architecture()[0] == '64bit'
    print(f"   Sistema 64-bit: {'✅ Sí' if is_64bit else '❌ No'}")
    
    return is_64bit

def check_visual_cpp_redistributable():
    """Verificar Visual C++ Redistributable"""
    print("\n🔍 Verificando Visual C++ Redistributable...")
    
    try:
        # Verificar si están instalados los redistributables
        result = subprocess.run('reg query "HKLM\\SOFTWARE\\Microsoft\\VisualStudio\\14.0\\VC\\Runtimes\\x64" /v Version', 
                              shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Visual C++ Redistributable 2015-2022 x64 encontrado")
            return True
        else:
            print("❌ Visual C++ Redistributable 2015-2022 x64 no encontrado")
            print("💡 Esto puede causar errores de DLL en PyQt6")
            return False
            
    except Exception as e:
        print(f"⚠️ No se pudo verificar VC++ Redistributable: {e}")
        return False

def download_vc_redistributable():
    """Proporcionar enlace para descargar VC++ Redistributable"""
    print("\n📥 DESCARGA REQUERIDA: Visual C++ Redistributable")
    print("=" * 50)
    print("Para solucionar el error de DLL de PyQt6, necesitas instalar:")
    print("")
    print("🔗 Microsoft Visual C++ Redistributable 2015-2022 x64:")
    print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("")
    print("📋 Pasos:")
    print("1. Descarga el archivo desde el enlace de arriba")
    print("2. Ejecuta como administrador")
    print("3. Instala y reinicia el sistema")
    print("4. Vuelve a ejecutar este script")
    print("")
    
    return False

def clean_and_reinstall_pyqt6():
    """Limpiar e instalar PyQt6 correctamente"""
    print("\n🧹 Limpiando e instalando PyQt6...")
    
    # Comandos de limpieza
    cleanup_commands = [
        ("pip uninstall PyQt6 PyQt6-Qt6 PyQt6-sip PyQt6-tools -y", "Limpiando instalación anterior"),
        ("pip cache purge", "Limpiando cache de pip"),
        ("pip install --upgrade pip", "Actualizando pip")
    ]
    
    for command, description in cleanup_commands:
        run_command(command, description)
    
    # Instalar PyQt6 con versiones específicas que funcionan bien
    install_commands = [
        # Instalar en orden específico para evitar conflictos
        ("pip install PyQt6-sip==13.5.2", "Instalando PyQt6-sip"),
        ("pip install PyQt6-Qt6==6.5.2", "Instalando PyQt6-Qt6"),
        ("pip install PyQt6==6.5.2", "Instalando PyQt6"),
    ]
    
    success = True
    for command, description in install_commands:
        if not run_command(command, description):
            print(f"⚠️ {description} falló, intentando versión alternativa...")
            # Intentar sin versión específica
            alt_command = command.split('==')[0]
            if not run_command(alt_command, f"{description} (sin versión específica)"):
                success = False
    
    return success

def test_pyqt6_import():
    """Probar import de PyQt6 con diagnóstico detallado"""
    print("\n🧪 Probando import de PyQt6...")
    
    try:
        print("   Importando PyQt6.QtCore...")
        from PyQt6.QtCore import Qt, QCoreApplication
        print("   ✅ PyQt6.QtCore importado")
        
        print("   Importando PyQt6.QtWidgets...")
        from PyQt6.QtWidgets import QApplication, QWidget
        print("   ✅ PyQt6.QtWidgets importado")
        
        print("   Importando PyQt6.QtGui...")
        from PyQt6.QtGui import QGuiApplication
        print("   ✅ PyQt6.QtGui importado")
        
        print("   Creando QApplication de prueba...")
        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication([])
        print("   ✅ QCoreApplication creado exitosamente")
        
        print("✅ PyQt6 funciona correctamente")
        return True
        
    except ImportError as e:
        print(f"❌ Error de import: {e}")
        
        # Diagnóstico adicional
        error_str = str(e).lower()
        if "dll load failed" in error_str:
            print("\n🔍 DIAGNÓSTICO: Error de DLL")
            print("   Causa probable: Falta Visual C++ Redistributable")
            print("   Solución: Instalar VC++ Redistributable 2015-2022")
        elif "no module named" in error_str:
            print("\n🔍 DIAGNÓSTICO: Módulo no encontrado")
            print("   Causa probable: PyQt6 no instalado correctamente")
            print("   Solución: Reinstalar PyQt6")
        
        return False
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return False

def create_minimal_test():
    """Crear test mínimo de PyQt6"""
    print("\n📝 Creando test mínimo de PyQt6...")
    
    test_content = '''#!/usr/bin/env python3
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
    
    label = QLabel("🎉 ¡PyQt6 funciona correctamente!\\n\\n🎯 El sistema PTZ está listo")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(label)
    
    print("   ✅ Ventana creada")
    
    print("5. Mostrando ventana...")
    window.show()
    print("   ✅ Ventana mostrada")
    
    print("\\n🎉 ¡ÉXITO! PyQt6 funciona perfectamente")
    print("💡 Cierra la ventana para continuar")
    
    return app.exec()
    
except ImportError as e:
    print(f"❌ Error de import: {e}")
    if "DLL load failed" in str(e):
        print("\\n💡 SOLUCIÓN:")
        print("   1. Instalar Visual C++ Redistributable 2015-2022 x64")
        print("   2. Reiniciar el sistema")
        print("   3. Volver a probar")
    return 1
    
except Exception as e:
    print(f"❌ Error inesperado: {e}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    try:
        with open("test_pyqt6_minimal.py", 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        print("✅ test_pyqt6_minimal.py creado")
        return True
        
    except Exception as e:
        print(f"❌ Error creando test: {e}")
        return False

def provide_solutions():
    """Proporcionar soluciones específicas"""
    print("\n🛠️ SOLUCIONES PARA ERROR DE DLL PYQT6")
    print("=" * 45)
    
    print("\n🥇 SOLUCIÓN PRINCIPAL (Recomendada):")
    print("   1. Descargar e instalar Visual C++ Redistributable:")
    print("      https://aka.ms/vs/17/release/vc_redist.x64.exe")
    print("   2. Ejecutar como administrador")
    print("   3. Reiniciar el sistema")
    print("   4. Ejecutar: python test_pyqt6_minimal.py")
    
    print("\n🥈 SOLUCIÓN ALTERNATIVA 1:")
    print("   pip uninstall PyQt6 -y")
    print("   pip install PyQt6==6.5.2")
    
    print("\n🥉 SOLUCIÓN ALTERNATIVA 2:")
    print("   conda install pyqt=6")
    
    print("\n🔧 SOLUCIÓN ALTERNATIVA 3 (si tienes MSYS2):")
    print("   pacman -S mingw-w64-x86_64-python-pyqt6")
    
    print("\n⚠️ IMPORTANTE:")
    print("   - PyQt6 requiere Visual C++ Runtime Libraries")
    print("   - El error de DLL es típico cuando faltan estas librerías")
    print("   - La analítica necesita PyQt6, por eso no podemos usar PyQt5")

def main():
    """Función principal"""
    print("🔧 SOLUCIONADOR DE DLL PYQT6 PARA WINDOWS")
    print("=" * 45)
    
    # Verificar sistema
    is_64bit = check_system_info()
    if not is_64bit:
        print("⚠️ Sistema de 32-bit detectado. PyQt6 funciona mejor en 64-bit.")
    
    # Verificar VC++ Redistributable
    has_vcredist = check_visual_cpp_redistributable()
    
    if not has_vcredist:
        download_vc_redistributable()
        provide_solutions()
        return False
    
    # Probar import actual
    if test_pyqt6_import():
        print("\n🎉 ¡PyQt6 YA FUNCIONA!")
        create_minimal_test()
        print("\n🚀 Ejecuta: python test_pyqt6_minimal.py")
        return True
    
    # Si no funciona, intentar reinstalar
    print("\n🔄 PyQt6 no funciona, intentando reinstalar...")
    
    if clean_and_reinstall_pyqt6():
        print("✅ Reinstalación completada")
        
        # Probar de nuevo
        if test_pyqt6_import():
            print("\n🎉 ¡PROBLEMA SOLUCIONADO!")
            create_minimal_test()
            print("\n🚀 Ejecuta: python test_pyqt6_minimal.py")
            return True
        else:
            print("\n❌ Aún hay problemas con PyQt6")
    
    # Si todo falla, mostrar soluciones
    provide_solutions()
    create_minimal_test()
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("1. Instalar Visual C++ Redistributable (enlace arriba)")
    print("2. Reiniciar el sistema")
    print("3. Ejecutar: python test_pyqt6_minimal.py")
    print("4. Si funciona: python app.py")
    
    return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)