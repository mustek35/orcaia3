#!/usr/bin/env python3
"""
Script para corregir problemas de inicialización en MainGUI
"""

import os
import shutil
from datetime import datetime

def fix_mainwindow_initialization():
    """Corregir el orden de inicialización en main_window.py"""
    print("🔧 Corrigiendo orden de inicialización en main_window.py...")
    
    # Buscar el archivo main_window.py en diferentes ubicaciones
    possible_paths = [
        "main_window.py",
        "ui/main_window.py", 
        "gui/main_window.py"
    ]
    
    main_window_path = None
    for path in possible_paths:
        if os.path.exists(path):
            main_window_path = path
            break
    
    if not main_window_path:
        print("❌ No se encontró archivo main_window.py")
        return False
    
    print(f"📁 Usando archivo: {main_window_path}")
    
    # Crear backup
    backup_path = f"{main_window_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    try:
        shutil.copy2(main_window_path, backup_path)
        print(f"📦 Backup creado: {backup_path}")
    except Exception as e:
        print(f"⚠️ No se pudo crear backup: {e}")
    
    try:
        with open(main_window_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Corrección 1: Mover inicialización PTZ después de setup_ui
        if "self.initialize_ptz_system()" in content and "self.setup_ui()" in content:
            # Buscar y reemplazar el orden
            lines = content.split('\n')
            new_lines = []
            ptz_init_line = None
            
            for line in lines:
                if "self.initialize_ptz_system()" in line:
                    ptz_init_line = line
                    continue  # Saltar esta línea por ahora
                elif "self.setup_ui()" in line:
                    new_lines.append(line)
                    if ptz_init_line:
                        new_lines.append(ptz_init_line)  # Agregar después de setup_ui
                        ptz_init_line = None
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
            print("✅ Orden de inicialización corregido")
        
        # Corrección 2: Agregar método append_debug seguro
        safe_append_debug = '''
    def append_debug(self, message: str):
        """Método seguro para agregar mensajes de debug"""
        try:
            # Filtrar mensajes de spam
            if any(substr in message for substr in ["hevc @", "VPS 0", "undecodable NALU", "Frame procesado"]):
                return
            
            # Verificar si debug_console existe
            if hasattr(self, 'debug_console') and self.debug_console:
                self.debug_console.append(message)
            else:
                # Fallback a print si no existe debug_console
                print(f"[DEBUG] {message}")
        except Exception as e:
            print(f"[DEBUG Error] {message} (Error: {e})")
'''
        
        # Reemplazar el método append_debug existente o agregarlo
        if "def append_debug(self, message: str):" in content:
            # Encontrar y reemplazar el método existente
            lines = content.split('\n')
            new_lines = []
            in_append_debug = False
            indent_level = 0
            
            for line in lines:
                if "def append_debug(self, message: str):" in line:
                    in_append_debug = True
                    indent_level = len(line) - len(line.lstrip())
                    # Agregar método corregido
                    method_lines = safe_append_debug.strip().split('\n')
                    for method_line in method_lines:
                        if method_line.strip():
                            new_lines.append(' ' * indent_level + method_line.lstrip())
                        else:
                            new_lines.append('')
                elif in_append_debug:
                    # Verificar si seguimos dentro del método
                    if line.strip() and len(line) - len(line.lstrip()) <= indent_level and not line.lstrip().startswith('#'):
                        in_append_debug = False
                        new_lines.append(line)
                    # Si estamos dentro del método, saltarlo
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
            print("✅ Método append_debug corregido")
        else:
            # Agregar método si no existe
            # Buscar el final de la clase MainGUI
            class_end_pattern = "class MainGUI"
            if class_end_pattern in content:
                # Agregar antes del final de la clase
                insertion_point = content.rfind('\n    def closeEvent(')
                if insertion_point == -1:
                    insertion_point = content.rfind('\n    def ')
                
                if insertion_point != -1:
                    content = content[:insertion_point] + safe_append_debug + content[insertion_point:]
                    print("✅ Método append_debug agregado")
        
        # Corrección 3: Hacer inicialización PTZ más segura
        safe_ptz_init = '''
    def initialize_ptz_system(self):
        """Inicializar sistema PTZ integrado de manera segura"""
        try:
            print("🚀 Inicializando sistema PTZ...")
            
            # Crear integración PTZ
            self.ptz_system = PTZSystemIntegration(main_app=self)
            self.ptz_control_interface = PTZControlInterface(self.ptz_system)
            
            # Configurar hooks de detección automática
            setup_ptz_integration_hooks()
            
            # Registrar callback para envío de detecciones
            self.ptz_system.register_detection_callback(self.on_detection_for_ptz)
            
            print("✅ Sistema PTZ inicializado correctamente")
            self.append_debug("🎯 Sistema PTZ listo para seguimiento automático")
            
        except Exception as e:
            print(f"❌ Error inicializando sistema PTZ: {e}")
            self.append_debug(f"⚠️ Sistema PTZ no disponible: {e}")
            self.ptz_system = None
            self.ptz_control_interface = None
'''
        
        # Reemplazar método initialize_ptz_system
        if "def initialize_ptz_system(self):" in content:
            lines = content.split('\n')
            new_lines = []
            in_method = False
            indent_level = 0
            
            for line in lines:
                if "def initialize_ptz_system(self):" in line:
                    in_method = True
                    indent_level = len(line) - len(line.lstrip())
                    # Agregar método corregido
                    method_lines = safe_ptz_init.strip().split('\n')
                    for method_line in method_lines:
                        if method_line.strip():
                            new_lines.append(' ' * indent_level + method_line.lstrip())
                        else:
                            new_lines.append('')
                elif in_method:
                    # Verificar si seguimos dentro del método
                    if line.strip() and len(line) - len(line.lstrip()) <= indent_level and not line.lstrip().startswith('#'):
                        in_method = False
                        new_lines.append(line)
                    # Si estamos dentro del método, saltarlo
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
            print("✅ Método initialize_ptz_system corregido")
        
        # Escribir archivo corregido
        with open(main_window_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Archivo {main_window_path} corregido")
        return True
        
    except Exception as e:
        print(f"❌ Error corrigiendo archivo: {e}")
        return False

def create_simple_app_py():
    """Crear un app.py simple que funcione"""
    print("\n📝 Creando app.py optimizado...")
    
    app_content = '''#!/usr/bin/env python3
"""
Aplicación Principal PTZ Tracker
Punto de entrada optimizado
"""

import sys
import os

# Agregar directorio actual al PATH
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def main():
    """Función principal"""
    try:
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        
        # Configurar OpenGL antes de crear QApplication
        try:
            QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseOpenGLES, True)
        except:
            pass
        
        # Crear aplicación
        app = QApplication(sys.argv)
        app.setQuitOnLastWindowClosed(True)
        
        # Buscar MainGUI en diferentes ubicaciones
        main_gui_class = None
        
        try:
            from ui.main_window import MainGUI
            main_gui_class = MainGUI
            print("✅ MainGUI cargado desde ui.main_window")
        except ImportError:
            try:
                from main_window import MainGUI
                main_gui_class = MainGUI
                print("✅ MainGUI cargado desde main_window")
            except ImportError:
                try:
                    from gui.components import MainGUI
                    main_gui_class = MainGUI
                    print("✅ MainGUI cargado desde gui.components")
                except ImportError:
                    print("❌ No se pudo encontrar MainGUI")
                    return 1
        
        # Crear y mostrar ventana principal
        if main_gui_class:
            gui = main_gui_class()
            gui.show()
            
            print("🚀 Aplicación PTZ Tracker iniciada")
            print("💡 Busca el menú '🎯 PTZ' para opciones de seguimiento")
            
            return app.exec()
        else:
            print("❌ No se pudo inicializar la aplicación")
            return 1
            
    except Exception as e:
        print(f"❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''
    
    try:
        with open("app.py", 'w', encoding='utf-8') as f:
            f.write(app_content)
        
        print("✅ app.py optimizado creado")
        return True
        
    except Exception as e:
        print(f"❌ Error creando app.py: {e}")
        return False

def test_application():
    """Probar que la aplicación se puede importar sin errores"""
    print("\n🧪 Probando importación de la aplicación...")
    
    try:
        import sys
        current_dir = os.getcwd()
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Probar imports principales
        try:
            from ui.main_window import MainGUI
            print("✅ MainGUI importado desde ui.main_window")
            return True
        except ImportError:
            try:
                from main_window import MainGUI
                print("✅ MainGUI importado desde main_window")
                return True
            except ImportError:
                print("❌ No se pudo importar MainGUI")
                return False
        
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def main():
    """Función principal"""
    print("🔧 CORRECCIÓN DE PROBLEMAS DE INICIALIZACIÓN")
    print("=" * 50)
    
    # Paso 1: Corregir main_window.py
    if fix_mainwindow_initialization():
        print("✅ main_window.py corregido")
    else:
        print("❌ Error corrigiendo main_window.py")
        return False
    
    # Paso 2: Crear app.py optimizado
    if create_simple_app_py():
        print("✅ app.py optimizado creado")
    else:
        print("⚠️ Error creando app.py")
    
    # Paso 3: Probar aplicación
    if test_application():
        print("✅ Aplicación lista para ejecutar")
    else:
        print("⚠️ Puede haber problemas de importación")
    
    print("\n🎉 CORRECCIÓN COMPLETADA")
    print("=" * 30)
    print("\n🚀 Ahora ejecuta:")
    print("  python app.py")
    print("  python main_window.py")
    print("  python ui/main_window.py")
    
    print("\n💡 Si aún hay errores:")
    print("  1. Verifica que debug_console se inicialice antes que PTZ")
    print("  2. Usa el app.py optimizado")
    print("  3. Revisa el orden en __init__ de MainGUI")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)