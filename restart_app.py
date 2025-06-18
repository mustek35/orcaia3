# restart_app.py
"""
Script para reiniciar la aplicación después del fix
"""
import subprocess
import sys
import time

def restart_app():
    print("🔄 REINICIANDO APLICACIÓN PTZ TRACKER")
    print("=" * 40)
    
    print("⏳ Esperando 2 segundos...")
    time.sleep(2)
    
    print("🚀 Iniciando app.py...")
    try:
        # Ejecutar app.py
        subprocess.run([sys.executable, "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando app.py: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Aplicación detenida por usuario")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    restart_app()
