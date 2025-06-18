import sys
import os
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Configurar environment para GStreamer
os.environ['GST_DEBUG'] = '2'
os.environ['GST_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/gstreamer-1.0'

def main():
    # Verificar requisitos antes de iniciar
    print("🔍 Verificando requisitos de Fase 2...")
    
    # Verificar GStreamer
    try:
        from core.gstreamer_video_reader import check_gstreamer_availability
        available, message = check_gstreamer_availability()
        if not available:
            print(f"❌ GStreamer: {message}")
            return 1
        print(f"✅ GStreamer: {message}")
    except Exception as e:
        print(f"❌ Error verificando GStreamer: {e}")
        return 1
    
    # Verificar CUDA
    try:
        from core.cuda_pipeline_processor import verify_cuda_pipeline_requirements
        requirements, issues = verify_cuda_pipeline_requirements()
        if issues:
            print("❌ CUDA Pipeline:")
            for issue in issues:
                print(f"  - {issue}")
            return 1
        print(f"✅ CUDA: {requirements.get('gpu_info', 'N/A')}")
    except Exception as e:
        print(f"❌ Error verificando CUDA: {e}")
        return 1
    
    # Iniciar aplicación
    app = QApplication(sys.argv)
    app.setApplicationName("Monitor PTZ - Fase 2")
    
    # Optimizaciones Qt
    app.setAttribute(Qt.ApplicationAttribute.AA_UseOpenGLES, True)
    
    # Importar y crear ventana principal (implementar en siguiente paso)
    from ui.main_window_phase2 import Phase2MainWindow
    window = Phase2MainWindow()
    window.show()
    
    print("🚀 Monitor PTZ Fase 2 iniciado")
    print("📊 Pipeline: GStreamer NVDEC + CUDA + Advanced Tracking")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())