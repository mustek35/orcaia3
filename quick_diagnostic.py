# quick_diagnostic.py - Diagnóstico rápido
import subprocess
import cv2
import sys

def check_ffmpeg():
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False

def check_opencv_ffmpeg():
    # Verificar si OpenCV tiene soporte FFmpeg
    return hasattr(cv2, 'CAP_FFMPEG') and cv2.CAP_FFMPEG is not None

print("🔍 Diagnóstico rápido:")
print(f"   FFmpeg instalado: {'✅' if check_ffmpeg() else '❌'}")
print(f"   OpenCV-FFmpeg:    {'✅' if check_opencv_ffmpeg() else '❌'}")

if not check_ffmpeg():
    print("\n💡 SOLUCIÓN RTSP:")
    print("   1. Descarga FFmpeg: https://ffmpeg.org/download.html")
    print("   2. Añádelo al PATH del sistema")
    print("   3. Reinicia el terminal")

if not check_opencv_ffmpeg():
    print("\n💡 SOLUCIÓN OpenCV:")
    print("   pip uninstall opencv-python")
    print("   pip install opencv-python")