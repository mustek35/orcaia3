# QUICK FIX 1: Para el error de CUDA Processor
# Reemplaza tu configuración actual:
config = {
    'model_path': 'yolov8n.pt',
    'batch_size': 2,
    'confidence_threshold': 0.5,
    'half_precision': True,
}

# Por esta configuración COMPLETA:
config = {
    'model_path': 'yolov8n.pt',
    'batch_size': 2,
    'confidence_threshold': 0.5,
    'half_precision': True,
    'compile_model': False,  # ✅ CLAVE FALTANTE
    'optimize_for_inference': True,
    'warmup_iterations': 3,
    'max_det': 300,
    'iou_threshold': 0.45,
    'device': 'cuda:0',
    'verbose': False
}

# QUICK FIX 2: Para el error de RTSP
# Antes de usar WindowsVideoReaderFactory, verifica OpenCV:
import cv2

def test_opencv_rtsp():
    rtsp_url = "rtsp://root:%40Remoto753524@19.10.10.132:554/media/video1"
    
    # Configurar OpenCV para RTSP
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Intentar con FFmpeg backend
    success = cap.open(rtsp_url, cv2.CAP_FFMPEG)
    
    if success:
        ret, frame = cap.read()
        if ret:
            print(f"✅ RTSP OK: {frame.shape}")
            cap.release()
            return True
    
    cap.release()
    return False

# Ejecutar antes de usar WindowsVideoReaderFactory
if test_opencv_rtsp():
    print("✅ OpenCV puede manejar el RTSP")
else:
    print("❌ Problema con OpenCV/RTSP - verifica FFmpeg")