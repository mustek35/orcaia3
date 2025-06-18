import time
import numpy as np
from onvif import ONVIFCamera

# Movimiento actual
current_pan_speed = 0.0
current_tilt_speed = 0.0
current_zoom_speed = 0.0

# Sensibilidad ajustable
PAN_SENSITIVITY = 0.005
TILT_SENSITIVITY = 0.005
MAX_PT_SPEED = 0.5
DEADZONE_X = 0.03
DEADZONE_Y = 0.03

deteccion_confirmada_streak = 0

def track_object_continuous(ip, puerto, usuario, contrasena, cx, cy, frame_w, frame_h):
    try:
        cam = ONVIFCamera(ip, puerto, usuario, contrasena)
        media = cam.create_media_service()
        ptz = cam.create_ptz_service()
        profile = media.GetProfiles()[0]
        token = profile.token

        # Obtener estado actual
        status = ptz.GetStatus({'ProfileToken': token})
        current_pan = status.Position.PanTilt.x
        current_tilt = status.Position.PanTilt.y

        center_x = frame_w / 2
        center_y = frame_h / 2

        dx = cx - center_x
        dy = cy - center_y

        # Aplicar deadzone
        if abs(dx) < frame_w * DEADZONE_X:
            dx = 0
        if abs(dy) < frame_h * DEADZONE_Y:
            dy = 0

        # Convertir a velocidades proporcionales
        pan_speed = np.clip(dx * PAN_SENSITIVITY, -MAX_PT_SPEED, MAX_PT_SPEED)
        tilt_speed = np.clip(-dy * TILT_SENSITIVITY, -MAX_PT_SPEED, MAX_PT_SPEED)  # Invertir eje Y

        global current_pan_speed, current_tilt_speed
        global deteccion_confirmada_streak

        if pan_speed == 0 and tilt_speed == 0:
            deteccion_confirmada_streak = 0
            print("📍 Objetivo centrado. Enviando Stop.")
            request = ptz.create_type('Stop')
            request.ProfileToken = token
            request.PanTilt = True
            request.Zoom = False
            ptz.Stop(request)
            current_pan_speed = 0.0
            current_tilt_speed = 0.0
            return

        deteccion_confirmada_streak += 1
        if deteccion_confirmada_streak < 3:
            print(f"⏳ Esperando confirmación de embarcación ({deteccion_confirmada_streak}/3)...")
            return

        # Enviar comando ContinuousMove
        request = ptz.create_type('ContinuousMove')
        request.ProfileToken = token
        request.Velocity = {
            'PanTilt': {'x': float(pan_speed), 'y': float(tilt_speed)},
            'Zoom': {'x': 0.0}
        }
        ptz.ContinuousMove(request)
        current_pan_speed = pan_speed
        current_tilt_speed = tilt_speed
        print(f"🎯 PTZ seguimiento continuo: pan_speed={pan_speed:.3f}, tilt_speed={tilt_speed:.3f}")

    except Exception as e:
        print(f"❌ Error en track_object_continuous: {e}")