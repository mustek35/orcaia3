# Solución de Problemas Sistema PTZ

## 🔍 Problemas Comunes

### Error: "No se puede conectar a cámara PTZ"
**Solución:**
1. Verificar IP, usuario y contraseña
2. Comprobar conectividad de red
3. Verificar que cámara soporta LightAPI

### Error: "Imports PTZ no encontrados"
**Solución:**
1. Verificar estructura de carpetas
2. Ejecutar `python setup_ptz_structure.py`
3. Verificar archivos __init__.py

### Seguimiento no funciona
**Solución:**
1. Verificar que cámara está configurada como tipo "PTZ"
2. Habilitar seguimiento en Panel PTZ
3. Configurar zona de seguimiento
4. Verificar detecciones YOLO

### Performance lento
**Solución:**
1. Reducir sensibilidad de seguimiento
2. Configurar zona más pequeña
3. Ajustar FPS de detección

## 📞 Contacto
Para soporte adicional, revisar logs en `logs/ptz_tracking.log`
