# Sistema PTZ Profesional

## 🎯 Descripción
Sistema completo de seguimiento PTZ integrado con detección de objetos YOLO.

## 🚀 Características
- Seguimiento automático basado en detecciones
- Control manual de cámaras PTZ
- Configuración de zonas de seguimiento
- Interfaz gráfica profesional
- API completa para cámaras PTZ
- Logs y estadísticas detalladas

## 📁 Estructura de Archivos
```
core/          # Núcleo del sistema PTZ
gui/           # Interfaces gráficas
config/        # Archivos de configuración
logs/          # Logs del sistema
scripts/       # Scripts de automatización
utils/         # Utilidades auxiliares
tests/         # Pruebas del sistema
docs/          # Documentación
```

## 🔧 Instalación
1. Ejecutar: `python scripts/setup_ptz_structure.py`
2. Configurar cámaras PTZ en `config/ptz_tracking_config.json`
3. Ejecutar aplicación principal

## 📖 Uso
- Panel PTZ: Configuración y control manual
- Seguimiento automático: Se activa con detecciones
- Parada de emergencia: Botón STOP en interfaz

## 🆘 Soporte
Ver `TROUBLESHOOTING.md` para solución de problemas.
