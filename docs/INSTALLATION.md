# Guía de Instalación Sistema PTZ

## 📋 Prerrequisitos
- Python 3.8+
- PyQt6
- requests
- numpy

## 🔧 Instalación Paso a Paso

### 1. Crear Estructura
```bash
python setup_ptz_structure.py
```

### 2. Instalar Dependencias
```bash
pip install PyQt6 requests numpy
```

### 3. Configurar Cámaras PTZ
Editar `config/ptz_tracking_config.json`:
```json
{
  "ptz_cameras": [
    {
      "ip": "192.168.1.100",
      "username": "admin", 
      "password": "password123",
      "tracking_enabled": true
    }
  ]
}
```

### 4. Ejecutar Integración
```bash
python scripts/integrate_ptz_with_existing_app.py
```

### 5. Iniciar Aplicación
```bash
python main_gui.py
```

## ✅ Verificación
- Verificar que aparecen botones PTZ en interfaz
- Probar conexión con cámara PTZ
- Configurar zona de seguimiento
