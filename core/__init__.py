"""
Núcleo del Sistema PTZ
Componentes principales de seguimiento y control
"""

# Importaciones principales
try:
    from .ptz_tracking_system import PTZTrackingManager, PTZTrackingThread, DetectionEvent
    from .light_api import LightAPI, PTZDirection, ZoomDirection, PresetInfo
    from .ptz_integration import PTZSystemIntegration, PTZControlInterface
    from .grid_utils import GridUtils, GridCell, GridZone
    from .detection_ptz_bridge import DetectionPTZBridge, detection_ptz_bridge
    
    __all__ = [
        'PTZTrackingManager', 'PTZTrackingThread', 'DetectionEvent',
        'LightAPI', 'PTZDirection', 'ZoomDirection', 'PresetInfo',
        'PTZSystemIntegration', 'PTZControlInterface', 
        'GridUtils', 'GridCell', 'GridZone',
        'DetectionPTZBridge', 'detection_ptz_bridge'
    ]
    
except ImportError as e:
    print(f"⚠️ Warning: Some PTZ modules not available: {e}")
    __all__ = []

__version__ = "1.0.0"
