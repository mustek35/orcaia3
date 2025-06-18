"""
Interfaces Gráficas del Sistema PTZ
Widgets y componentes de UI
"""

# Importaciones PTZ
try:
    from .ptz_config_widget import PTZConfigWidget, PTZControlThread
    PTZ_GUI_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: PTZ GUI components not available: {e}")
    PTZ_GUI_AVAILABLE = False

# Importaciones existentes
try:
    from .grilla_widget import GrillaWidget
    GRILLA_AVAILABLE = True
except ImportError:
    GRILLA_AVAILABLE = False

try:
    from .components import MainGUI
    MAIN_GUI_AVAILABLE = True
except ImportError:
    MAIN_GUI_AVAILABLE = False

# Construir __all__ dinámicamente
__all__ = []
if PTZ_GUI_AVAILABLE:
    __all__.extend(['PTZConfigWidget', 'PTZControlThread'])
if GRILLA_AVAILABLE:
    __all__.append('GrillaWidget')
if MAIN_GUI_AVAILABLE:
    __all__.append('MainGUI')

__version__ = "1.0.0"
