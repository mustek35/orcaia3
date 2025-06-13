import os
os.environ["FFREPORT"] = "file=ffreport.log:level=quiet"
os.environ["FFMPEG_LOGLEVEL"] = "panic"
os.environ["QT_LOGGING_RULES"] = "qt.multimedia.ffmpeg=false;qt.multimedia.playbackengine=false"
os.environ["QT_MEDIA_FFMPEG_LOGLEVEL"] = "fatal"

import cv2
# Establecer nivel de log en OpenCV si el módulo de logging está disponible:
if hasattr(cv2, "utils") and hasattr(cv2.utils, "logging"):
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

import sys
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainGUI

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = MainGUI()
    gui.show()
    sys.exit(app.exec())
