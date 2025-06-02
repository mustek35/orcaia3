from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QDialogButtonBox, QComboBox, QLabel, QSpinBox
)
from configuracion import ConfiguracionWidget
from PyQt6.QtCore import pyqtSignal

class ConfiguracionDialog(QDialog):
    iniciar_camara_secundaria = pyqtSignal(object)

    def __init__(self, parent=None, camera_list=None):
        super().__init__(parent)
        self.setWindowTitle("Configuración del Sistema")
        self.setMinimumSize(400, 400)

        self.camera_list = camera_list or []
        self.selected_camera = None

        self.layout = QVBoxLayout()

        self.camera_selector = QComboBox()
        self.camera_selector.addItems([
            f"{cam.get('ip', 'IP desconocida')} - {cam.get('tipo', 'Tipo desconocido')}"
            for cam in self.camera_list
        ])
        self.camera_selector.currentIndexChanged.connect(self.update_camera_selection)
        self.layout.addWidget(QLabel("Seleccionar Cámara"))
        self.layout.addWidget(self.camera_selector)

        self.modelo_selector = QComboBox()
        self.modelo_selector.addItems(["Embarcaciones", "Personas", "Autos", "Barcos"])
        self.layout.addWidget(QLabel("Modelo de detección"))
        self.layout.addWidget(self.modelo_selector)

        self.conf_selector = QComboBox()
        self.conf_selector.addItems(["0.25", "0.5", "0.75"])
        self.layout.addWidget(QLabel("Confianza mínima"))
        self.layout.addWidget(self.conf_selector)

        self.imgsz_selector = QComboBox()
        self.imgsz_selector.addItems(["640", "960", "1280", "1920"])
        self.layout.addWidget(QLabel("Resolución de análisis (imgsz)"))
        self.layout.addWidget(self.imgsz_selector)

        self.intervalo_label = QLabel("Intervalo de detección (frames)")
        self.intervalo_input = QSpinBox()
        self.intervalo_input.setRange(1, 500)
        self.intervalo_input.setValue(80)
        self.layout.addWidget(self.intervalo_label)
        self.layout.addWidget(self.intervalo_input)

        self.config_widget = ConfiguracionWidget()
        self.layout.addWidget(self.config_widget)

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.aceptar_configuracion)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

        self.setLayout(self.layout)

        if self.camera_list:
            self.camera_selector.setCurrentIndex(0)
            self.update_camera_selection()

    def update_camera_selection(self):
        idx = self.camera_selector.currentIndex()
        if idx >= 0:
            self.selected_camera = self.camera_list[idx]

            modelo = self.selected_camera.get("modelo", "Personas")
            conf = str(self.selected_camera.get("confianza", 0.5))
            imgsz = str(self.selected_camera.get("imgsz", 640))
            intervalo = int(self.selected_camera.get("intervalo", 80))

            modelo_idx = self.modelo_selector.findText(modelo)
            conf_idx = self.conf_selector.findText(conf)
            imgsz_idx = self.imgsz_selector.findText(imgsz)

            self.modelo_selector.setCurrentIndex(modelo_idx if modelo_idx >= 0 else 0)
            self.conf_selector.setCurrentIndex(conf_idx if conf_idx >= 0 else 1)
            self.imgsz_selector.setCurrentIndex(imgsz_idx if imgsz_idx >= 0 else 0)
            self.intervalo_input.setValue(intervalo)

            self.config_widget.combo_res.setCurrentText(self.selected_camera.get("resolucion", "main"))
            self.config_widget.score_slider.setValue(int(self.selected_camera.get("umbral", 0.5) * 100))
            self.config_widget.save_checkbox.setChecked(self.selected_camera.get("guardar_capturas", False))
            self.config_widget.centinela_checkbox.setChecked(self.selected_camera.get("modo_centinela", False))

    def obtener_config(self):
        if self.selected_camera is not None:
            self.selected_camera["modelo"] = self.modelo_selector.currentText()
            self.selected_camera["confianza"] = float(self.conf_selector.currentText())
            self.selected_camera["intervalo"] = self.intervalo_input.value()
            self.selected_camera["imgsz"] = int(self.imgsz_selector.currentText())

            config = self.config_widget.obtener_config()
            self.selected_camera.update(config)

            return {
                "camara": self.selected_camera,
                "configuracion": config
            }

    def aceptar_configuracion(self):
        result = self.obtener_config()
        self.iniciar_camara_secundaria.emit(result["camara"])
        self.accept()
