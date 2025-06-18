from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QGridLayout, QScrollArea, 
                             QPushButton, QHBoxLayout, QMessageBox, QDialog, 
                             QDialogButtonBox, QSpacerItem, QSizePolicy, QFileDialog)
from PyQt6.QtGui import QPixmap, QCursor
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread # Asegurar QThread explícitamente
import os
from datetime import datetime
import glob
import json
import shutil

class ImageDetailDialog(QDialog): # Mantener esta clase como está definida previamente
    image_deleted_signal = pyqtSignal(str) 

    def __init__(self, image_path, metadata, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.metadata = metadata 
        self.setWindowTitle("🖼️ Captura Detallada")
        self.setMinimumWidth(400)
        main_layout = QVBoxLayout(self)
        self.image_label = QLabel()
        pixmap = QPixmap(self.image_path)
        if not pixmap.isNull():
            self.image_label.setPixmap(pixmap.scaled(300, 300, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        else:
            self.image_label.setText("No se pudo cargar la imagen.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_label)
        self.metadata_label = QLabel()
        self.metadata_label.setTextFormat(Qt.TextFormat.RichText)
        self.metadata_label.setText(self.format_metadata())
        main_layout.addWidget(self.metadata_label)
        main_layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        button_layout = QHBoxLayout()
        self.download_button = QPushButton("Descargar")
        self.download_button.clicked.connect(self.download_image)
        button_layout.addWidget(self.download_button)
        self.delete_button = QPushButton("Borrar")
        self.delete_button.clicked.connect(self.delete_image) 
        button_layout.addWidget(self.delete_button)
        button_layout.addSpacerItem(QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        button_layout.addWidget(self.ok_button)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def format_metadata(self):
        path_text = self.image_path
        fecha = self.metadata.get("fecha", "❓")
        hora = self.metadata.get("hora", "❓")
        modelo = self.metadata.get("modelo", "❓")
        coords_bbox_str = "No disponibles"
        coords_bbox_data = self.metadata.get("coordenadas_frame_original")
        if isinstance(coords_bbox_data, (list, tuple)) and len(coords_bbox_data) == 4:
            coords_bbox_str = f"[{coords_bbox_data[0]}, {coords_bbox_data[1]}, {coords_bbox_data[2]}, {coords_bbox_data[3]}]"
        elif isinstance(coords_bbox_data, str): 
            coords_bbox_str = coords_bbox_data
        coords_ptz_str = "No disponibles"
        coords_ptz_data = self.metadata.get("coordenadas_ptz")
        if coords_ptz_data and not (isinstance(coords_ptz_data, str) and coords_ptz_data.lower() == "no disponibles"):
            if isinstance(coords_ptz_data, (list, tuple)) and len(coords_ptz_data) >= 2: 
                 coords_ptz_str = f"({coords_ptz_data[0]}, {coords_ptz_data[1]})" 
            else:
                 coords_ptz_str = str(coords_ptz_data)
        elif isinstance(coords_ptz_data, str): 
            coords_ptz_str = coords_ptz_data
        confianza = str(self.metadata.get("confianza", "No disponible"))
        return (
            f"<b>Ruta:</b><br>{path_text}<br><br>"
            f"<b>Fecha:</b> {fecha}<br>"
            f"<b>Hora:</b> {hora}<br>"
            f"<b>Modelo:</b> {modelo}<br>"
            f"<b>BBox Original:</b> {coords_bbox_str}<br>"
            f"<b>PTZ:</b> {coords_ptz_str}<br>"
            f"<b>Confianza:</b> {confianza}"
        )

    def download_image(self):
        if not os.path.exists(self.image_path):
            QMessageBox.warning(self, "Error", "El archivo de imagen original no existe.")
            return
        original_filename = os.path.basename(self.image_path)
        save_path, _ = QFileDialog.getSaveFileName(self, "Guardar imagen como...",
            os.path.join(os.path.expanduser("~"), "Downloads", original_filename), 
            "JPEG Image (*.jpg *.jpeg);;PNG Image (*.png);;All Files (*)")
        if save_path: 
            try:
                shutil.copy2(self.image_path, save_path) 
                QMessageBox.information(self, "Éxito", f"Imagen guardada en:\n{save_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error al Guardar", f"No se pudo guardar la imagen.\nError: {e}")

    def delete_image(self):
        confirm_msg = QMessageBox.warning(self, "Confirmar Borrado",
            "¿Está seguro de que desea borrar esta imagen y sus metadatos?\nEsta acción no se puede deshacer.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if confirm_msg == QMessageBox.StandardButton.Yes:
            image_file_to_delete = self.image_path
            metadata_file_to_delete = os.path.splitext(self.image_path)[0] + ".json"
            error_messages = []
            if os.path.exists(image_file_to_delete):
                try: os.remove(image_file_to_delete)
                except Exception as e: error_messages.append(f"No se pudo borrar el archivo de imagen: {e}")
            else: error_messages.append("El archivo de imagen no existía.") 
            if os.path.exists(metadata_file_to_delete):
                try: os.remove(metadata_file_to_delete)
                except Exception as e: error_messages.append(f"No se pudo borrar el archivo de metadatos: {e}")
            else: error_messages.append("El archivo de metadatos no existía.")
            image_actually_gone = not os.path.exists(image_file_to_delete)
            metadata_actually_gone = not os.path.exists(metadata_file_to_delete)
            only_non_existence_errors = all("no existía" in msg for msg in error_messages)
            if image_actually_gone and metadata_actually_gone:
                if not error_messages or only_non_existence_errors:
                     QMessageBox.information(self, "Éxito", "Imagen y metadatos borrados correctamente o ya no existían.")
                else: 
                     QMessageBox.warning(self, "Error Parcial o Inconsistencia", "Los archivos ya no existen, pero se reportaron errores durante el proceso.\nDetalles:\n" + "\n".join(error_messages))
                self.image_deleted_signal.emit(self.image_path)
                self.accept()
            else: 
                QMessageBox.critical(self, "Error al Borrar", "Ocurrió un error al intentar borrar los archivos.\nDetalles:\n" + "\n".join(error_messages))

class UpdateResumenThread(QThread):
    datos_listos = pyqtSignal(dict, list) 
    error_ocurrido = pyqtSignal(str) 

    def __init__(self, parent=None):
        super().__init__(parent)
        # self.hoy_str se establecerá en run para usar la fecha actual en cada ejecución

    def run(self):
        try:
            hoy_str = datetime.now().strftime("%Y-%m-%d") # Fecha actual para esta ejecución
            conteos = {}
            base = "capturas"
            carpetas_conteo = {
                "personas": "Personas", "autos": "Autos", 
                "barcos": "Barcos", "embarcaciones": "Embarcaciones"
            }
            for carpeta_key, _ in carpetas_conteo.items():
                ruta_conteo = os.path.join(base, carpeta_key, hoy_str)
                count = 0
                if os.path.exists(ruta_conteo):
                    count = len([f for f in os.listdir(ruta_conteo) if f.endswith(".jpg")])
                conteos[carpeta_key] = count

            imagenes_totales_sorted = []
            for carpeta_key in carpetas_conteo.keys():
                ruta_glob = os.path.join(base, carpeta_key, hoy_str, "*.jpg")
                encontrados = glob.glob(ruta_glob)
                imagenes_totales_sorted.extend(encontrados)
            
            if imagenes_totales_sorted:
                imagenes_totales_sorted.sort(key=os.path.getmtime, reverse=True)
            
            self.datos_listos.emit(conteos, imagenes_totales_sorted)
        except Exception as e:
            self.error_ocurrido.emit(f"Error en UpdateResumenThread: {e}")

class ResumenDeteccionesWidget(QWidget):
    log_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(350)
        self.pagina_actual = 0
        self.imagenes_totales = []
        self.items_per_page = 20 

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.titulo = QLabel("📊 Detecciones por Día")
        self.titulo.setStyleSheet("font-weight: bold; font-size: 16px;")
        self.layout.addWidget(self.titulo)

        self.label_personas = QLabel("Personas: Cargando...") # Estado inicial
        self.label_autos = QLabel("Autos: Cargando...")
        self.label_barcos = QLabel("Barcos: Cargando...")
        self.label_embarcaciones = QLabel("Embarcaciones: Cargando...")

        self.layout.addWidget(self.label_personas)
        self.layout.addWidget(self.label_autos)
        self.layout.addWidget(self.label_barcos)
        self.layout.addWidget(self.label_embarcaciones)

        self.titulo_imagenes = QLabel("\n🖼️ Últimas Capturas:")
        self.titulo_imagenes.setStyleSheet("font-weight: bold; margin-top: 10px;")
        self.layout.addWidget(self.titulo_imagenes)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QGridLayout()
        self.scroll_content.setLayout(self.scroll_layout)
        self.scroll.setWidget(self.scroll_content)
        self.layout.addWidget(self.scroll)

        self.controles_layout = QHBoxLayout()
        self.btn_anterior = QPushButton("⏪ Anterior")
        self.btn_siguiente = QPushButton("Siguiente ⏩")
        self.btn_anterior.clicked.connect(self.pagina_anterior)
        self.btn_siguiente.clicked.connect(self.pagina_siguiente)
        self.controles_layout.addWidget(self.btn_anterior)
        self.controles_layout.addWidget(self.btn_siguiente)
        self.layout.addLayout(self.controles_layout)
        
        self.btn_anterior.setEnabled(False) # Deshabilitar inicialmente
        self.btn_siguiente.setEnabled(False)

        # Hilo para actualización
        self.update_thread = UpdateResumenThread(self)
        self.update_thread.datos_listos.connect(self._procesar_datos_resumen)
        self.update_thread.error_ocurrido.connect(self._manejar_error_resumen)
        self.update_thread.finished.connect(self._on_update_thread_finished)

        self.timer_actualizacion_resumen = QTimer(self) # Renombrado para claridad
        self.timer_actualizacion_resumen.timeout.connect(self.actualizar_resumen)
        self.timer_actualizacion_resumen.start(10000)  # Actualizar cada 10 segundos

        self.actualizar_resumen() # Carga inicial

    def _procesar_datos_resumen(self, conteos, imagenes_totales):
        self.label_personas.setText(f"Personas: {conteos.get('personas', 0)}")
        self.label_autos.setText(f"Autos: {conteos.get('autos', 0)}")
        self.label_barcos.setText(f"Barcos: {conteos.get('barcos', 0)}")
        self.label_embarcaciones.setText(f"Embarcaciones: {conteos.get('embarcaciones', 0)}")

        self.imagenes_totales = imagenes_totales
        self.pagina_actual = 0 
        self.mostrar_pagina()
        
        self.btn_anterior.setEnabled(self.pagina_actual > 0)
        self.btn_siguiente.setEnabled(len(self.imagenes_totales) > self.items_per_page * (self.pagina_actual + 1))


    def _manejar_error_resumen(self, error_msg):
        self.log_signal.emit(f"Error actualizando resumen: {error_msg}")
        self.label_personas.setText("Personas: Error") # Indicar error en UI
        self.label_autos.setText("Autos: Error")
        self.label_barcos.setText("Barcos: Error")
        self.label_embarcaciones.setText("Embarcaciones: Error")
        # Re-habilitar botones para permitir reintento manual o automático por timer
        self.btn_anterior.setEnabled(False) # O True si se quiere permitir paginación de datos viejos
        self.btn_siguiente.setEnabled(False)


    def _on_update_thread_finished(self):
        # print("Hilo de actualización de resumen finalizado.")
        # No es necesario re-habilitar botones aquí si el timer sigue activo
        # y _procesar_datos_resumen o _manejar_error_resumen ya los ajustan.
        pass

    def actualizar_resumen(self):
        if not self.update_thread.isRunning():
            self.btn_anterior.setEnabled(False)
            self.btn_siguiente.setEnabled(False)
            # Actualizar etiquetas a "Cargando..."
            self.label_personas.setText("Personas: Cargando...")
            self.label_autos.setText("Autos: Cargando...")
            self.label_barcos.setText("Barcos: Cargando...")
            self.label_embarcaciones.setText("Embarcaciones: Cargando...")
            self.update_thread.start()
        # else:
            # print("La actualización del resumen ya está en curso.")


    def actualizar_imagenes(self): # Este método ya no es necesario, el hilo lo reemplaza
        pass # El contenido fue movido a UpdateResumenThread y _procesar_datos_resumen

    def mostrar_pagina(self):
        for i in reversed(range(self.scroll_layout.count())):
            widget = self.scroll_layout.itemAt(i).widget()
            if widget: widget.deleteLater()

        if not self.imagenes_totales:
            label = QLabel("❌ No se encontraron imágenes para hoy.") # Mensaje más específico
            self.scroll_layout.addWidget(label, 0, 0, 1, 3) # Span all columns
            self.btn_anterior.setEnabled(False)
            self.btn_siguiente.setEnabled(False)
            return

        inicio = self.pagina_actual * self.items_per_page
        fin = inicio + self.items_per_page
        pagina_imagenes = self.imagenes_totales[inicio:fin]

        columnas = 3
        for idx, path in enumerate(pagina_imagenes):
            pixmap = QPixmap(path)
            if pixmap.isNull(): continue
            thumb = QLabel()
            thumb.setFixedSize(120, 120)
            thumb.setStyleSheet("QLabel { border: 1px solid #888; margin: 4px; } QLabel:hover { border: 2px solid #00FF00; }")
            thumb.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            thumb.setPixmap(pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            thumb.setToolTip(os.path.basename(path))
            thumb.mousePressEvent = lambda e, p=path: self.mostrar_modal(p)
            fila = idx // columnas
            col = idx % columnas
            self.scroll_layout.addWidget(thumb, fila, col)
        
        self.btn_anterior.setEnabled(self.pagina_actual > 0)
        self.btn_siguiente.setEnabled(len(self.imagenes_totales) > fin)


    def pagina_anterior(self):
        if self.pagina_actual > 0:
            self.pagina_actual -= 1
            self.mostrar_pagina()

    def pagina_siguiente(self):
        # No necesita len(self.imagenes_totales) == 0 porque mostrar_pagina ya lo maneja
        total_paginas = (len(self.imagenes_totales) + self.items_per_page - 1) // self.items_per_page
        if self.pagina_actual < total_paginas - 1:
            self.pagina_actual += 1
            self.mostrar_pagina()

    def mostrar_modal(self, path):
        metadata_path = os.path.splitext(path)[0] + ".json"
        loaded_metadata = { 
            "fecha": "❓", "hora": "❓", "modelo": "❓",
            "coordenadas_frame_original": "No disponibles", 
            "coordenadas_ptz": "No disponibles",          
            "confianza": "No disponible"
        }
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    data_from_file = json.load(f)
                    loaded_metadata["fecha"] = data_from_file.get("fecha", "❓")
                    loaded_metadata["hora"] = data_from_file.get("hora", "❓")
                    loaded_metadata["modelo"] = data_from_file.get("modelo", "❓")
                    loaded_metadata["coordenadas_frame_original"] = data_from_file.get("coordenadas_frame_original", "No disponibles")
                    loaded_metadata["coordenadas_ptz"] = data_from_file.get("coordenadas_ptz", "No disponibles")
                    loaded_metadata["confianza"] = str(data_from_file.get("confianza", "No disponible"))
            except Exception as e:
                self.log_signal.emit(f"Error cargando metadata {metadata_path}: {e}") # Usar log_signal
        
        dialog = ImageDetailDialog(image_path=path, metadata=loaded_metadata, parent=self)
        dialog.image_deleted_signal.connect(self.handle_image_deleted) 
        dialog.exec()

    def handle_image_deleted(self, deleted_image_path):
        self.log_signal.emit(f"🖼️ Imagen {os.path.basename(deleted_image_path)} borrada.") 
        if deleted_image_path in self.imagenes_totales:
            self.imagenes_totales.remove(deleted_image_path)
            # No es necesario recalcular pagina_actual aquí si mostrar_pagina lo hace
            # o si la lógica de _procesar_datos_resumen se vuelve a llamar indirectamente.
            # Por ahora, la llamada directa a mostrar_pagina es la más simple.
            # Ajustar pagina_actual si es necesario para evitar quedar fuera de rango
            total_imagenes = len(self.imagenes_totales)
            if total_imagenes == 0:
                self.pagina_actual = 0
            else:
                max_pagina = (total_imagenes -1) // self.items_per_page
                if self.pagina_actual > max_pagina:
                    self.pagina_actual = max_pagina
            self.mostrar_pagina() 
        else:
            self.log_signal.emit(f"⚠️ Imagen {os.path.basename(deleted_image_path)} no encontrada en la lista interna al intentar refrescar.")
