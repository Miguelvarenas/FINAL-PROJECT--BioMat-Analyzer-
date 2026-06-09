# =====================================================================
# VISTA - Proyecto final Informática II - Arquitectura MVC
# =====================================================================
# Este archivo NO diseña la interfaz desde cero. Carga los .ui creados en
# Qt Designer y configura sus widgets para que el Controlador los use.
# =====================================================================

import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLineEdit,
    QMainWindow,
    QMessageBox,
)
from PyQt5.uic import loadUi


def obtener_ruta_ui(nombre_archivo):
    carpeta_proyecto = os.path.dirname(os.path.abspath(__file__))
    for carpeta in ("interfaces", "Interfaces"):
        ruta = os.path.join(carpeta_proyecto, carpeta, nombre_archivo)
        if os.path.exists(ruta):
            return ruta
    raise FileNotFoundError(
        f"No se encontró {nombre_archivo}. Debe estar dentro de la carpeta interfaces o Interfaces."
    )


class VistaBase(QMainWindow):
    def validar_widgets(self, nombres_widgets):
        faltantes = [nombre for nombre in nombres_widgets if not hasattr(self, nombre)]
        if faltantes:
            raise AttributeError(
                "Faltan widgets en el archivo .ui:\n- "
                + "\n- ".join(faltantes)
                + "\n\nCorrige el objectName de esos elementos en Qt Designer."
            )

    def mostrar_mensaje(self, titulo, mensaje, tipo="info"):
        if tipo == "error":
            QMessageBox.critical(self, titulo, mensaje)
        elif tipo == "advertencia":
            QMessageBox.warning(self, titulo, mensaje)
        else:
            QMessageBox.information(self, titulo, mensaje)


class VentanaLogin(VistaBase):
    def __init__(self):
        super().__init__()
        loadUi(obtener_ruta_ui("bienvenida_login.ui"), self)
        self.validar_widgets(["txt_usuario", "txt_password", "btn_login"])
        self.configurar_login()

    def configurar_login(self):
        self.setWindowTitle("BioMed Analyzer - Inicio de sesión")
        self.txt_usuario.setPlaceholderText("Usuario")
        self.txt_password.setPlaceholderText("Contraseña")
        self.txt_password.setEchoMode(QLineEdit.Password)
        self.btn_login.setText("Iniciar sesión")
        self.btn_login.setCursor(Qt.PointingHandCursor)
        self.btn_login.setDefault(True)

    def obtener_credenciales(self):
        return self.txt_usuario.text().strip(), self.txt_password.text().strip()

    def limpiar_password(self):
        self.txt_password.clear()
        self.txt_password.setFocus()


class VentanaPrincipal(VistaBase):
    def __init__(self):
        super().__init__()
        loadUi(obtener_ruta_ui("principal.ui"), self)
        self.validar_widgets([
            "lbl_nombre_usuario",
            "lbl_fecha_foto",
            "lbl_axial",
            "lbl_graficos_senales",
            "lbl_grafico_tabular",
            "btn_exportar_nifti",
            "slider_axial",
            "slider_sagital",
            "slider_coronal",
            "combo_binarizacion",
            "combo_morfologia",
            "spin_kernel",
            "slider_ruido",
            "combo_scatter_x",
            "combo_scatter_y",
            "tabla_dicom",
            "tabla_resumen",
        ])
        self.configurar_principal()
        self.configurar_controles_imagenes()
        self.configurar_controles_senales()
        self.configurar_tablas()
        self.configurar_contenedores_graficos()

    def configurar_principal(self):
        self.setWindowTitle("BioMed Analyzer - Panel diagnóstico multimodal")
        self.lbl_nombre_usuario.setText("Profesional: No autenticado")
        self.lbl_fecha_foto.setText("Sesión pendiente de autenticación fotográfica")

    def configurar_controles_imagenes(self):
        for slider in (self.slider_axial, self.slider_sagital, self.slider_coronal):
            slider.setMinimum(0)
            slider.setMaximum(0)
            slider.setValue(0)
            slider.setTickPosition(slider.TicksBelow)
            slider.setTickInterval(1)
            slider.setCursor(Qt.PointingHandCursor)

        self.combo_binarizacion.clear()
        self.combo_binarizacion.addItems([
            "Ninguno",
            "Otsu automático",
            "Binario 80",
            "Binario 100",
            "Binario 128",
            "Binario 160",
            "Binario invertido 80",
            "Binario invertido 128",
            "Truncado 128",
            "ToZero 128",
            "ToZero invertido 128",
        ])
        self.combo_binarizacion.setCurrentIndex(0)

        self.combo_morfologia.clear()
        self.combo_morfologia.addItems([
            "Ninguno",
            "Erosión",
            "Dilatación",
            "Apertura",
            "Cierre",
            "Gradiente",
            "TopHat",
            "BlackHat",
        ])
        self.combo_morfologia.setCurrentIndex(0)

        self.spin_kernel.setRange(1, 31)
        self.spin_kernel.setSingleStep(2)
        self.spin_kernel.setValue(3)
        self.btn_exportar_nifti.setText("Exportar NIfTI")
        self.btn_exportar_nifti.setCursor(Qt.PointingHandCursor)

        if hasattr(self, "btn_camara"):
            self.btn_camara.setText("Capturar foto de sesión")
            self.btn_camara.setCursor(Qt.PointingHandCursor)

    def configurar_controles_senales(self):
        self.slider_ruido.setRange(0, 100)
        self.slider_ruido.setValue(0)
        self.slider_ruido.setTickPosition(self.slider_ruido.TicksBelow)
        self.slider_ruido.setTickInterval(10)
        self.slider_ruido.setToolTip("Control suave de ruido gaussiano: 0 sin ruido, 100 ruido bajo/moderado.")
        self.slider_ruido.setCursor(Qt.PointingHandCursor)

    def configurar_tablas(self):
        for tabla in (self.tabla_dicom, self.tabla_resumen):
            tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)
            tabla.setSelectionBehavior(QAbstractItemView.SelectRows)
            tabla.setAlternatingRowColors(True)
            tabla.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.tabla_dicom.setColumnCount(2)
        self.tabla_dicom.setHorizontalHeaderLabels(["Propiedad Médica", "Valor"])

    def configurar_contenedores_graficos(self):
        self.lbl_axial.setText("")
        self.lbl_graficos_senales.setText("")
        self.lbl_grafico_tabular.setText("")
        self.lbl_axial.setMinimumSize(650, 420)
        self.lbl_graficos_senales.setMinimumSize(550, 360)
        self.lbl_grafico_tabular.setMinimumSize(550, 480)


__all__ = ["VentanaLogin", "VentanaPrincipal", "obtener_ruta_ui"]
