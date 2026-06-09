import os

import cv2
import matplotlib
import numpy as np

matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHeaderView, QTableWidgetItem, QVBoxLayout, QFileDialog, QMessageBox



class Controlador:
    def __init__(self, vista_login, vista_principal, modelo_db, modelo_img, modelo_senales, modelo_tab):
        self.vista_login = vista_login
        self.vista_principal = vista_principal
        self.db = modelo_db
        self.img = modelo_img
        self.sen = modelo_senales
        self.senales = modelo_senales
        self.tab = modelo_tab
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.roi_center_row = None
        self.roi_center_col = None
        self.roi_size_mm = 75

        self.inicializar_graficos()
        self.cargar_datos_iniciales()
        self.conectar_eventos()
        self.aplicar_estilo_login()

    # =================================
    # CONEXIONES VISTA -> CONTROLADOR
    # =================================
    def conectar_eventos(self):
        self.vista_login.btn_login.clicked.connect(self.procesar_login)

        self.cid_roi_click = self.canvas_img.mpl_connect("button_press_event", self.mover_roi_desde_click)

        if hasattr(self.vista_principal, "btn_camara"):
            self.vista_principal.btn_camara.clicked.connect(self.capturar_foto_usuario)

        self.vista_principal.btn_exportar_nifti.clicked.connect(self.exportar_volumen_nifti)
        self.vista_principal.btn_guardar_zoom.clicked.connect(self.guardar_imagen_zoom)
        self.vista_principal.slider_axial.valueChanged.connect(self.actualizar_imagenes)
        self.vista_principal.slider_sagital.valueChanged.connect(self.actualizar_imagenes)
        self.vista_principal.slider_coronal.valueChanged.connect(self.actualizar_imagenes)
        self.vista_principal.combo_binarizacion.currentIndexChanged.connect(self.actualizar_imagenes)
        self.vista_principal.combo_morfologia.currentIndexChanged.connect(self.actualizar_imagenes)
        self.vista_principal.spin_kernel.valueChanged.connect(self.actualizar_imagenes)
        # Ruido: valueChanged y sliderMoved para que la gráfica responda mientras se arrastra el control.
        # setTracking(True) permite actualización continua sin esperar a soltar el mouse.
        if hasattr(self.vista_principal, "slider_ruido"):
            self.vista_principal.slider_ruido.setTracking(True)
            self.vista_principal.slider_ruido.valueChanged.connect(self.modificar_ruido_senal)
            self.vista_principal.slider_ruido.sliderMoved.connect(self.modificar_ruido_senal)
        self.vista_principal.combo_scatter_x.currentIndexChanged.connect(self.actualizar_grafico_tabular)
        self.vista_principal.combo_scatter_y.currentIndexChanged.connect(self.actualizar_grafico_tabular)

    def aplicar_estilo_login(self):
        self.vista_login.btn_login.setStyleSheet("""
            QPushButton {
                background-color: #e3f2fd;
                color: #1565c0;
                border: 2px solid #bbdefb;
                border-radius: 6px;
                font-weight: bold;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #1565c0;
                color: white;
                border: 2px solid #0d47a1;
            }
            QPushButton:pressed {
                background-color: #0a2f66;
            }
        """)

    # ==============================
    # CANVAS EMBEBIDOS EN LA VISTA
    # ==============================
    def _asignar_layout_canvas(self, contenedor, canvas):
        layout = contenedor.layout()
        if layout is None:
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            contenedor.setLayout(layout)
        layout.addWidget(canvas)

    def inicializar_graficos(self):
        self.vista_principal.lbl_axial.setMinimumWidth(650)
        self.vista_principal.lbl_axial.setMinimumHeight(420)
        self.fig_img = Figure(figsize=(8.0, 7.0), dpi=100)
        self.canvas_img = FigureCanvas(self.fig_img)
        self.canvas_img.setFocusPolicy(Qt.StrongFocus)
        self.canvas_img.setMouseTracking(True)
        self.ax_axial = self.fig_img.add_subplot(221)
        self.ax_zoom = self.fig_img.add_subplot(222)
        self.ax_sagital = self.fig_img.add_subplot(223)
        self.ax_coronal = self.fig_img.add_subplot(224)
        self.fig_img.subplots_adjust(left=0.04, right=0.98, bottom=0.05, top=0.92, hspace=0.35, wspace=0.18)
        self._asignar_layout_canvas(self.vista_principal.lbl_axial, self.canvas_img)

        self.fig_senales = Figure(figsize=(7.0, 5.0), dpi=100)
        self.canvas_senales = FigureCanvas(self.fig_senales)
        self.ax_orig = self.fig_senales.add_subplot(221)
        self.ax_ruid = self.fig_senales.add_subplot(222)
        self.ax_prom = self.fig_senales.add_subplot(223)
        self.ax_std = self.fig_senales.add_subplot(224)
        self.fig_senales.subplots_adjust(left=0.08, right=0.95, bottom=0.10, top=0.88, hspace=0.48, wspace=0.28)
        self._asignar_layout_canvas(self.vista_principal.lbl_graficos_senales, self.canvas_senales)

        self.vista_principal.lbl_grafico_tabular.setMinimumWidth(550)
        self.vista_principal.lbl_grafico_tabular.setMinimumHeight(480)
        self.fig_tabular = Figure(figsize=(8.0, 6.0), dpi=100)
        self.canvas_tabular = FigureCanvas(self.fig_tabular)
        self.ax_tabular = [self.fig_tabular.add_subplot(2, 3, i + 1) for i in range(6)]
        self.fig_tabular.subplots_adjust(left=0.08, right=0.96, bottom=0.10, top=0.90, hspace=0.50, wspace=0.35)
        self._asignar_layout_canvas(self.vista_principal.lbl_grafico_tabular, self.canvas_tabular)

    # ===============================
    # CARGA INICIAL DE DATOS REALES
    # ===============================
    def cargar_datos_iniciales(self):
        print("\n--- Ejecutando Pipelines de Carga Física de Datos Médicos Reales ---")

        ruta_carpeta_dicom = os.path.join(self.base_dir, "dicom_folder", "Datos")
        print(f"[RUTA DICOM] {ruta_carpeta_dicom}")
        self.img.cargar_serie_dicom(ruta_carpeta_dicom)

        ruta_archivo_mat_real = os.path.join(self.base_dir, "heartbeat_data", "heartbeat_data", "000.mat")
        print(f"[RUTA MAT] {ruta_archivo_mat_real}")
        self.senales.cargar_archivo_mat_real(ruta_archivo_mat_real)

        ruta_archivo_csv_real = os.path.join(self.base_dir, "val_features.csv")
        print(f"[RUTA CSV] {ruta_archivo_csv_real}")
        df_tab = self.tab.cargar_datos_clinicos_simulados(ruta_archivo_csv_real)

        ruta_meta = os.path.join(self.base_dir, "metadatos_paciente.csv")
        self.img.exportar_metadatos_csv(ruta_meta)
        print(f"[IMG] Metadatos exportados a: {ruta_meta}")

        self.configurar_rangos_sliders()
        self.configurar_controles_tabulares(df_tab)
        self.cargar_tabla_dicom()

        self.senales.seleccionar_y_recortar_canal(numero_canal=0, p_inicial=0, p_final=500)

        if hasattr(self.vista_principal, "slider_ruido"):
            self.vista_principal.slider_ruido.setRange(0, 100)
            self.vista_principal.slider_ruido.setSingleStep(1)
            self.vista_principal.slider_ruido.setPageStep(10)
            self.vista_principal.slider_ruido.setTracking(True)
            self.vista_principal.slider_ruido.setValue(0)

        self.actualizar_imagenes()
        self.actualizar_graficos_senales()
        self.actualizar_resumen_tabular()
        self.actualizar_grafico_tabular()

    def configurar_rangos_sliders(self):
        if self.img.matriz_hu is None:
            return
        z, y, x = self.img.matriz_hu.shape
        self.vista_principal.slider_axial.setRange(0, z - 1)
        self.vista_principal.slider_sagital.setRange(0, y - 1)
        self.vista_principal.slider_coronal.setRange(0, x - 1)
        self.vista_principal.slider_axial.setValue(z // 2)
        self.vista_principal.slider_sagital.setValue(y // 2)
        self.vista_principal.slider_coronal.setValue(x // 2)

        if self.roi_center_row is None or self.roi_center_col is None:
            self.roi_center_row = y // 2
            self.roi_center_col = x // 2

    def configurar_controles_tabulares(self, df_tab):
        columnas = self.tab.columnas_numericas()
        if not columnas:
            columnas = list(df_tab.columns)

        self.vista_principal.combo_scatter_x.clear()
        self.vista_principal.combo_scatter_y.clear()
        self.vista_principal.combo_scatter_x.addItems(columnas)
        self.vista_principal.combo_scatter_y.addItems(columnas)

        if len(columnas) > 1:
            self.vista_principal.combo_scatter_y.setCurrentIndex(1)

    def cargar_tabla_dicom(self):
        meta = self.img.metadata_dict
        self.vista_principal.tabla_dicom.setRowCount(len(meta))
        self.vista_principal.tabla_dicom.setColumnCount(2)
        self.vista_principal.tabla_dicom.setHorizontalHeaderLabels(["Propiedad Médica", "Valor"])
        self.vista_principal.tabla_dicom.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        for idx, (k, v) in enumerate(meta.items()):
            self.vista_principal.tabla_dicom.setItem(idx, 0, QTableWidgetItem(str(k)))
            self.vista_principal.tabla_dicom.setItem(idx, 1, QTableWidgetItem(str(v)))

    # ============================
    # SEGURIDAD / LOGIN / CÁMARA
    # ============================
    def procesar_login(self):
        usuario = self.vista_login.txt_usuario.text()
        contra = self.vista_login.txt_password.text()
        if self.db.validar_usuario(usuario, contra):
            self.vista_login.close()
            nombre = self.db.usuario_actual.get("nombre", usuario)
            rol = self.db.usuario_actual.get("rol", "usuario")
            self.vista_principal.lbl_nombre_usuario.setText(f"Profesional: {nombre} | Rol: {rol}")
            self.vista_principal.show()
        else:
            if hasattr(self.vista_login, "mostrar_mensaje"):
                self.vista_login.mostrar_mensaje("Acceso denegado", "Usuario o contraseña incorrectos.", "error")

    def capturar_foto_usuario(self):
        ruta, fecha = self.db.capturar_foto_opencv()
        if ruta:
            self.vista_principal.lbl_fecha_foto.setText(f"Autenticado el: {fecha} | Foto: {ruta}")
            self.db.registrar_evento_en_mongo("Captura fotográfica de sesión mediante OpenCV")

    # =======================================================
    # IMÁGENES DICOM / HU / ZOOM / BINARIZACIÓN / MORFOLOGÍA
    # =======================================================
    def _obtener_umbral_desde_combo(self, texto):
        numeros = [int(s) for s in str(texto).replace("(", " ").replace(")", " ").split() if s.isdigit()]
        return numeros[0] if numeros else 128

    def _mostrar_imagen(self, ax, imagen, titulo, bgr=False, aspect='auto', binaria=False):
        ax.clear()
        if imagen is None:
            ax.set_title(titulo)
            ax.axis('off')
            return

        if len(imagen.shape) == 3:
            img = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB) if bgr else imagen
            ax.imshow(img, aspect=aspect)
        else:
            interpolacion = "nearest" if binaria else "bilinear"
            ax.imshow(
                imagen,
                cmap='gray',
                aspect=aspect,
                interpolation=interpolacion,
                vmin=0,
                vmax=255,
            )

        ax.set_title(titulo, fontsize=8, pad=8)
        ax.axis('off')

    def mover_roi_desde_click(self, event):
        """
        Intento opcional de mover la ROI con clic sobre el plano axial.
        Si el clic no responde por configuración de Qt/Matplotlib, la ROI se
        mueve de forma segura con los sliders sagital (Y) y coronal (X).
        """
        if event.inaxes != self.ax_axial:
            return
        if self.img.corte_actual_8bit is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        rows, cols = self.img.corte_actual_8bit.shape

        if event.button == 3:
            nueva_fila = rows // 2
            nueva_columna = cols // 2
        else:
            nueva_columna = max(0, min(int(event.xdata), cols - 1))
            nueva_fila = max(0, min(int(event.ydata), rows - 1))

        self.vista_principal.slider_sagital.setValue(nueva_fila)
        self.vista_principal.slider_coronal.setValue(nueva_columna)
        print(f"[ROI] Centro actualizado -> Y/Fila: {nueva_fila}, X/Columna: {nueva_columna}")

    def _redimensionar_como_zoom(self, imagen):
        if imagen is None:
            return None
        if self.img.recorte_zoom_resized is None:
            return imagen
        alto, ancho = self.img.recorte_zoom_resized.shape[:2]
        return cv2.resize(imagen.astype(np.uint8), (ancho, alto), interpolation=cv2.INTER_NEAREST)

    def actualizar_imagenes(self):
        if self.img.matriz_hu is None:
            return

        idx_axial = self.vista_principal.slider_axial.value()
        idx_sagital = self.vista_principal.slider_sagital.value()
        idx_coronal = self.vista_principal.slider_coronal.value()

        z, y, x = self.img.matriz_hu.shape
        proporcion_z = y / z if z > 0 else 1

        corte_axial = self.img.obtener_corte_normalizado_8bit('axial', idx_axial)
        corte_sagital = self.img.obtener_corte_normalizado_8bit('sagital', idx_sagital)
        corte_coronal = self.img.obtener_corte_normalizado_8bit('coronal', idx_coronal)

        if corte_axial is None:
            return


        self.roi_center_row = max(0, min(idx_sagital, corte_axial.shape[0] - 1))
        self.roi_center_col = max(0, min(idx_coronal, corte_axial.shape[1] - 1))

        img_caja, recorte_zoom = self.img.obtener_recorte_zoom_mm(
            self.roi_center_row,
            self.roi_center_col,
            size_mm=self.roi_size_mm,
        )

        tipo_bin = self.vista_principal.combo_binarizacion.currentText()
        tipo_morf = self.vista_principal.combo_morfologia.currentText()
        tam_kernel = self.vista_principal.spin_kernel.value()
        umbral = self._obtener_umbral_desde_combo(tipo_bin)

        base_roi = self.img.recorte_zoom_8bit
        imagen_procesada = recorte_zoom
        es_binaria = False

        if tipo_bin and "ninguno" not in tipo_bin.lower():
            procesada = self.img.segmentar_imagen(tipo=tipo_bin, umbral=umbral, imagen_base=base_roi)
            es_binaria = "binario" in tipo_bin.lower() or "otsu" in tipo_bin.lower()
        else:
            procesada = base_roi

        if tipo_morf and "ninguno" not in tipo_morf.lower():
            procesada = self.img.aplicar_morfologia(tipo_morf, tam_kernel, imagen_base=procesada)
            es_binaria = es_binaria or tipo_morf.lower() in ("erosión", "erosion", "dilatación", "dilatacion", "apertura", "cierre")

        if procesada is not None:
            imagen_procesada = self._redimensionar_como_zoom(procesada)

        titulo_axial = (
            f"Axial con ROI navegable {self.roi_size_mm} mm (Z: {idx_axial})"
            f"\nROI Y={self.roi_center_row} / X={self.roi_center_col} | Mueve sliders Sagital(Y) y Coronal(X)"
        )
        titulo_zoom = f"Zoom ROI / Filtro: {tipo_bin} + {tipo_morf}"

        self._mostrar_imagen(self.ax_axial, img_caja, titulo_axial, bgr=True, aspect='equal')
        self._mostrar_imagen(self.ax_zoom, imagen_procesada, titulo_zoom, aspect='equal', binaria=es_binaria)
        self._mostrar_imagen(self.ax_sagital, corte_sagital, f"Sagital interactivo (Y: {idx_sagital})", aspect=proporcion_z)
        self._mostrar_imagen(self.ax_coronal, corte_coronal, f"Coronal interactivo (X: {idx_coronal})", aspect=proporcion_z)
        self.canvas_img.draw()

    # =========================
    # SEÑALES BIOMÉDICAS .MAT
    # =========================
    def obtener_eje_estadistico_desde_vista(self):
        posibles = [
            ("radio_eje_0", 0),
            ("radio_eje_1", 1),
            ("radio_eje_2", 2),
        ]
        for nombre, eje in posibles:
            if hasattr(self.vista_principal, nombre) and getattr(self.vista_principal, nombre).isChecked():
                return eje
        return 0

    def actualizar_graficos_senales(self):
        """
        Renderiza las señales biomédicas dentro de la interfaz.

        Corrección importante:
        - La señal modificada usa el MISMO rango Y que la original para que el usuario
          perciba el efecto real del ruido y no parezca una imagen estática por autoescala.
        - En la gráfica ruidosa se superpone la original como referencia.
        - El título muestra el porcentaje actual del slider y la desviación usada.
        """
        for ax in (self.ax_orig, self.ax_ruid, self.ax_prom, self.ax_std):
            ax.clear()

        original = self.senales.canal_seleccionado_datos
        ruidosa = self.senales.datos_ruidosos

        valor_slider = self.vista_principal.slider_ruido.value() if hasattr(self.vista_principal, "slider_ruido") else 0
        desviacion_usada = getattr(self, "desviacion_ruido_actual", 0.0)

        if original is not None:
            x = np.arange(len(original))
            self.ax_orig.plot(x, original, linewidth=1.0)
            self.ax_orig.set_title("Canal biomédico original")
            self.ax_orig.set_xlabel("Muestras")
            self.ax_orig.set_ylabel("Amplitud")
            self.ax_orig.grid(True, alpha=0.25)

            margen = max(1e-9, float(np.nanstd(original)) * 0.35)
            y_min = float(np.nanmin(original)) - margen
            y_max = float(np.nanmax(original)) + margen
            if y_max > y_min:
                self.ax_orig.set_ylim(y_min, y_max)
        else:
            y_min = y_max = None

        if ruidosa is not None:
            x = np.arange(len(ruidosa))
            if original is not None and len(original) == len(ruidosa):
                # Referencia original debajo de la señal modificada.
                self.ax_ruid.plot(x, original, linewidth=0.8, alpha=0.55, label="Original")
            self.ax_ruid.plot(x, ruidosa, linewidth=1.0, label="Con ruido")
            self.ax_ruid.set_title(f"Señal con ruido: {valor_slider}% | σ={desviacion_usada:.4g}")
            self.ax_ruid.set_xlabel("Muestras")
            self.ax_ruid.set_ylabel("Amplitud")
            self.ax_ruid.grid(True, alpha=0.25)
            self.ax_ruid.legend(loc="upper right", fontsize=7)
            if original is not None and y_max is not None and y_max > y_min:
                self.ax_ruid.set_ylim(y_min, y_max)

        eje = self.obtener_eje_estadistico_desde_vista()
        promedio, desviacion = self.senales.calcular_promedio_desviacion_eje(eje=eje)
        if promedio is not None and desviacion is not None:
            limite = min(250, len(promedio), len(desviacion))
            x = np.arange(limite)
            self.ax_prom.stem(x, promedio[:limite], basefmt=" ")
            self.ax_prom.set_title(f"Promedio sobre eje {eje}")
            self.ax_prom.set_xlabel("Índice")
            self.ax_prom.set_ylabel("Media")
            self.ax_prom.grid(True, alpha=0.25)

            self.ax_std.stem(x, desviacion[:limite], basefmt=" ")
            self.ax_std.set_title(f"Desviación estándar sobre eje {eje}")
            self.ax_std.set_xlabel("Índice")
            self.ax_std.set_ylabel("STD")
            self.ax_std.grid(True, alpha=0.25)

        self.fig_senales.tight_layout()
        self.canvas_senales.draw_idle()
        self.canvas_senales.flush_events()

    def modificar_ruido_senal(self, valor_slider=None):
        try:
            if valor_slider is None or not isinstance(valor_slider, int):
                valor_slider = self.vista_principal.slider_ruido.value()
            
            original = self.senales.canal_seleccionado_datos
            if original is None:
                return

            porcentaje = valor_slider / 100.0
            
            amplitud = float(np.nanmax(original) - np.nanmin(original))
            if amplitud == 0 or np.isnan(amplitud):
                amplitud = 1.0

            desviacion = porcentaje * 0.15 * amplitud
            self.desviacion_ruido_actual = desviacion

            if valor_slider == 0:
                self.senales.datos_ruidosos = original.copy()
            else:
                self.senales.agregar_ruido_gaussiano(desviacion)

            self.actualizar_graficos_senales()
            
        except Exception as e:
            print(f"[Error Señales] Problema al inyectar ruido: {e}")
    # =================
    # DATOS TABULARES
    # ==================
    def actualizar_resumen_tabular(self):
        resumen = self.tab.obtener_resumen_completo_para_qtable()
        self.vista_principal.tabla_resumen.setRowCount(len(resumen))
        self.vista_principal.tabla_resumen.setColumnCount(len(resumen.columns))
        self.vista_principal.tabla_resumen.setHorizontalHeaderLabels(list(resumen.columns))
        self.vista_principal.tabla_resumen.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        for i in range(len(resumen)):
            for j, col in enumerate(resumen.columns):
                valor = resumen.iloc[i, j]
                self.vista_principal.tabla_resumen.setItem(i, j, QTableWidgetItem(str(valor)))

    def actualizar_grafico_tabular(self):
        if self.tab.df is None:
            return

        columnas = self.tab.columnas_numericas()
        columnas_4 = columnas[:4]

        for ax in self.ax_tabular:
            ax.clear()

        for i, columna in enumerate(columnas_4):
            datos = self.tab.df[columna].dropna().values
            self.ax_tabular[i].plot(datos[:400])
            self.ax_tabular[i].set_title(f"Plot individual: {columna}", fontsize=8)
            self.ax_tabular[i].set_xlabel("Registro")
            self.ax_tabular[i].set_ylabel(columna)

        cx = self.vista_principal.combo_scatter_x.currentText()
        cy = self.vista_principal.combo_scatter_y.currentText()
        ax_scatter = self.ax_tabular[4]
        if cx and cy and cx in self.tab.df.columns and cy in self.tab.df.columns:
            df_xy = self.tab.df[[cx, cy]].dropna()
            ax_scatter.scatter(df_xy[cx], df_xy[cy], alpha=0.5)
            ax_scatter.set_xlabel(cx)
            ax_scatter.set_ylabel(cy)
            ax_scatter.set_title("Scatter dinámico", fontsize=8)

        self.ax_tabular[5].axis('off')
        self.ax_tabular[5].text(
            0.02,
            0.90,
            "Columnas graficadas:\n" + "\n".join(columnas_4),
            va="top",
            fontsize=9,
        )

        self.fig_tabular.tight_layout()
        self.canvas_tabular.draw()

    def actualizar_grafico_scatter(self):
        self.actualizar_grafico_tabular()

    # ==================
    # EXPORTACIÓN NIFTI
    # ==================
    def exportar_volumen_nifti(self):
        try:
            ruta_salida = os.path.join(self.base_dir, "volumen_diagnostico.nii")
            ruta = self.img.exportar_volumen_nifti(ruta_salida)
            print(f"[NIFTI] Éxito. Archivo guardado en: {ruta}")
            self.db.registrar_evento_en_mongo("Exportación de volumen DICOM a NIfTI")
        except ImportError:
            print("[NIFTI] Error: necesitas instalar la librería nibabel. Ejecuta: pip install nibabel")
        except Exception as e:
            print(f"[NIFTI] Error inesperado al exportar NIfTI: {e}")

    def guardar_imagen_zoom(self):
        import matplotlib.patches as patches
        
        if self.img.corte_actual_8bit is None or self.img.recorte_zoom_resized is None:
            QMessageBox.warning(self.vista_principal, "Error", "No hay imagen médica cargada para guardar.")
            return

        ruta_completa, _ = QFileDialog.getSaveFileName(
            self.vista_principal,
            "Guardar Reporte de Zoom (2 Subplots)",
            "reporte_zoom_medico.png",
            "Imágenes PNG (*.png);;Imágenes JPEG (*.jpg)"
        )

        if ruta_completa:
            try:
                fig = Figure(figsize=(10, 5))
                ax1 = fig.add_subplot(1, 2, 1)
                ax2 = fig.add_subplot(1, 2, 2)

                ax1.imshow(self.img.corte_actual_8bit, cmap='gray')
                ax1.set_title("Corte Axial Completo - Ubicación ROI")
                
                spacing_r, spacing_c = self.img.pixel_spacing
                alto_px = max(8, int(self.roi_size_mm / (spacing_r if spacing_r > 0 else 1.0)))
                ancho_px = max(8, int(self.roi_size_mm / (spacing_c if spacing_c > 0 else 1.0)))
                r_min = max(0, self.roi_center_row - alto_px // 2)
                c_min = max(0, self.roi_center_col - ancho_px // 2)

                rect = patches.Rectangle((c_min, r_min), ancho_px, alto_px, linewidth=2, edgecolor='red', facecolor='none')
                ax1.add_patch(rect)

                ax2.imshow(self.img.recorte_zoom_resized, cmap='gray')
                ax2.set_title("Región de Interés (Zoom)")

                fig.tight_layout()
                fig.savefig(ruta_completa, dpi=150)

                self.db.registrar_evento_en_mongo("Exportación de Zoom en 2 Subplots")
                QMessageBox.information(self.vista_principal, "Éxito", f"Reporte guardado correctamente en:\n{ruta_completa}")
            
            except Exception as e:
                QMessageBox.critical(self.vista_principal, "Error", f"Error al generar el PDF/Imagen: {e}")

    