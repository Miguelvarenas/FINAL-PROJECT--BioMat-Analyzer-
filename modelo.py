import os
import glob
import datetime

import cv2
import numpy as np
import pandas as pd
from pymongo import MongoClient
import scipy.io as sio

try:
    import pydicom
except ImportError:
    pydicom = None


# ===============
#BASE DE DATOS 
# ===============
class BaseDatos:
    def __init__(self):
        self.cliente = None
        self.db = None
        self.coleccion_usuarios = None
        self.coleccion_sesiones = None
        self.usuario_actual = None
        self.modo_respaldo = False
        self.conectar_mongodb()

    def conectar_mongodb(self):
        try:
            uri = "mongodb+srv://miguelvarenas:123456789Mv@cluster0.ufwy5lm.mongodb.net/?appName=Cluster0"
            self.cliente = MongoClient(
                uri,
                serverSelectionTimeoutMS=6000,
                tls=True,
                tlsAllowInvalidCertificates=True,
            )
            self.db = self.cliente["BioMed_Analyzer"]
            self.coleccion_usuarios = self.db["usuarios"]
            self.coleccion_sesiones = self.db["sesiones"]
            self.cliente.server_info()
            self.modo_respaldo = False
            print("[DB] Conexión exitosa a MongoDB Atlas.")
        except Exception as e:
            print(f"[DB] Advertencia: Modo Respaldo Local Activado ({e})")
            self.modo_respaldo = True

    def validar_usuario(self, username, password):
        if not self.modo_respaldo:
            try:
                usuario = self.coleccion_usuarios.find_one({"id": username, "password": password})
                if usuario:
                    self.usuario_actual = usuario
                    return True
            except Exception:
                pass

        if username == "admin" and password == "1234":
            self.usuario_actual = {
                "id": "admin",
                "nombre": "Estudiante Bioingeniería (Modo Respaldo)",
                "rol": "administrador",
            }
            return True
        return False

    def registrar_evento_en_mongo(self, descripcion_evento):
        if self.usuario_actual is None:
            return

        fecha_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log = {
            "id_usuario": self.usuario_actual["id"],
            "evento": descripcion_evento,
            "fecha": fecha_str,
        }

        if not self.modo_respaldo:
            try:
                self.db["auditoria"].insert_one(log)
            except Exception:
                print(f"[AUDITORÍA LOCAL]: {log}")
        else:
            print(f"[AUDITORÍA LOCAL - RESPALDO]: {log}")

    def capturar_foto_opencv(self):
        if self.usuario_actual is None:
            return "", ""

        captura = cv2.VideoCapture(0)
        fecha_actual = datetime.datetime.now()
        fecha_str = fecha_actual.strftime("%Y-%m-%d_%H-%M-%S")
        id_user = self.usuario_actual["id"]
        ruta_foto = f"foto_sesion_{id_user}_{fecha_str}.jpg"

        if not captura.isOpened():
            print("[OPENCV] Cámara no detectada. Se registra sesión sin captura real.")
            registro = {"id_usuario": id_user, "ruta_foto": "foto_simulada.jpg", "fecha_sesion": fecha_str}
            if not self.modo_respaldo:
                try:
                    self.coleccion_sesiones.insert_one(registro)
                except Exception:
                    pass
            return "foto_simulada.jpg", fecha_str

        ret, frame = captura.read()
        captura.release()

        if ret:
            cv2.imwrite(ruta_foto, frame)
            registro = {"id_usuario": id_user, "ruta_foto": ruta_foto, "fecha_sesion": fecha_str}
            if not self.modo_respaldo:
                try:
                    self.coleccion_sesiones.insert_one(registro)
                except Exception:
                    pass
            return ruta_foto, fecha_str

        return "", ""


# ==================================================
#  PROCESAMIENTO DE IMÁGENES MÉDICAS DICOM / NIFTI
# ==================================================
class ProcesamientoImagenes:
    def __init__(self, db_modulo=None):
        self.matriz_3d = None
        self.matriz_hu = None
        self.corte_actual_8bit = None
        self.recorte_zoom_8bit = None
        self.recorte_zoom_resized = None
        self.metadata_dict = {}
        self.pixel_spacing = (1.0, 1.0)
        self.slice_thickness = 1.0
        self.db = db_modulo

    def _buscar_archivos_dicom(self, ruta_carpeta):
        archivos = []
        for patron in ("*.dcm", "*.DCM"):
            archivos.extend(glob.glob(os.path.join(ruta_carpeta, "**", patron), recursive=True))
        return sorted(set(archivos))

    def _parsear_hora_dicom(self, hora_dicom):
        if not hora_dicom:
            return None
        texto = str(hora_dicom).split(".")[0].zfill(6)
        try:
            return datetime.datetime.strptime(texto[:6], "%H%M%S")
        except Exception:
            return None

    def _calcular_duracion_estudio(self, study_time, series_time):
        inicio = self._parsear_hora_dicom(study_time)
        fin = self._parsear_hora_dicom(series_time)
        if inicio is None or fin is None:
            return "No disponible"
        if fin < inicio:
            fin += datetime.timedelta(days=1)
        return str(fin - inicio)

    def cargar_serie_dicom(self, ruta_carpeta):
        """Carga una serie DICOM real, reconstruye volumen 3D y extrae metadatos clínicos."""
        print(f"[IMG] Intentando cargar volumen DICOM real desde: {ruta_carpeta}")

        if pydicom is None:
            print("[ALERTA IMG] Falta instalar pydicom. Ejecuta: pip install pydicom")
            return self._crear_respaldo_sintetico()

        archivos_dicom = self._buscar_archivos_dicom(ruta_carpeta)
        if not archivos_dicom:
            print("[ALERTA IMG] No se encontraron archivos .dcm/.DCM. Activando respaldo sintético...")
            return self._crear_respaldo_sintetico()

        try:
            slices = [pydicom.dcmread(archivo) for archivo in archivos_dicom]
            slices.sort(
                key=lambda x: float(getattr(x, "SliceLocation", getattr(x, "InstanceNumber", 0)))
            )

            primer_corte = slices[0]
            pixel_spacing = getattr(primer_corte, "PixelSpacing", [1.0, 1.0])
            self.pixel_spacing = (float(pixel_spacing[0]), float(pixel_spacing[1]))
            self.slice_thickness = float(getattr(primer_corte, "SliceThickness", 1.0))

            study_time = str(getattr(primer_corte, "StudyTime", ""))
            series_time = str(getattr(primer_corte, "SeriesTime", ""))
            duracion = self._calcular_duracion_estudio(study_time, series_time)

            self.metadata_dict = {
                "PatientID": str(getattr(primer_corte, "PatientID", "ANON-001")),
                "PatientName": str(getattr(primer_corte, "PatientName", "Paciente Anónimo")),
                "StudyDate": str(getattr(primer_corte, "StudyDate", "No disponible")),
                "StudyTime": study_time if study_time else "No disponible",
                "StudyModality": str(getattr(primer_corte, "Modality", "No disponible")),
                "StudyDescription": str(getattr(primer_corte, "StudyDescription", "No disponible")),
                "SeriesTime": series_time if series_time else "No disponible",
                "DuracionEstudio": duracion,
                "Manufacturer": str(getattr(primer_corte, "Manufacturer", "No disponible")),
                "RescaleSlope": str(getattr(primer_corte, "RescaleSlope", "1.0")),
                "RescaleIntercept": str(getattr(primer_corte, "RescaleIntercept", "-1024.0")),
                "PixelSpacing_mm": f"{self.pixel_spacing[0]} x {self.pixel_spacing[1]}",
                "SliceThickness_mm": str(self.slice_thickness),
                "Dimensions": f"{len(slices)} x {primer_corte.Rows} x {primer_corte.Columns}",
            }

            self.matriz_3d = np.stack([corte.pixel_array for corte in slices]).astype(np.int16)
            print(f"[IMG] ¡Éxito! Volumen médico 3D estructurado correctamente. Dimensiones: {self.matriz_3d.shape}")
            self.conversion_hounsfield()

            if self.db:
                self.db.registrar_evento_en_mongo("Carga exitosa de volumen DICOM real")

            return self.matriz_3d

        except Exception as e:
            print(f"[ERROR IMG] Fallo al procesar DICOM real: {e}. Activando respaldo sintético...")
            return self._crear_respaldo_sintetico()

    def _crear_respaldo_sintetico(self):
        self.matriz_3d = np.random.randint(10, 1200, (64, 512, 512), dtype=np.int16)
        self.pixel_spacing = (1.0, 1.0)
        self.slice_thickness = 1.0
        self.metadata_dict = {
            "PatientID": "UdeA-2026-BIO",
            "PatientName": "Paciente Anónimo Bio (Simulado)",
            "StudyDate": "20260607",
            "StudyTime": "120000",
            "StudyModality": "CT",
            "StudyDescription": "Tomografía simulada de respaldo",
            "SeriesTime": "120300",
            "DuracionEstudio": "0:03:00",
            "Manufacturer": "SIMULADO",
            "RescaleSlope": "1.0",
            "RescaleIntercept": "-1024.0",
            "PixelSpacing_mm": "1.0 x 1.0",
            "SliceThickness_mm": "1.0",
            "Dimensions": "64 x 512 x 512",
        }
        self.conversion_hounsfield()
        return self.matriz_3d

    def conversion_hounsfield(self):
        if self.matriz_3d is None:
            return None
        slope = float(self.metadata_dict.get("RescaleSlope", 1.0))
        intercept = float(self.metadata_dict.get("RescaleIntercept", -1024.0))
        self.matriz_hu = self.matriz_3d.astype(np.float32) * slope + intercept
        return self.matriz_hu

    def exportar_metadatos_csv(self, ruta_salida="metadatos_paciente.csv"):
        df = pd.DataFrame(list(self.metadata_dict.items()), columns=["Propiedad Médica", "Valor"])
        df.to_csv(ruta_salida, index=False, encoding="utf-8-sig")
        return ruta_salida

    def exportar_volumen_nifti(self, ruta_salida="volumen_diagnostico.nii"):
        if self.matriz_hu is None:
            raise ValueError("No hay matriz HU cargada para exportar a NIfTI.")
        try:
            import nibabel as nib
        except ImportError as exc:
            raise ImportError("Necesitas instalar nibabel: pip install nibabel") from exc

        affine = np.eye(4)
        affine[0, 0] = self.pixel_spacing[1]
        affine[1, 1] = self.pixel_spacing[0]
        affine[2, 2] = self.slice_thickness
        nifti_img = nib.Nifti1Image(self.matriz_hu.astype(np.float32), affine)
        nib.save(nifti_img, ruta_salida)
        return ruta_salida

    def obtener_corte_normalizado_8bit(self, plano, indice):
        if self.matriz_hu is None:
            return None

        eje = 0 if plano == "axial" else (1 if plano == "sagital" else 2)
        idx = max(0, min(int(indice), self.matriz_hu.shape[eje] - 1))

        if plano == "axial":
            corte = self.matriz_hu[idx, :, :]
        elif plano == "sagital":
            corte = self.matriz_hu[:, idx, :]
        else:
            corte = self.matriz_hu[:, :, idx]

        corte_8 = self.normalizar_imagen_uint8(corte)
        if plano == "axial":
            self.corte_actual_8bit = corte_8
        return corte_8

    def normalizar_imagen_uint8(self, matriz):
        matriz = np.asarray(matriz, dtype=np.float32)
        c_min, c_max = np.min(matriz), np.max(matriz)
        if c_max - c_min == 0:
            return np.zeros(matriz.shape, dtype=np.uint8)
        return (255 * (matriz - c_min) / (c_max - c_min)).astype(np.uint8)

    def obtener_recorte_zoom_mm(
        self,
        center_row=None,
        center_col=None,
        size_mm=75,
        resize_factor=2.2,
        ruta_guardado=None,
    ):
        """
        Recorta una ROI navegable desde el corte axial activo.

        - La ROI sale de la matriz axial normalizada a uint8.
        - Dibuja el recuadro en BGR usando OpenCV.
        - Escribe las dimensiones en milímetros dentro del corte.
        - Redimensiona el recorte con cv2.resize para cumplir el zoom.
        """
        if self.corte_actual_8bit is None:
            return None, None

        rows, cols = self.corte_actual_8bit.shape
        center_row = rows // 2 if center_row is None else int(center_row)
        center_col = cols // 2 if center_col is None else int(center_col)

        center_row = max(0, min(center_row, rows - 1))
        center_col = max(0, min(center_col, cols - 1))

        spacing_row, spacing_col = self.pixel_spacing
        spacing_row = spacing_row if spacing_row > 0 else 1.0
        spacing_col = spacing_col if spacing_col > 0 else 1.0

        alto_px = max(8, int(size_mm / spacing_row))
        ancho_px = max(8, int(size_mm / spacing_col))
        half_h = alto_px // 2
        half_w = ancho_px // 2

        r_min = max(0, center_row - half_h)
        r_max = min(rows, center_row + half_h)
        c_min = max(0, center_col - half_w)
        c_max = min(cols, center_col + half_w)

        if (r_max - r_min) < alto_px:
            if r_min == 0:
                r_max = min(rows, alto_px)
            elif r_max == rows:
                r_min = max(0, rows - alto_px)
        if (c_max - c_min) < ancho_px:
            if c_min == 0:
                c_max = min(cols, ancho_px)
            elif c_max == cols:
                c_min = max(0, cols - ancho_px)

        self.recorte_zoom_8bit = self.corte_actual_8bit[r_min:r_max, c_min:c_max].copy()

        img_caja = cv2.cvtColor(self.corte_actual_8bit, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_caja, (c_min, r_min), (c_max, r_max), (0, 255, 0), 2)

        texto = f"ROI {size_mm:.0f} x {size_mm:.0f} mm | Espesor {self.slice_thickness:.1f} mm"
        y_texto = r_min - 8 if r_min > 24 else r_max + 18
        y_texto = max(18, min(y_texto, rows - 8))
        cv2.putText(
            img_caja,
            texto,
            (max(5, c_min), y_texto),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        if self.recorte_zoom_8bit.size > 0:
            recorte_suave = cv2.GaussianBlur(self.recorte_zoom_8bit, (3, 3), 0)
            nuevo_ancho = max(1, int(recorte_suave.shape[1] * resize_factor))
            nuevo_alto = max(1, int(recorte_suave.shape[0] * resize_factor))
            self.recorte_zoom_resized = cv2.resize(
                recorte_suave,
                (nuevo_ancho, nuevo_alto),
                interpolation=cv2.INTER_CUBIC,
            )
            if ruta_guardado:
                cv2.imwrite(ruta_guardado, self.recorte_zoom_resized)
        else:
            self.recorte_zoom_resized = None

        return img_caja, self.recorte_zoom_resized

    def _mejorar_contraste_roi(self, imagen_base):
        """Aplica CLAHE para que la ROI sea más legible antes de segmentar."""
        if imagen_base is None:
            return None
        img = imagen_base.astype(np.uint8)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def segmentar_imagen(self, tipo="Binario", umbral=128, imagen_base=None):
        """
        Segmentación OpenCV sobre el recorte DICOM.
        Incluye binario, binario invertido, truncado, ToZero, ToZero invertido y Otsu.
        Si el resultado sale casi negro/blanco, cambia automáticamente a Otsu para
        que el usuario pueda apreciar la región segmentada durante la sustentación.
        """
        base_img = imagen_base if imagen_base is not None else (
            self.recorte_zoom_8bit if self.recorte_zoom_8bit is not None else self.corte_actual_8bit
        )
        if base_img is None:
            return None

        base_img = self._mejorar_contraste_roi(base_img)
        tipo_normalizado = str(tipo).lower()

        if "otsu" in tipo_normalizado:
            _, img_bin = cv2.threshold(base_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return img_bin

        if "binario invertido" in tipo_normalizado or "binary inv" in tipo_normalizado:
            flag = cv2.THRESH_BINARY_INV
        elif "tozero invertido" in tipo_normalizado or "to zero invertido" in tipo_normalizado:
            flag = cv2.THRESH_TOZERO_INV
        elif "tozero" in tipo_normalizado or "to zero" in tipo_normalizado:
            flag = cv2.THRESH_TOZERO
        elif "truncado" in tipo_normalizado or "trunc" in tipo_normalizado:
            flag = cv2.THRESH_TRUNC
        elif "binario" in tipo_normalizado:
            flag = cv2.THRESH_BINARY
        else:
            return base_img

        _, img_bin = cv2.threshold(base_img, int(umbral), 255, flag)

        if flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            blancos = np.count_nonzero(img_bin) / img_bin.size
            if blancos < 0.02 or blancos > 0.98:
                _, img_bin = cv2.threshold(base_img, 0, 255, flag + cv2.THRESH_OTSU)

        return img_bin

    def aplicar_morfologia(self, operacion="erosion", kernel_size=3, imagen_base=None):
        base_img = imagen_base if imagen_base is not None else (
            self.recorte_zoom_8bit if self.recorte_zoom_8bit is not None else self.corte_actual_8bit
        )
        if base_img is None:
            return None

        base_img = base_img.astype(np.uint8)

        k = int(kernel_size)
        if k < 1:
            k = 1
        if k % 2 == 0:
            k += 1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        op = str(operacion).lower()

        if op in ("erosion", "erosión"):
            return cv2.erode(base_img, kernel, iterations=1)
        if op in ("dilatacion", "dilatación"):
            return cv2.dilate(base_img, kernel, iterations=1)
        if op == "apertura":
            return cv2.morphologyEx(base_img, cv2.MORPH_OPEN, kernel)
        if op == "cierre":
            return cv2.morphologyEx(base_img, cv2.MORPH_CLOSE, kernel)
        if op == "gradiente":
            return cv2.morphologyEx(base_img, cv2.MORPH_GRADIENT, kernel)
        if op in ("tophat", "top hat", "sombrero blanco"):
            return cv2.morphologyEx(base_img, cv2.MORPH_TOPHAT, kernel)
        if op in ("blackhat", "black hat", "sombrero negro"):
            return cv2.morphologyEx(base_img, cv2.MORPH_BLACKHAT, kernel)
        return base_img


# ===================================================
# PROCESAMIENTO DE SEÑALES MULTIDIMENSIONALES (.MAT)
# ===================================================
class ProcesamientoSenales:
    def __init__(self):
        self.matriz_3d = None
        self.matriz_2d_canales = None
        self.canal_seleccionado_datos = None
        self.datos_ruidosos = None

    def cargar_archivo_mat_real(self, ruta_archivo):
        print(f"[SEÑALES] Intentando cargar archivo real desde: {ruta_archivo}")
        try:
            mat_dict = sio.loadmat(ruta_archivo)
            nombre_variable = None
            for llave, valor in mat_dict.items():
                if not llave.startswith("__") and isinstance(valor, np.ndarray):
                    nombre_variable = llave
                    break

            if nombre_variable is None:
                raise ValueError("No se encontró una matriz numérica utilizable en el .mat")

            datos = mat_dict[nombre_variable]
            print(f"[SEÑALES] Variable detectada: '{nombre_variable}' con dimensiones originales {datos.shape}")
            datos_limpios = np.squeeze(datos)
            print(f"[SEÑALES] Dimensiones corregidas y aplanadas para interfaz: {datos_limpios.shape}")

            if len(datos_limpios.shape) == 3:
                self.matriz_3d = datos_limpios
                self.matriz_2d_canales = datos_limpios.reshape(-1, datos_limpios.shape[-1])
            elif len(datos_limpios.shape) == 2:
                self.matriz_2d_canales = datos_limpios
                self.matriz_3d = np.expand_dims(datos_limpios, axis=0)
            else:
                self.matriz_2d_canales = np.expand_dims(datos_limpios, axis=0)
                self.matriz_3d = np.expand_dims(self.matriz_2d_canales, axis=0)

            return self.matriz_2d_canales

        except Exception as e:
            print(f"[ALERTA SEÑALES] Error en archivo real ({e}). Activando simulación de respaldo...")
            self.matriz_3d = np.random.randn(3, 10, 1000) * 5.0
            self.matriz_2d_canales = self.matriz_3d.reshape(30, 1000)
            return self.matriz_2d_canales

    def seleccionar_y_recortar_canal(self, numero_canal=0, p_inicial=0, p_final=500):
        if self.matriz_2d_canales is None:
            return None
        idx_canal = max(0, min(int(numero_canal), self.matriz_2d_canales.shape[0] - 1))
        datos_completos = self.matriz_2d_canales[idx_canal, :]
        ini = max(0, min(int(p_inicial), len(datos_completos) - 2))
        fin = max(ini + 1, min(int(p_final), len(datos_completos)))
        self.canal_seleccionado_datos = datos_completos[ini:fin]
        self.datos_ruidosos = self.canal_seleccionado_datos.copy()
        return self.canal_seleccionado_datos

    def agregar_ruido_gaussiano(self, nivel_ruido):
        """
        Aplica ruido gaussiano de forma proporcional a la amplitud 
        real de la señal analógica cargada.
        """
        if self.datos_ruidosos is None:
            return

        std_senal = np.std(self.datos_ruidosos)

        factor_maximo = 0.25
        
        sigma_proporcional = (nivel_ruido / 100.0) * std_senal * factor_maximo

        ruido = np.random.normal(0, sigma_proporcional, len(self.datos_ruidosos))

        self.senal_procesada = self.datos_ruidosos + ruido

    def calcular_promedio_desviacion_eje(self, eje=0):
        if self.matriz_3d is None:
            return None, None
        eje = max(0, min(int(eje), self.matriz_3d.ndim - 1))
        promedio = np.mean(self.matriz_3d, axis=eje)
        desviacion = np.std(self.matriz_3d, axis=eje)
        return np.ravel(promedio), np.ravel(desviacion)
    


# ======================================
#  DATOS TABULARES CLÍNICOS CSV / EXCEL
# ======================================
class ProcesamientoTabular:
    def __init__(self):
        self.df = None

    def cargar_datos_clinicos_simulados(self, ruta_csv=None):
        if ruta_csv and os.path.exists(ruta_csv):
            print(f"[TABULAR] ¡Éxito! Cargando archivo clínico real desde: {ruta_csv}")
            try:
                extension = os.path.splitext(ruta_csv)[1].lower()
                if extension in (".xlsx", ".xls"):
                    self.df = pd.read_excel(ruta_csv)
                else:
                    self.df = pd.read_csv(ruta_csv)
                return self.df
            except Exception as e:
                print(f"[ERROR TABULAR] No se pudo leer el archivo ({e}). Pasando a simulación...")

        print("[TABULAR] Archivo no encontrado o corrupto. Activando simulación de variables clínicas...")
        data = {
            "Edad": np.random.randint(18, 85, size=100),
            "Presion_Sistolica": np.random.randint(110, 160, size=100),
            "Colesterol_mg_dl": np.random.randint(150, 280, size=100),
            "Frecuencia_Cardiaca": np.random.randint(60, 100, size=100),
            "Glucosa_Basal": np.random.randint(70, 140, size=100),
        }
        self.df = pd.DataFrame(data)
        return self.df

    def columnas_numericas(self):
        if self.df is None:
            return []
        return list(self.df.select_dtypes(include=[np.number]).columns)

    def obtener_resumen_describe(self):
        if self.df is None:
            return None
        return self.df.describe(include="all")

    def obtener_resumen_info_simulado(self):
        if self.df is None:
            return []
        info_list = []
        for col in self.df.columns:
            info_list.append({
                "Columna": col,
                "Tipo de Dato": str(self.df[col].dtype),
                "No Nulos": str(self.df[col].notnull().sum()),
                "Nulos": str(self.df[col].isnull().sum()),
            })
        return info_list

    """def obtener_resumen_completo_para_qtable(self):
        if self.df is None:
            return pd.DataFrame(columns=["Sección", "Columna", "Métrica", "Valor"])

        filas = []
        for info in self.obtener_resumen_info_simulado():
            filas.append({"Sección": "info()", "Columna": info["Columna"], "Métrica": "Tipo de Dato", "Valor": info["Tipo de Dato"]})
            filas.append({"Sección": "info()", "Columna": info["Columna"], "Métrica": "No Nulos", "Valor": info["No Nulos"]})
            filas.append({"Sección": "info()", "Columna": info["Columna"], "Métrica": "Nulos", "Valor": info["Nulos"]})

        describe = self.df.describe(include="all").fillna("")
        for metrica in describe.index:
            for columna in describe.columns:
                valor = describe.loc[metrica, columna]
                filas.append({"Sección": "describe()", "Columna": columna, "Métrica": metrica, "Valor": str(valor)})

        return pd.DataFrame(filas)"""
    
    def obtener_resumen_completo_para_qtable(self):
        if self.df is None:
            return pd.DataFrame(columns=["Columna", "Métrica", "Valor"])

        df_stats = self.df.describe().round(2) # Redondeamos a 2 decimales para que se vea limpio
        
        filas = []
        for columna in df_stats.columns:
            for metrica in df_stats.index:
                valor = df_stats.loc[metrica, columna]
                filas.append({
                    "Columna": columna, 
                    "Métrica": str(metrica).upper(), # Convertimos 'mean' a 'MEAN', 'std' a 'STD'
                    "Valor": str(valor)
                })

        return pd.DataFrame(filas)
