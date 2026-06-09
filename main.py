import sys

from PyQt5.QtWidgets import QApplication

from controlador import Controlador
from modelo import BaseDatos, ProcesamientoImagenes, ProcesamientoSenales, ProcesamientoTabular
from vista import VentanaLogin, VentanaPrincipal


if __name__ == "__main__":
    app = QApplication(sys.argv)

    estilo_pastel = """
        QMainWindow {
            background-color: #F8FAFC;
        }
        QTabWidget::panel {
            border: 1px solid #E2E8F0;
            background: #FFFFFF;
            border-radius: 8px;
        }
        QTabBar::tab {
            background: #EDF2F7;
            color: #4A5568;
            padding: 10px 18px;
            border-top-left-radius: 6px;
            border-top-right-radius: 6px;
            margin-right: 4px;
            font-weight: bold;
        }
        QTabBar::tab:selected {
            background: #FFFFFF;
            color: #3182CE;
            border: 1px solid #E2E8F0;
            border-bottom-color: #FFFFFF;
        }
        QPushButton {
            background-color: #EBF8FF;
            color: #2B6CB0;
            border: 1px solid #BEE3F8;
            border-radius: 6px;
            padding: 7px 15px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #BEE3F8;
            color: #2C5282;
        }
        QSlider::groove:horizontal {
            border: 1px solid #E2E8F0;
            height: 6px;
            background: #EDF2F7;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #BEE3F8;
            border: 1px solid #90CDF4;
            width: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        QTableWidget {
            gridline-color: #E2E8F0;
            border: 1px solid #CBD5E0;
            background: #FFFFFF;
        }
        QHeaderView::section {
            background-color: #F7FAFC;
            color: #4A5568;
            padding: 4px;
            font-weight: bold;
            border: 1px solid #E2E8F0;
        }
    """
    app.setStyleSheet(estilo_pastel)

    m_db = BaseDatos()
    m_img = ProcesamientoImagenes(db_modulo=m_db)
    m_sen = ProcesamientoSenales()
    m_tab = ProcesamientoTabular()

    v_login = VentanaLogin()
    v_principal = VentanaPrincipal()

    controlador_sistema = Controlador(
        vista_login=v_login,
        vista_principal=v_principal,
        modelo_db=m_db,
        modelo_img=m_img,
        modelo_senales=m_sen,
        modelo_tab=m_tab,
    )

    v_login.show()
    sys.exit(app.exec_())
