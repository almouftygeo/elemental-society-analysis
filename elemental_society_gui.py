import sys, os, re, warnings, traceback
import numpy as np
import pandas as pd
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, QLabel,
    QPushButton, QFileDialog, QMessageBox, QListWidget, QListWidgetItem,
    QDialog, QDialogButtonBox, QTableWidget, QTableWidgetItem, QTextEdit,
    QComboBox, QSpinBox, QStackedWidget, QTabWidget, QHeaderView,
    QProgressBar, QFrame, QDoubleSpinBox, QCheckBox, QGroupBox, QLineEdit,
    QSplitter, QRadioButton, QButtonGroup, QAbstractItemView, QInputDialog
)
from PyQt5.QtCore import Qt, QSize, QRectF, QPointF, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPainter, QPen, QColor, QPainterPath, QPolygonF

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns

from scipy import stats as scipystats
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

APPNAME = "Elemental Society Analysis"
APPVERSION = " v1.0.0"
DEVELOPER = "Mohamed Almoufty Abdoalmoutleb Abaker"
EMAIL = "almoufty121@gmail.com"


MW = {
    "SiO2": 60.084, "TiO2": 79.866, "Al2O3": 101.961, "Fe2O3": 159.688, "FeO": 71.844,
    "MnO": 70.937, "MgO": 40.304, "CaO": 56.077, "Na2O": 61.979, "K2O": 94.196,
    "P2O5": 141.944, "H2O": 18.015, "CO2": 44.009
}

class Palette:
    PRIMARY = "#4F46E5"; PRIMARYDARK = "#4338CA"; PRIMARYLIGHT = "#EEF2FF"
    SUCCESS = "#059669"; WARNING = "#D97706"; DANGER = "#DC2626"; INFO = "#0284C7"
    SIDEBARBG = "#111827"; SIDEBARBGHOVER = "#1F2937"
    SIDEBARTEXT = "#9CA3AF"; SIDEBARTEXTACTIVE = "#FFFFFF"
    BG = "#F3F4F6"; SURFACE = "#FFFFFF"; BORDER = "#E5E7EB"
    TEXTPRIMARY = "#111827"; TEXTSECONDARY = "#6B7280"

class Icons:
    cache = {}
    
    @classmethod
    def get(cls, name, color=Palette.TEXTPRIMARY, size=20):
        key = (name, color, size)
        if key in cls.cache:
            return cls.cache[key]
        pix = QPixmap(size, size)
        pix.fill(Qt.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.Antialiasing)
        pen = QPen(QColor(color))
        pen.setWidthF(max(1.2, size/11.0))
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        m = size*0.15
        s = size-2*m
        draw = getattr(cls, f"draw{name}", lambda *a: None)
        draw(p, m, s, size, color)
        p.end()
        icon = QIcon(pix)
        cls.cache[key] = icon
        return icon

    @staticmethod
    def drawhome(p, m, s, sz, c):
        path = QPainterPath()
        path.moveTo(m, m+s*0.45)
        path.lineTo(sz/2, m)
        path.lineTo(m+s, m+s*0.45)
        p.drawPath(path)
        p.drawRect(QRectF(m+s*0.18, m+s*0.45, s*0.64, s*0.55))

    @staticmethod
    def drawupload(p, m, s, sz, c):
        cx = sz/2
        p.drawLine(QPointF(cx, m), QPointF(cx, m+s*0.6))
        path = QPainterPath()
        path.moveTo(QPointF(cx-s*0.25, m+s*0.22))
        path.lineTo(QPointF(cx, m))
        path.lineTo(QPointF(cx+s*0.25, m+s*0.22))
        p.drawPath(path)
        p.drawLine(QPointF(m, m+s*0.85), QPointF(m+s, m+s*0.85))

    @staticmethod
    def drawdownload(p, m, s, sz, c):
        cx = sz/2
        p.drawLine(QPointF(cx, m), QPointF(cx, m+s*0.6))
        path = QPainterPath()
        path.moveTo(QPointF(cx-s*0.25, m+s*0.38))
        path.lineTo(QPointF(cx, m+s*0.6))
        path.lineTo(QPointF(cx+s*0.25, m+s*0.38))
        p.drawPath(path)
        p.drawLine(QPointF(m, m+s*0.85), QPointF(m+s, m+s*0.85))

    @staticmethod
    def drawbroom(p, m, s, sz, c):
        p.drawLine(QPointF(m+s*0.2, m), QPointF(m+s*0.75, m+s*0.65))
        tri = QPolygonF([QPointF(m+s*0.6, m+s*0.55), QPointF(m+s, m+s*0.6), QPointF(m+s*0.7, m+s)])
        p.drawPolygon(tri)

    @staticmethod
    def drawchart(p, m, s, sz, c):
        base = m+s
        p.drawRect(QRectF(m, base-s*0.35, s*0.2, s*0.35))
        p.drawRect(QRectF(m+s*0.4, base-s*0.65, s*0.2, s*0.65))
        p.drawRect(QRectF(m+s*0.8-s*0.2, base-s*0.95, s*0.2, s*0.95))

    @staticmethod
    def drawflask(p, m, s, sz, c):
        neckw = s*0.2
        cx = sz/2
        p.drawLine(QPointF(cx-neckw/2, m), QPointF(cx-neckw/2, m+s*0.35))
        p.drawLine(QPointF(cx+neckw/2, m), QPointF(cx+neckw/2, m+s*0.35))
        body = QPainterPath()
        body.moveTo(cx-neckw/2, m+s*0.35)
        body.lineTo(m, m+s)
        body.lineTo(m+s, m+s)
        body.lineTo(cx+neckw/2, m+s*0.35)
        p.drawPath(body)

    @staticmethod
    def drawlist(p, m, s, sz, c):
        for i in range(3):
            y = m+s*0.15+i*0.35
            p.drawEllipse(QPointF(m+s*0.06, y), 1.2, 1.2)
            p.drawLine(QPointF(m+s*0.22, y), QPointF(m+s, y))

    @staticmethod
    def drawtrend(p, m, s, sz, c):
        path = QPainterPath()
        path.moveTo(m, m+s*0.8)
        path.lineTo(m+s*0.3, m+s*0.4)
        path.lineTo(m+s*0.55, m+s*0.6)
        path.lineTo(m+s, m)
        p.drawPath(path)

    @staticmethod
    def drawtrash(p, m, s, sz, c):
        p.drawLine(QPointF(m, m+s*0.2), QPointF(m+s, m+s*0.2))
        p.drawRect(QRectF(m+s*0.15, m+s*0.2, s*0.7, s*0.75))
        p.drawLine(QPointF(m+s*0.35, m+s*0.2), QPointF(m+s*0.4, m))
        p.drawLine(QPointF(m+s*0.65, m+s*0.2), QPointF(m+s*0.6, m))

    @staticmethod
    def drawrefresh(p, m, s, sz, c):
        rect = QRectF(m, m, s, s)
        p.drawArc(rect, 30*16, 300*16)
        p.drawLine(QPointF(m+s, m+s*0.1), QPointF(m+s*0.75, m))
        p.drawLine(QPointF(m+s, m+s*0.1), QPointF(m+s*0.85, m+s*0.35))

    @staticmethod
    def drawdot(p, m, s, sz, c):
        p.setBrush(QColor(c))
        p.drawEllipse(QRectF(m, m, s, s))
        p.setBrush(Qt.NoBrush)

    @staticmethod
    def drawhelp(p, m, s, sz, c):
        p.drawEllipse(QRectF(m, m, s, s))
        p.drawText(QRectF(m, m, s, s), Qt.AlignCenter, "?")

    @staticmethod
    def drawinfo(p, m, s, sz, c):
        p.drawEllipse(QRectF(m, m, s, s))
        p.drawText(QRectF(m, m, s, s), Qt.AlignCenter, "i")

    @staticmethod
    def drawdocument(p, m, s, sz, c):
        p.drawRect(QRectF(m, m, s*0.75, s))
        for i in range(3):
            y = m+s*0.3+i*0.22
            p.drawLine(QPointF(m+s*0.15, y), QPointF(m+s*0.6, y))

def styledbutton(text, iconname=None, kind="default", tooltip=None):
    btn = QPushButton(text)
    if iconname:
        col = "#FFFFFF" if kind in ("primary","success","danger","warning") else Palette.TEXTSECONDARY
        btn.setIcon(Icons.get(iconname, color=col, size=16))
        btn.setIconSize(QSize(16,16))
    if tooltip:
        btn.setToolTip(tooltip)
    btn.setProperty("kind", kind)
    btn.setCursor(Qt.PointingHandCursor)
    btn.setFont(QFont("Segoe UI", 10))
    return btn

# --- Dialogs ---
class PlotConfigDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Visualizations")
        layout = QVBoxLayout(self)
        self.cbs = {}
        for name in ["Histogram","PDF KDE","CDF","Box Plot","Q-Q Plot"]:
            cb = QCheckBox(name)
            cb.setChecked(name in ["Histogram","PDF KDE"])
            self.cbs[name] = cb
            layout.addWidget(cb)
        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
    def get_selection(self):
        return [k for k,v in self.cbs.items() if v.isChecked()]

class DataFilterDialog(QDialog):
    def __init__(self, columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Data")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Column:"))
        self.col_combo = QComboBox()
        self.col_combo.addItems(columns)
        layout.addWidget(self.col_combo)
        self.type_combo = QComboBox()
        self.type_combo.addItems(["Threshold","Outliers (IQR)","Percentile"])
        layout.addWidget(QLabel("Type:"))
        layout.addWidget(self.type_combo)
        self.val_input = QDoubleSpinBox()
        self.val_input.setRange(-999999,999999)
        self.val_input.setValue(95)
        layout.addWidget(QLabel("Value:"))
        layout.addWidget(self.val_input)
        self.op_combo = QComboBox()
        self.op_combo.addItems(["<",">","=="])
        layout.addWidget(QLabel("Operator:"))
        layout.addWidget(self.op_combo)
        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
    def get_filter_config(self):
        return self.col_combo.currentText(), self.type_combo.currentText(), self.val_input.value(), self.op_combo.currentText()

class AdvancedCleanDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Data Cleaning")
        layout = QVBoxLayout(self)
        self.cbs = {}
        for name,label in [("remove_spaces","Remove spaces"),("remove_special","Remove special chars"),
                           ("standardize_delim","Standardize delimiters"),("drop_null_rows","Drop null rows"),
                           ("drop_null_cols","Drop null cols >50%"),("fill_numeric","Fill numeric with median"),
                           ("fill_text","Fill text with 'Unknown'")]:
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.cbs[name] = cb
            layout.addWidget(cb)
        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

class DeleteDataDialog(QDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Delete Data")
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        t1 = QWidget()
        t1l = QVBoxLayout()
        self.rb_rows_range = QRadioButton("By range")
        self.rb_rows_condition = QRadioButton("By condition")
        self.rb_rows_all = QRadioButton("All rows")
        self.rb_rows_range.setChecked(True)
        for rb in [self.rb_rows_range,self.rb_rows_condition,self.rb_rows_all]:
            t1l.addWidget(rb)
        self.row_range = QLineEdit("0-10")
        t1l.addWidget(self.row_range)
        self.row_cond_col = QComboBox()
        self.row_cond_col.addItems(df.columns.tolist())
        self.row_cond_val = QLineEdit()
        t1l.addWidget(self.row_cond_col)
        t1l.addWidget(self.row_cond_val)
        t1.setLayout(t1l)
        t2 = QWidget()
        t2l = QVBoxLayout()
        self.col_list = QListWidget()
        self.col_list.addItems(df.columns.tolist())
        self.col_list.setSelectionMode(QAbstractItemView.MultiSelection)
        t2l.addWidget(self.col_list)
        t2.setLayout(t2l)
        t3 = QWidget()
        t3l = QVBoxLayout()
        self.cb_remove_from_app = QCheckBox("Completely remove")
        t3l.addWidget(self.cb_remove_from_app)
        t3.setLayout(t3l)
        self.tabs.addTab(t1,"Rows")
        self.tabs.addTab(t2,"Columns")
        self.tabs.addTab(t3,"Global")
        layout.addWidget(self.tabs)
        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

class DisplayControlDialog(QDialog):
    def __init__(self, total_rows, total_cols, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Display Control")
        layout = QVBoxLayout(self)
        g = QGridLayout()
        g.addWidget(QLabel("Rows:"),0,0)
        self.sb_rows = QSpinBox()
        self.sb_rows.setRange(10,max(10,total_rows))
        self.sb_rows.setValue(min(100,total_rows))
        g.addWidget(self.sb_rows,0,1)
        g.addWidget(QLabel("Cols:"),1,0)
        self.sb_cols = QSpinBox()
        self.sb_cols.setRange(1,max(1,total_cols))
        self.sb_cols.setValue(min(20,total_cols))
        g.addWidget(self.sb_cols,1,1)
        self.cb_full = QCheckBox("Show all")
        g.addWidget(self.cb_full,2,0,1,2)
        layout.addLayout(g)
        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

class IndexColumnDialog(QDialog):
    def __init__(self, columns, current_index=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Index & Active Columns")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Index Column:"))
        self.index_combo = QComboBox()
        self.index_combo.addItem("None")
        self.index_combo.addItems(columns)
        if current_index:
            idx = self.index_combo.findText(current_index)
            if idx >= 0:
                self.index_combo.setCurrentIndex(idx)
        layout.addWidget(self.index_combo)
        layout.addWidget(QLabel("Active Columns:"))
        self.active_list = QListWidget()
        self.active_list.addItems(columns)
        self.active_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for i in range(self.active_list.count()):
            self.active_list.item(i).setSelected(True)
        layout.addWidget(self.active_list)
        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

class ElementRatioDialog(QDialog):
    def __init__(self, numeric_cols, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Element Ratio")
        layout = QVBoxLayout(self)
        g = QGridLayout()
        g.addWidget(QLabel("Numerator:"),0,0)
        self.num_combo = QComboBox()
        self.num_combo.addItems(numeric_cols)
        g.addWidget(self.num_combo,0,1)
        g.addWidget(QLabel("Denominator:"),1,0)
        self.den_combo = QComboBox()
        self.den_combo.addItems(numeric_cols)
        g.addWidget(self.den_combo,1,1)
        g.addWidget(QLabel("Name:"),2,0)
        self.name_input = QLineEdit("Ratio")
        g.addWidget(self.name_input,2,1)
        layout.addLayout(g)
        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

class SVMDialog(QDialog):
    def __init__(self, df_columns, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SVM Classification")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Target Column (categorical):"))
        self.target_combo = QComboBox()
        self.target_combo.addItems(df_columns)
        layout.addWidget(self.target_combo)
        layout.addWidget(QLabel("Feature Columns (NUMERIC only):"))
        self.feature_list = QListWidget()
        self.feature_list.addItems(df_columns)
        self.feature_list.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.feature_list)
        g = QGridLayout()
        g.addWidget(QLabel("Kernel:"),0,0)
        self.kernel_combo = QComboBox()
        self.kernel_combo.addItems(["linear","poly"])
        g.addWidget(self.kernel_combo,0,1)
        g.addWidget(QLabel("C:"),1,0)
        self.c_spin = QDoubleSpinBox()
        self.c_spin.setRange(0.01,1000)
        self.c_spin.setValue(1.0)
        g.addWidget(self.c_spin,1,1)
        g.addWidget(QLabel("Degree:"),2,0)
        self.degree_spin = QSpinBox()
        self.degree_spin.setRange(2,10)
        self.degree_spin.setValue(3)
        g.addWidget(self.degree_spin,2,1)
        g.addWidget(QLabel("Test Size:"),3,0)
        self.test_size = QDoubleSpinBox()
        self.test_size.setRange(0.1,0.5)
        self.test_size.setValue(0.3)
        g.addWidget(self.test_size,3,1)
        layout.addLayout(g)
        btns = QDialogButtonBox(QDialogButtonBox.Ok|QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)
    def get_params(self):
        target = self.target_combo.currentText()
        features = [item.text() for item in self.feature_list.selectedItems()]
        if target in features:
            features.remove(target)
        return {'target':target,'features':features,'kernel':self.kernel_combo.currentText(),
                'C':self.c_spin.value(),'degree':self.degree_spin.value(),'test_size':self.test_size.value()}

# --- Data Logic ---
def normalize_column_name(c):
    if c is None:
        return "Unnamed"
    c = str(c).strip()
    c = re.sub(r"\s+"," ",c)
    return "Unnamed" if c=="" or c.lower() in ["nan","none","null"] else c

def clean_dataframe_advanced(df, options):
    if df is None or df.empty:
        return df, "No data"
    if options.get('remove_spaces',True):
        df.columns = df.columns.str.strip()
    if options.get('remove_special',True):
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = df[col].astype(str).str.replace(r'[;:""\'\']','',regex=True)
            except:
                pass
    if options.get('standardize_delim',True):
        for col in df.columns:
            try:
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(',','.')
                    df[col] = pd.to_numeric(df[col],errors='ignore')
            except:
                pass
    if options.get('drop_null_rows',False):
        df = df.dropna()
    if options.get('drop_null_cols',False):
        thresh = len(df)*0.5
        cols = df.columns[df.isnull().sum()>thresh]
        if len(cols)>0:
            df = df.drop(columns=cols)
    if options.get('fill_numeric',False):
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
    df = df.drop_duplicates()
    for col in df.columns:
        try:
            if not pd.api.types.is_numeric_dtype(df[col]):
                series = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.\-eE]','',regex=True),errors='coerce')
                if series.notna().sum()/max(len(series),1)>0.8:
                    df[col] = series
        except:
            pass
    return df, f"Cleaned: {df.shape}"

class EnhancedGeochemistryEngine:
    @staticmethod
    def calculate_alteration_indices(df, indices):
        results = {}
        def find_col(name):
            for col in df.columns:
                if col.lower().strip() == name.lower() or name.lower() in col.lower().strip():
                    return col
            return None
        al2o3_col = find_col("Al2O3")
        cao_col = find_col("CaO")
        na2o_col = find_col("Na2O")
        k2o_col = find_col("K2O")
        for idx in indices:
            try:
                if idx == "CIA" and all(c is not None for c in [al2o3_col,cao_col,na2o_col,k2o_col]):
                    al2o3 = pd.to_numeric(df[al2o3_col],errors='coerce')
                    cao = pd.to_numeric(df[cao_col],errors='coerce')
                    na2o = pd.to_numeric(df[na2o_col],errors='coerce')
                    k2o = pd.to_numeric(df[k2o_col],errors='coerce')
                    al_m = al2o3/MW["Al2O3"]; ca_m = cao/MW["CaO"]; na_m = na2o/MW["Na2O"]; k_m = k2o/MW["K2O"]
                    ca_star = np.minimum(ca_m,na_m)
                    results["CIA"] = round(float((al_m/(al_m+ca_star+na_m+k_m+1e-9)).mean()*100),2)
                elif idx == "CIA":
                    results["CIA"] = "Missing: Al2O3, CaO, Na2O, K2O"
                if idx == "CIW" and all(c is not None for c in [al2o3_col,cao_col,na2o_col]):
                    al2o3 = pd.to_numeric(df[al2o3_col],errors='coerce')
                    cao = pd.to_numeric(df[cao_col],errors='coerce')
                    na2o = pd.to_numeric(df[na2o_col],errors='coerce')
                    al_m = al2o3/MW["Al2O3"]; ca_m = cao/MW["CaO"]; na_m = na2o/MW["Na2O"]
                    results["CIW"] = round(float((al_m/(al_m+ca_m+na_m+1e-9)).mean()*100),2)
                elif idx == "CIW":
                    results["CIW"] = "Missing: Al2O3, CaO, Na2O"
            except Exception as e:
                results[f"{idx}_Error"] = str(e)
        return results
    
    @staticmethod
    def calculate_element_ratios(df, elements):
        results = {}
        def find_col(name):
            for col in df.columns:
                if col.lower().strip() == name.lower() or name.lower() in col.lower().strip():
                    return col
            return None
        std_ratios = [("K2O/Na2O","K2O","Na2O"),("Rb/Sr","Rb","Sr"),("Nb/Y","Nb","Y"),
                     ("Cu/Zn","Cu","Zn"),("Au/Ag","Au","Ag"),("Sr/Y","Sr","Y")]
        for name,num,den in std_ratios:
            try:
                nc = find_col(num); dc = find_col(den)
                if nc and dc:
                    r = pd.to_numeric(df[nc],errors='coerce')/(pd.to_numeric(df[dc],errors='coerce')+1e-9)
                    results[name] = round(float(r.mean()),3)
            except:
                pass
        return results
    
    @staticmethod
    def calculate_grade_statistics(df, elements, threshold=95):
        results = {}
        def find_col(name):
            for col in df.columns:
                if col.lower().strip() == name.lower() or name.lower() in col.lower().strip():
                    return col
            return None
        for el in elements:
            col = find_col(el)
            if col:
                data = pd.to_numeric(df[col],errors='coerce').dropna()
                if len(data) > 0:
                    thresh = data.quantile(threshold/100)
                    anomalies = data[data > thresh]
                    results[el] = {"N":len(data),"Mean":round(data.mean(),2),"Median":round(data.median(),2),
                                  "Std":round(data.std(),2),"Min":round(data.min(),2),"Max":round(data.max(),2),
                                  f"P{threshold}":round(thresh,2),"Anomalies":len(anomalies),
                                  "Anomaly%":round(len(anomalies)/len(data)*100,1)}
        return results

class EnhancedStatisticsEngine:
    @staticmethod
    def compute(df):
        if df is None or df.empty:
            return pd.DataFrame()
        nd = df.select_dtypes(include=[np.number])
        if nd.empty:
            return pd.DataFrame()
        rows = []
        for col in nd.columns:
            x = nd[col].dropna()
            if len(x) < 3:
                continue
            rows.append({"Variable":col,"N":len(x),"Mean":round(x.mean(),4),"Median":round(x.median(),4),
                        "Std":round(x.std(),4),"Min":round(x.min(),4),"Max":round(x.max(),4),
                        "Skewness":round(float(scipystats.skew(x)),4),
                        "Kurtosis":round(float(scipystats.kurtosis(x)),4)})
        return pd.DataFrame(rows)

# --- Main Application ---
class ElementalSocietyAnalysis(QMainWindow):
    def __init__(self):
        super().__init__()
        self.raw_df = None
        self.df = None
        self.filename = None
        self.last_geochem_data = {"alterationindices":{},"elementratios":{},"gradestats":{}}
        self.last_plot_config = ["Histogram","PDF KDE"]
        self.index_column = None
        self.active_columns = []
        self.display_rows = 100
        self.display_cols = 20
        self.show_all = False
        self.last_cluster_data = None
        self.svm_params = None
        self.last_svm_data = None
        self.last_analysis_report = ""
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle(f"{APPNAME} v{APPVERSION}")
        self.setGeometry(50,50,1600,950)
        self.setStyleSheet(self.get_stylesheet())
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0,0,0,0)
        self.stack = QStackedWidget()
        self.create_home_page()
        self.create_data_page()
        self.create_dist_page()
        self.create_analysis_page()
        self.create_geochem_page()
        self.create_clustering_page()
        self.create_svm_page()
        self.create_log_page()
        self.sidebar = self.build_sidebar()
        main_layout.addWidget(self.sidebar)
        right = QWidget()
        r_layout = QVBoxLayout(right)
        r_layout.setContentsMargins(0,0,0,0)
        self.toolbar = self.build_toolbar()
        r_layout.addWidget(self.toolbar)
        self.status_panel = self.build_status_panel()
        r_layout.addWidget(self.status_panel)
        r_layout.addWidget(self.stack)
        main_layout.addWidget(right)
        self.stack.setCurrentIndex(0)

    def log(self, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        if hasattr(self,'log_text'):
            self.log_text.append(f"[{ts}] {msg}")

    def set_status(self, msg, error=False):
        if hasattr(self,'status_lbl'):
            self.status_lbl.setText(msg)
            self.status_lbl.setStyleSheet(f"color:{Palette.DANGER if error else Palette.PRIMARY};font-weight:bold;")

    def get_analysis_df(self):
        if self.df is None:
            return None
        if not self.active_columns:
            return self.df.copy()
        cols = [c for c in self.active_columns if c in self.df.columns]
        if self.index_column and self.index_column in self.df.columns and self.index_column not in cols:
            cols.insert(0,self.index_column)
        return self.df[cols].copy() if cols else self.df.copy()

    def get_numeric_cols(self):
        if self.df is None:
            return []
        return self.df.select_dtypes(include=[np.number]).columns.tolist()

    def build_sidebar(self):
        w = QWidget()
        w.setFixedWidth(230)
        w.setObjectName("Sidebar")
        l = QVBoxLayout(w)
        l.setContentsMargins(0,0,0,0)
        brand = QLabel("Elemental\nSociety Pro")
        brand.setObjectName("BrandLabel")
        brand.setAlignment(Qt.AlignCenter)
        l.addWidget(brand)
        self.nav = QListWidget()
        self.nav.addItems(["Home","Dataset","Distributions","Analysis","Geochemistry","Clustering","SVM Classifier","Logs"])
        self.nav.setCurrentRow(0)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        l.addWidget(self.nav)
        l.addStretch()
        help_btn = QPushButton("  Help / About")
        help_btn.setIcon(Icons.get("help",color=Palette.SIDEBARTEXT,size=18))
        help_btn.setFlat(True)
        help_btn.setCursor(Qt.PointingHandCursor)
        help_btn.setStyleSheet(f"color:{Palette.SIDEBARTEXT};padding:10px;border:none;")
        help_btn.clicked.connect(self.show_help)
        l.addWidget(help_btn)
        ver = QLabel(f"v{APPVERSION}\n{DEVELOPER}")
        ver.setAlignment(Qt.AlignCenter)
        ver.setStyleSheet(f"color:{Palette.SIDEBARTEXT};padding:10px;font-size:10px;")
        l.addWidget(ver)
        return w

    def build_toolbar(self):
        w = QWidget()
        w.setObjectName("Toolbar")
        l = QHBoxLayout(w)
        l.setContentsMargins(10,5,10,5)
        self.btn_upload = styledbutton("Upload","upload","primary")
        self.btn_upload.clicked.connect(self.upload_file)
        self.btn_clean = styledbutton("Clean","broom","warning")
        self.btn_clean.clicked.connect(self.advanced_clean_data)
        self.btn_filter = styledbutton("Filter","list","default")
        self.btn_filter.clicked.connect(self.open_filter_dialog)
        self.btn_reset = styledbutton("Reset","refresh","default")
        self.btn_reset.clicked.connect(self.reset_data)
        l.addWidget(self.btn_upload); l.addWidget(self.btn_clean); l.addWidget(self.btn_filter); l.addWidget(self.btn_reset)
        sep = QFrame(); sep.setFrameShape(QFrame.VLine); l.addWidget(sep)
        self.btn_export_csv = styledbutton("Export Data","download","default")
        self.btn_export_csv.clicked.connect(self.export_data_csv)
        self.btn_export_fig = styledbutton("Export Fig","trend","success")
        self.btn_export_fig.clicked.connect(self.export_current_figure)
        self.btn_report = styledbutton("Report","document","info")
        self.btn_report.clicked.connect(self.generate_full_report)
        l.addWidget(self.btn_export_csv); l.addWidget(self.btn_export_fig); l.addWidget(self.btn_report)
        l.addStretch()
        return w

    def build_status_panel(self):
        w = QWidget()
        w.setObjectName("StatusPanel")
        l = QHBoxLayout(w)
        l.setContentsMargins(10,5,10,5)
        self.status_lbl = QLabel("Ready")
        l.addWidget(self.status_lbl)
        l.addStretch()
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.setVisible(False)
        l.addWidget(self.progress)
        return w

    def create_home_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        l.setAlignment(Qt.AlignCenter)
        card = QWidget()
        card.setStyleSheet("background:white;border-radius:12px;padding:40px;margin:20px;")
        cl = QVBoxLayout(card)
        title = QLabel(f"Welcome to {APPNAME}")
        title.setStyleSheet(f"font-size:32px;font-weight:bold;color:{Palette.PRIMARY};")
        title.setAlignment(Qt.AlignCenter)
        cl.addWidget(title)
        sub = QLabel("Geochemical  Analysis  App")
        cl.addWidget(sub)
        dev = QLabel(f"{DEVELOPER}\n{EMAIL}")
        dev.setAlignment(Qt.AlignCenter)
        cl.addWidget(dev)
        bl = QHBoxLayout()
        bl.setAlignment(Qt.AlignCenter)
        up = styledbutton("Upload Data File","upload","primary")
        up.setMinimumHeight(45)
        up.clicked.connect(self.upload_file)
        bl.addWidget(up)
        cl.addLayout(bl)
        l.addWidget(card)
        self.stack.addWidget(p)

    def create_data_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        t1 = QHBoxLayout()
        t1.addWidget(QLabel("View:"))
        self.combo_data_view = QComboBox()
        self.combo_data_view.addItems(["Working Data","Raw Data","Analysis Data"])
        self.combo_data_view.currentTextChanged.connect(self.switch_data_view)
        t1.addWidget(self.combo_data_view)
        t1.addStretch()
        self.btn_refresh_view = styledbutton("Refresh","refresh","default")
        self.btn_refresh_view.clicked.connect(lambda: self.update_table())
        self.btn_display = styledbutton("Display","chart","default")
        self.btn_display.clicked.connect(self.control_display)
        self.btn_index = styledbutton("Index/Active","list","default")
        self.btn_index.clicked.connect(self.set_index_columns)
        self.btn_rename = styledbutton("Rename","list","default")
        self.btn_rename.clicked.connect(self.rename_columns_dialog)
        self.btn_units = styledbutton("Units","refresh","default")
        self.btn_units.clicked.connect(self.unify_units)
        self.btn_delete = styledbutton("Delete","trash","danger")
        self.btn_delete.clicked.connect(self.delete_data)
        for btn in [self.btn_refresh_view,self.btn_display,self.btn_index,self.btn_rename,self.btn_units,self.btn_delete]:
            t1.addWidget(btn)
        l.addLayout(t1)
        t2 = QHBoxLayout()
        self.btn_element_ratio = styledbutton("Calculate Ratio","flask","info")
        self.btn_element_ratio.clicked.connect(self.calculate_element_ratio)
        t2.addWidget(self.btn_element_ratio)
        t2.addStretch()
        l.addLayout(t2)
        self.data_info = QLabel("No data")
        l.addWidget(self.data_info)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        l.addWidget(self.table)
        self.stack.addWidget(p)

    def create_dist_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        c = QHBoxLayout()
        c.addWidget(QLabel("Variable:"))
        self.dist_var = QComboBox()
        c.addWidget(self.dist_var)
        self.btn_plot_config = styledbutton("Config","chart","primary")
        self.btn_plot_config.clicked.connect(self.show_plot_config_safe)
        c.addWidget(self.btn_plot_config)
        self.btn_update_dist = styledbutton("Update","refresh","success")
        self.btn_update_dist.clicked.connect(self.update_distribution)
        c.addWidget(self.btn_update_dist)
        self.btn_export_dist = styledbutton("Export","download","info")
        self.btn_export_dist.clicked.connect(self.export_distribution_plot)
        c.addWidget(self.btn_export_dist)
        c.addStretch()
        l.addLayout(c)
        self.dist_fig = plt.Figure(figsize=(10,8))
        self.dist_canvas = FigureCanvas(self.dist_fig)
        l.addWidget(self.dist_canvas)
        self.stack.addWidget(p)

    def create_analysis_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        tabs = QTabWidget()
        t1 = QWidget()
        l1 = QVBoxLayout(t1)
        sc = QHBoxLayout()
        self.btn_refresh_stats = styledbutton("Refresh","refresh","primary")
        self.btn_refresh_stats.clicked.connect(self.update_statistics)
        self.btn_export_stats = styledbutton("Export","download","default")
        self.btn_export_stats.clicked.connect(self.export_statistics)
        sc.addWidget(self.btn_refresh_stats); sc.addWidget(self.btn_export_stats); sc.addStretch()
        l1.addLayout(sc)
        self.stats_table = QTableWidget()
        self.stats_table.setAlternatingRowColors(True)
        l1.addWidget(self.stats_table)
        tabs.addTab(t1,"Statistics")
        t2 = QWidget()
        l2 = QVBoxLayout(t2)
        c2 = QHBoxLayout()
        c2.addWidget(QLabel("Type:"))
        self.analysis_type = QComboBox()
        self.analysis_type.addItems(["Correlation Matrix","PCA Analysis","Scatter Plot"])
        c2.addWidget(self.analysis_type)
        self.cb_show_pca = QCheckBox("Show PCA Vectors")
        self.cb_show_pca.setChecked(True)
        c2.addWidget(self.cb_show_pca)
        c2.addWidget(QLabel("Vars:"))
        self.analysis_vars = QListWidget()
        self.analysis_vars.setMaximumHeight(80)
        self.analysis_vars.setSelectionMode(QAbstractItemView.MultiSelection)
        c2.addWidget(self.analysis_vars)
        self.btn_update_analysis = styledbutton("Update","refresh","success")
        self.btn_update_analysis.clicked.connect(self.update_analysis)
        c2.addWidget(self.btn_update_analysis)
        self.btn_export_analysis = styledbutton("Export","download","info")
        self.btn_export_analysis.clicked.connect(self.export_analysis_plot)
        c2.addWidget(self.btn_export_analysis)
        c2.addStretch()
        l2.addLayout(c2)
        self.ana_fig = plt.Figure(figsize=(12,8))
        self.ana_canvas = FigureCanvas(self.ana_fig)
        l2.addWidget(self.ana_canvas)
        self.ana_report = QTextEdit()
        self.ana_report.setReadOnly(True)
        self.ana_report.setMaximumHeight(150)
        l2.addWidget(self.ana_report)
        tabs.addTab(t2,"Plots")
        l.addWidget(tabs)
        self.stack.addWidget(p)

    def create_geochem_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        tabs = QTabWidget()
        t1 = QWidget()
        t1l = QVBoxLayout()
        h1 = QHBoxLayout()
        self.alteration_cbs = {}
        for idx in ["CIA","CIW","PIA","ICV","WIP"]:
            cb = QCheckBox(idx)
            cb.setChecked(idx=="CIA")
            self.alteration_cbs[idx] = cb
            h1.addWidget(cb)
        h1.addStretch()
        btn_alt = styledbutton("Run","refresh","primary")
        btn_alt.clicked.connect(self.calculate_alteration_only)
        h1.addWidget(btn_alt)
        t1l.addLayout(h1)
        self.alteration_view = QTextEdit()
        self.alteration_view.setReadOnly(True)
        t1l.addWidget(self.alteration_view)
        t1.setLayout(t1l)
        tabs.addTab(t1,"Alteration Indices")
        t2 = QWidget()
        t2l = QVBoxLayout()
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Elements:"))
        self.ratio_selector = QListWidget()
        self.ratio_selector.setMaximumHeight(80)
        self.ratio_selector.setSelectionMode(QAbstractItemView.MultiSelection)
        h2.addWidget(self.ratio_selector)
        btn_rat = styledbutton("Run","refresh","primary")
        btn_rat.clicked.connect(self.calculate_ratios_only)
        h2.addWidget(btn_rat)
        t2l.addLayout(h2)
        self.ratio_view = QTextEdit()
        self.ratio_view.setReadOnly(True)
        t2l.addWidget(self.ratio_view)
        t2.setLayout(t2l)
        tabs.addTab(t2,"Element Ratios")
        t3 = QWidget()
        t3l = QVBoxLayout()
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Threshold %:"))
        self.grade_threshold = QDoubleSpinBox()
        self.grade_threshold.setRange(50,99)
        self.grade_threshold.setValue(95)
        h3.addWidget(self.grade_threshold)
        h3.addWidget(QLabel("Elements:"))
        self.grade_elements_list = QListWidget()
        self.grade_elements_list.setMaximumHeight(80)
        self.grade_elements_list.addItems(["Au","Ag","Cu","Pb","Zn","Ni","Co","As"])
        self.grade_elements_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for i in range(5):
            self.grade_elements_list.item(i).setSelected(True)
        h3.addWidget(self.grade_elements_list)
        btn_grd = styledbutton("Run","refresh","primary")
        btn_grd.clicked.connect(self.calculate_grades_only)
        h3.addWidget(btn_grd)
        t3l.addLayout(h3)
        self.grade_view = QTextEdit()
        self.grade_view.setReadOnly(True)
        t3l.addWidget(self.grade_view)
        t3.setLayout(t3l)
        tabs.addTab(t3,"Grade Statistics")
        l.addWidget(tabs)
        exp = QHBoxLayout()
        btn_exp_all = styledbutton("Export All to Excel","download","success")
        btn_exp_all.clicked.connect(self.export_geochem_excel)
        exp.addWidget(btn_exp_all)
        exp.addStretch()
        l.addLayout(exp)
        self.stack.addWidget(p)

    def calculate_alteration_only(self):
        if self.df is None: return
        adf = self.get_analysis_df()
        if adf is None: return
        selected = [k for k,v in self.alteration_cbs.items() if v.isChecked()]
        if not selected: QMessageBox.warning(self,"Error","Select at least one index"); return
        try:
            results = EnhancedGeochemistryEngine.calculate_alteration_indices(adf,selected)
            self.last_geochem_data["alterationindices"] = results
            self.alteration_view.setHtml(self.dict_to_html("Alteration Indices",results))
            self.log("Alteration indices calculated")
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def calculate_ratios_only(self):
        if self.df is None: return
        adf = self.get_analysis_df()
        if adf is None: return
        try:
            results = EnhancedGeochemistryEngine.calculate_element_ratios(adf,[])
            self.last_geochem_data["elementratios"] = results
            self.ratio_view.setHtml(self.dict_to_html("Element Ratios",results))
            self.log(f"Ratios: {len(results)}")
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def calculate_grades_only(self):
        if self.df is None: return
        adf = self.get_analysis_df()
        if adf is None: return
        selected = [item.text() for item in self.grade_elements_list.selectedItems()]
        if not selected: QMessageBox.warning(self,"Error","Select at least one element"); return
        try:
            results = EnhancedGeochemistryEngine.calculate_grade_statistics(adf,selected,self.grade_threshold.value())
            self.last_geochem_data["gradestats"] = results
            self.grade_view.setHtml(self.dict_to_html("Grade Statistics",results,True))
            self.log("Grade stats calculated")
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def dict_to_html(self, title, data, is_nested=False):
        h = [f'<h3 style="color:{Palette.PRIMARY};">{title}</h3>']
        if not data: h.append("<p>No data. Click Run to calculate.</p>"); return "".join(h)
        h.append('<table border="1" style="width:100%;border-collapse:collapse;font-size:12px;">')
        if is_nested:
            fv = next(iter(data.values()))
            if isinstance(fv,dict):
                keys = list(fv.keys())
                h.append('<tr style="background:#f0f0f0;"><th>Element</th>'+''.join(f'<th>{k}</th>' for k in keys)+'</tr>')
                for k,v in data.items(): h.append(f'<tr><td><b>{k}</b></td>'+''.join(f'<td style="text-align:right;">{v.get(ik,"N/A")}</td>' for ik in keys)+'</tr>')
        else:
            h.append('<tr style="background:#f0f0f0;"><th>Parameter</th><th>Value</th></tr>')
            for k,v in data.items(): h.append(f'<tr><td>{k}</td><td style="text-align:right;">{v}</td></tr>')
        h.append('</table>')
        return "".join(h)

    # --- CLUSTERING PAGE ---
    def create_clustering_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        c = QHBoxLayout()
        c.addWidget(QLabel("Method:"))
        self.cluster_method = QComboBox()
        self.cluster_method.addItems(["K-Means","DBSCAN"])
        self.cluster_method.currentTextChanged.connect(self.toggle_cluster_params)
        c.addWidget(self.cluster_method)
        self.kmeans_group = QGroupBox("K-Means")
        kg = QHBoxLayout()
        kg.addWidget(QLabel("k:"))
        self.kmeans_k = QSpinBox()
        self.kmeans_k.setRange(2,20)
        self.kmeans_k.setValue(3)
        kg.addWidget(self.kmeans_k)
        self.kmeans_group.setLayout(kg)
        c.addWidget(self.kmeans_group)
        self.dbscan_group = QGroupBox("DBSCAN")
        dg = QHBoxLayout()
        dg.addWidget(QLabel("eps:"))
        self.dbscan_eps = QDoubleSpinBox()
        self.dbscan_eps.setRange(0.01,10)
        self.dbscan_eps.setValue(0.5)
        dg.addWidget(self.dbscan_eps)
        dg.addWidget(QLabel("MinPts:"))
        self.dbscan_min_samples = QSpinBox()
        self.dbscan_min_samples.setRange(2,100)
        self.dbscan_min_samples.setValue(5)
        dg.addWidget(self.dbscan_min_samples)
        self.dbscan_group.setLayout(dg)
        self.dbscan_group.setVisible(False)
        c.addWidget(self.dbscan_group)
        c.addStretch()
        l.addLayout(c)
        vl = QHBoxLayout()
        vl.addWidget(QLabel("Variables for clustering:"))
        self.cluster_vars = QListWidget()
        self.cluster_vars.setSelectionMode(QAbstractItemView.MultiSelection)
        self.cluster_vars.setMaximumHeight(80)
        vl.addWidget(self.cluster_vars)
        l.addLayout(vl)
        bl = QHBoxLayout()
        self.btn_run_cluster = styledbutton("Run Clustering","refresh","primary")
        self.btn_run_cluster.clicked.connect(self.run_clustering)
        self.btn_export_cluster_fig = styledbutton("Export Figure","download","success")
        self.btn_export_cluster_fig.clicked.connect(self.export_cluster_figure)
        self.btn_export_assign = styledbutton("Export Assignments","download","default")
        self.btn_export_assign.clicked.connect(self.export_cluster_assignments)
        bl.addWidget(self.btn_run_cluster)
        bl.addWidget(self.btn_export_cluster_fig)
        bl.addWidget(self.btn_export_assign)
        bl.addStretch()
        l.addLayout(bl)
        split = QSplitter(Qt.Horizontal)
        self.cluster_fig = plt.Figure(figsize=(8,6))
        self.cluster_canvas = FigureCanvas(self.cluster_fig)
        split.addWidget(self.cluster_canvas)
        right_widget = QWidget()
        rl = QVBoxLayout(right_widget)
        self.cluster_stats = QTableWidget()
        self.cluster_stats.setAlternatingRowColors(True)
        rl.addWidget(self.cluster_stats)
        self.cluster_features_label = QLabel("🔍 Distinguishing features will appear here after running clustering")
        self.cluster_features_label.setWordWrap(True)
        self.cluster_features_label.setStyleSheet(f"padding:10px; background:white; border:1px solid {Palette.BORDER}; border-radius:4px;")
        rl.addWidget(self.cluster_features_label)
        split.addWidget(right_widget)
        split.setSizes([600,300])
        l.addWidget(split)
        self.stack.addWidget(p)

    def toggle_cluster_params(self, method):
        self.kmeans_group.setVisible(method == "K-Means")
        self.dbscan_group.setVisible(method == "DBSCAN")

    def run_clustering(self):
        if self.df is None:
            QMessageBox.warning(self,"Error","Load data first")
            return
        adf = self.get_analysis_df()
        if adf is None:
            return
        sel = [item.text() for item in self.cluster_vars.selectedItems()]
        if not sel:
            sel = adf.select_dtypes(include=[np.number]).columns.tolist()
        else:
            sel = [c for c in sel if c in adf.columns and pd.api.types.is_numeric_dtype(adf[c])]
        if len(sel) < 2:
            QMessageBox.warning(self,"Error","Need 2+ numeric columns")
            return
        X = adf[sel].dropna().values
        if len(X) == 0:
            QMessageBox.warning(self,"Error","No complete cases")
            return
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        method = self.cluster_method.currentText()
        
        try:
            if method == "K-Means":
                k = self.kmeans_k.value()
                model = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = model.fit_predict(X_scaled)
            else:
                eps = self.dbscan_eps.value()
                min_samples = self.dbscan_min_samples.value()
                model = DBSCAN(eps=eps, min_samples=min_samples)
                labels = model.fit_predict(X_scaled)
            
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            
            cluster_features = {}
            for lab in unique_labels:
                if lab == -1:
                    continue
                mask = labels == lab
                cluster_data = X_scaled[mask]
                if len(cluster_data) > 0:
                    means = cluster_data.mean(axis=0)
                    overall_means = X_scaled.mean(axis=0)
                    deviations = means - overall_means
                    top_idx = np.argsort(np.abs(deviations))[-5:][::-1]
                    cluster_features[lab] = [(sel[i], deviations[i]) for i in top_idx]
            
            pca = SklearnPCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            self.last_cluster_data = {
                'labels': labels, 'X_pca': X_pca, 'model': model,
                'selected_vars': sel, 'df': adf[sel].dropna(),
                'cluster_features': cluster_features, 'n_clusters': n_clusters, 'method': method
            }
            
            self.cluster_fig.clear()
            ax = self.cluster_fig.add_subplot(111)
            colors = plt.cm.tab10(np.linspace(0, 1, max(n_clusters, 1)))
            
            for i, lab in enumerate(unique_labels):
                mask = labels == lab
                if lab == -1:
                    color, marker, lbl = 'black', 'x', f'Noise ({np.sum(mask)})'
                else:
                    color = colors[i % len(colors)] if n_clusters > 0 else 'blue'
                    marker, lbl = 'o', f'Cluster {lab} ({np.sum(mask)})'
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], marker=marker,
                          alpha=0.7, edgecolors='white', s=50, label=lbl)
            
            if method == "K-Means" and hasattr(model, 'cluster_centers_'):
                centers_pca = pca.transform(model.cluster_centers_)
                ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X',
                          s=200, edgecolors='black', linewidths=2, label='Centers')
            
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax.set_title(f"{method} Clustering Results ({n_clusters} clusters)")
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            self.cluster_fig.tight_layout()
            self.cluster_canvas.draw()
            
            self.show_cluster_stats(labels, X_scaled, sel)
            
            feature_text = "🔍 Distinguishing Features per Cluster:\n\n"
            for lab, features in cluster_features.items():
                feature_text += f"📌 Cluster {lab}:\n"
                for var, dev in features:
                    direction = "↑ HIGH" if dev > 0 else "↓ LOW"
                    feature_text += f"   • {var}: {direction} (deviation: {dev:.2f})\n"
                feature_text += "\n"
            if not cluster_features:
                feature_text = "No clusters found with sufficient members for feature analysis."
            self.cluster_features_label.setText(feature_text)
            
            self.log(f"Clustering complete: {n_clusters} clusters found")
            self.set_status(f"Clustering: {n_clusters} clusters")
            
        except Exception as e:
            QMessageBox.critical(self,"Clustering Error",str(e))
            self.log(f"Clustering error: {e}")

    def show_cluster_stats(self, labels, X_scaled, var_names):
        unique_labels = np.unique(labels)
        rows = []
        for cl in unique_labels:
            mask = labels == cl
            count = np.sum(mask)
            row = {'Cluster': f'Cluster {cl}' if cl != -1 else 'Noise', 'Count': count}
            if count > 0:
                cluster_data = X_scaled[mask]
                for i, var in enumerate(var_names):
                    row[var] = round(float(cluster_data[:, i].mean()), 2)
            else:
                for var in var_names:
                    row[var] = 'N/A'
            rows.append(row)
        stats_df = pd.DataFrame(rows)
        self.cluster_stats.setRowCount(len(stats_df))
        self.cluster_stats.setColumnCount(len(stats_df.columns))
        self.cluster_stats.setHorizontalHeaderLabels(stats_df.columns.tolist())
        for i in range(len(stats_df)):
            for j in range(len(stats_df.columns)):
                self.cluster_stats.setItem(i, j, QTableWidgetItem(str(stats_df.iloc[i, j])))
        self.cluster_stats.resizeColumnsToContents()

    def export_cluster_figure(self):
        if not hasattr(self,'cluster_fig'): return
        path, _ = QFileDialog.getSaveFileName(self,"Save","clustering.png","PNG (*.png);;PDF (*.pdf)")
        if path: self.cluster_fig.savefig(path, dpi=300, bbox_inches='tight'); self.log("Saved cluster figure")

    def export_cluster_assignments(self):
        if self.last_cluster_data is None: return
        data = self.last_cluster_data
        df_out = data['df'].copy()
        df_out['Cluster'] = data['labels']
        path, _ = QFileDialog.getSaveFileName(self,"Save","clusters.csv","CSV (*.csv)")
        if path: df_out.to_csv(path, index=False); self.log("Saved cluster assignments")

    def create_svm_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        c = QHBoxLayout()
        self.btn_svm_config = styledbutton("Configure","list","primary")
        self.btn_svm_config.clicked.connect(self.configure_svm)
        self.btn_run_svm = styledbutton("Run","refresh","success")
        self.btn_run_svm.clicked.connect(self.run_svm)
        self.btn_overfit = styledbutton("Overfitting","trend","warning")
        self.btn_overfit.clicked.connect(self.visualize_overfitting)
        self.btn_export_svm = styledbutton("Export Fig","download","info")
        self.btn_export_svm.clicked.connect(self.export_svm_figure)
        c.addWidget(self.btn_svm_config); c.addWidget(self.btn_run_svm); c.addWidget(self.btn_overfit); c.addWidget(self.btn_export_svm)
        c.addStretch()
        l.addLayout(c)
        split = QSplitter(Qt.Vertical)
        self.svm_fig = plt.Figure(figsize=(12,5))
        self.svm_canvas = FigureCanvas(self.svm_fig)
        split.addWidget(self.svm_canvas)
        rw = QWidget()
        rl = QVBoxLayout(rw)
        rl.addWidget(QLabel("Results:"))
        self.svm_accuracy_table = QTableWidget()
        self.svm_accuracy_table.setMaximumHeight(80)
        rl.addWidget(self.svm_accuracy_table)
        self.svm_report = QTextEdit()
        self.svm_report.setReadOnly(True)
        rl.addWidget(self.svm_report)
        split.addWidget(rw)
        split.setSizes([400,200])
        l.addWidget(split)
        self.stack.addWidget(p)

    def create_log_page(self):
        p = QWidget()
        l = QVBoxLayout(p)
        lc = QHBoxLayout()
        btn_clear = styledbutton("Clear","trash","danger"); btn_clear.clicked.connect(self.clear_log)
        btn_export_log = styledbutton("Export","download","default"); btn_export_log.clicked.connect(self.export_log)
        lc.addWidget(btn_clear); lc.addWidget(btn_export_log); lc.addStretch()
        l.addLayout(lc)
        self.log_text = QTextEdit(); self.log_text.setReadOnly(True)
        l.addWidget(self.log_text)
        self.stack.addWidget(p)

    def show_help(self):
        QMessageBox.about(self,"About",f"{APPNAME} v{APPVERSION}\n\n{DEVELOPER}\n{EMAIL}")

    # --- Data Operations ---
    def upload_file(self):
        path, _ = QFileDialog.getOpenFileName(self,"Open","","CSV/Excel (*.csv *.xlsx *.xls);;All (*)")
        if not path: return
        try:
            self.filename = os.path.basename(path)
            ext = os.path.splitext(path)[1].lower()
            if ext == '.csv':
                for enc in ['utf-8','latin1','iso-8859-1','cp1252']:
                    try: self.raw_df = pd.read_csv(path, encoding=enc); break
                    except: continue
            elif ext in ['.xlsx','.xls']: self.raw_df = pd.read_excel(path, engine='openpyxl')
            if self.raw_df is None or self.raw_df.empty: raise ValueError("Cannot read file")
            self.raw_df.columns = [normalize_column_name(c) for c in self.raw_df.columns]
            self.df = self.raw_df.copy()
            self.active_columns = self.df.columns.tolist()
            self.log(f"Loaded {self.filename}: {self.df.shape}")
            self.set_status(f"Loaded: {self.filename}")
            self.populate_vars(); self.update_table(); self.update_data_info()
            self.update_analysis_vars(); self.update_clustering_vars(); self.update_ratio_selector()
            self.stack.setCurrentIndex(1)
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def update_ratio_selector(self):
        if self.df is not None:
            self.ratio_selector.clear()
            self.ratio_selector.addItems(self.df.select_dtypes(include=[np.number]).columns.tolist())

    def advanced_clean_data(self):
        if self.df is None: return
        dlg = AdvancedCleanDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            opts = {k: v.isChecked() for k,v in dlg.cbs.items()}
            self.df, report = clean_dataframe_advanced(self.df, opts)
            self.active_columns = [c for c in self.active_columns if c in self.df.columns]
            self.log(report); self.update_table(); self.update_data_info()
            self.populate_vars(); self.update_analysis_vars(); self.update_clustering_vars(); self.update_ratio_selector()

    def open_filter_dialog(self):
        if self.df is None: return
        dlg = DataFilterDialog(self.df.columns.tolist(), self)
        if dlg.exec_() == QDialog.Accepted: self.apply_filter(*dlg.get_filter_config())

    def apply_filter(self, col, ftype, val, op):
        if col not in self.df.columns: return
        try:
            s = pd.to_numeric(self.df[col], errors='coerce'); mask = pd.Series([True]*len(self.df))
            if ftype == "Outliers (IQR)": q1,q3 = s.quantile(0.25),s.quantile(0.75); iqr = q3-q1; mask = (s>=q1-1.5*iqr)&(s<=q3+1.5*iqr)
            elif ftype == "Percentile": thresh = s.quantile(val/100.0); mask = s<thresh if op=="<" else s>thresh if op==">" else s==thresh
            else: mask = s<val if op=="<" else s>val if op==">" else s==val
            self.df = self.df[mask].reset_index(drop=True); self.log(f"Filtered: kept {mask.sum()}"); self.update_table(); self.populate_vars()
        except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def delete_data(self):
        if self.df is None: return
        dlg = DeleteDataDialog(self.df, self)
        if dlg.exec_() == QDialog.Accepted:
            try:
                if dlg.rb_rows_range.isChecked(): parts = dlg.row_range.text().split('-'); start,end = int(parts[0]),int(parts[1]); self.df = self.df.drop(self.df.index[start:end+1]).reset_index(drop=True)
                elif dlg.rb_rows_condition.isChecked(): col = dlg.row_cond_col.currentText(); val = dlg.row_cond_val.text(); self.df = self.df[self.df[col].astype(str)!=val].reset_index(drop=True)
                elif dlg.rb_rows_all.isChecked(): self.df = self.df.iloc[0:0]
                selected = [item.text() for item in dlg.col_list.selectedItems()]
                if selected: self.df = self.df.drop(columns=selected, errors='ignore'); self.active_columns = [c for c in self.active_columns if c not in selected]
                if dlg.cb_remove_from_app.isChecked(): self.df = None; self.raw_df = None; self.active_columns = []
                self.update_table(); self.update_data_info()
            except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def control_display(self):
        if self.df is None: return
        dlg = DisplayControlDialog(len(self.df), len(self.df.columns), self)
        if dlg.exec_() == QDialog.Accepted: self.show_all = dlg.cb_full.isChecked(); self.display_rows = dlg.sb_rows.value(); self.display_cols = dlg.sb_cols.value(); self.update_table()

    def set_index_columns(self):
        if self.df is None: return
        dlg = IndexColumnDialog(self.df.columns.tolist(), self.index_column, self)
        if dlg.exec_() == QDialog.Accepted: self.index_column = dlg.index_combo.currentText() if dlg.index_combo.currentText()!="None" else None; self.active_columns = [item.text() for item in dlg.active_list.selectedItems()]; self.update_data_info()

    def rename_columns_dialog(self):
        if self.df is None: return
        old, ok = QInputDialog.getItem(self,"Rename","Column:",self.df.columns.tolist(),0,False)
        if ok and old: new, ok2 = QInputDialog.getText(self,"Rename",f"New name:")
        if ok2 and new: self.df.columns = [new if c==old else c for c in self.df.columns]; self.update_table()

    def calculate_element_ratio(self):
        if self.df is None: return
        nc = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(nc)<2: return
        dlg = ElementRatioDialog(nc, self)
        if dlg.exec_() == QDialog.Accepted:
            try: self.df[dlg.name_input.text()] = pd.to_numeric(self.df[dlg.num_combo.currentText()],errors='coerce')/(pd.to_numeric(self.df[dlg.den_combo.currentText()],errors='coerce')+1e-9); self.update_table(); self.populate_vars()
            except Exception as e: QMessageBox.critical(self,"Error",str(e))

    def reset_data(self):
        if self.raw_df is None: return
        if QMessageBox.question(self,"Reset","Reset all?")==QMessageBox.Yes: self.df = self.raw_df.copy(); self.active_columns = self.df.columns.tolist(); self.index_column = None; self.update_table(); self.update_data_info(); self.populate_vars(); self.log("Data reset")

    def unify_units(self):
        if self.df is None: return
        for c in self.df.columns:
            if "ptc" in c.lower(): new = c.replace("ptc","%"); self.df.rename(columns={c:new}, inplace=True)
        self.update_table()

    def switch_data_view(self, mode):
        if mode=="Raw Data" and self.raw_df is not None: self.update_table(data=self.raw_df)
        elif mode=="Analysis Data": self.update_table(data=self.get_analysis_df())
        else: self.update_table(data=self.df)

    def update_data_info(self):
        if self.df is not None: self.data_info.setText(f"Rows:{self.df.shape[0]} Cols:{self.df.shape[1]} | Active:{len(self.active_columns)}")

    def populate_vars(self):
        if self.df is not None: self.dist_var.clear(); self.dist_var.addItems(self.df.select_dtypes(include=[np.number]).columns.tolist())

    def update_analysis_vars(self):
        if self.df is not None: self.analysis_vars.clear(); self.analysis_vars.addItems(self.df.select_dtypes(include=[np.number]).columns.tolist())
        for i in range(min(10,self.analysis_vars.count())): self.analysis_vars.item(i).setSelected(True)

    def update_clustering_vars(self):
        if self.df is not None: self.cluster_vars.clear(); self.cluster_vars.addItems(self.df.select_dtypes(include=[np.number]).columns.tolist())
        for i in range(min(10,self.cluster_vars.count())): self.cluster_vars.item(i).setSelected(True)

    def update_table(self, data=None):
        df = data if data is not None else self.df
        if df is None: self.table.setRowCount(0); self.table.setColumnCount(0); return
        disp = df if self.show_all else df.head(self.display_rows).iloc[:,:min(self.display_cols,len(df.columns))]
        self.table.setRowCount(len(disp)); self.table.setColumnCount(len(disp.columns)); self.table.setHorizontalHeaderLabels([str(c) for c in disp.columns])
        for r in range(len(disp)):
            for c in range(len(disp.columns)): val = disp.iloc[r,c]; txt = "" if pd.isna(val) else f"{val:.4g}" if isinstance(val,(float,np.floating)) else str(val); self.table.setItem(r,c,QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()

    def show_plot_config_safe(self):
        dlg = PlotConfigDialog(self)
        if dlg.exec_()==QDialog.Accepted: self.last_plot_config = dlg.get_selection()

    def update_distribution(self):
        if self.df is None: return
        var = self.dist_var.currentText()
        if not var: return
        sel = self.last_plot_config or ["Histogram","PDF KDE"]
        data = pd.to_numeric(self.df[var],errors='coerce').dropna()
        if len(data)==0: return
        self.dist_fig.clear(); n = len(sel); cols = 2 if n>1 else 1; rows = (n+1)//2; gs = gridspec.GridSpec(rows,cols,figure=self.dist_fig)
        for i,pt in enumerate(sel):
            ax = self.dist_fig.add_subplot(gs[i])
            if pt=="Histogram": ax.hist(data,bins=30,color=Palette.PRIMARY,alpha=0.7)
            elif pt=="PDF KDE": sns.kdeplot(data,ax=ax,color=Palette.PRIMARY,fill=True)
            elif pt=="CDF": ax.plot(np.sort(data),np.linspace(0,1,len(data)))
            elif pt=="Box Plot": ax.boxplot(data,vert=False)
            elif pt=="Q-Q Plot": scipystats.probplot(data,dist="norm",plot=ax)
            ax.set_xlabel(var); ax.grid(True,alpha=0.3)
        self.dist_fig.tight_layout(); self.dist_canvas.draw()

    def export_distribution_plot(self):
        path, _ = QFileDialog.getSaveFileName(self,"Save","distribution.png","PNG (*.png);;PDF (*.pdf)")
        if path: self.dist_fig.savefig(path,dpi=300,bbox_inches='tight')

    # --- PCA ANALYSIS WITH FEATURES REPORT ---
    def update_analysis(self):
        if self.df is None: return
        adf = self.get_analysis_df()
        if adf is None: return
        sel = [item.text() for item in self.analysis_vars.selectedItems()]
        if not sel: sel = adf.select_dtypes(include=[np.number]).columns.tolist()[:10]
        av = [v for v in sel if v in adf.columns]
        if not av: return
        nd = adf[av].select_dtypes(include=[np.number])
        if nd.shape[1] < 2: return
        atype = self.analysis_type.currentText()
        self.ana_fig.clear()
        report = ""
        
        if atype == "Correlation Matrix":
            ax = self.ana_fig.add_subplot(111)
            if nd.shape[1] <= 15:
                corr = nd.corr()
                sns.heatmap(corr, ax=ax, cmap="coolwarm", annot=nd.shape[1]<=8, fmt='.2f')
                strong_corr = []
                for i in range(len(corr.columns)):
                    for j in range(i+1, len(corr.columns)):
                        if abs(corr.iloc[i,j]) > 0.7:
                            strong_corr.append(f"{corr.columns[i]}-{corr.columns[j]}: {corr.iloc[i,j]:.2f}")
                if strong_corr:
                    report = "🔗 Strong correlations (|r|>0.7):\n" + "\n".join(strong_corr)
                else:
                    report = "No strong correlations found (|r|<0.7 for all pairs)"
            ax.set_title("Correlation Matrix")
            
        elif atype == "PCA Analysis":
            ax = self.ana_fig.add_subplot(111)
            try:
                clean = nd.dropna()
                scaled = StandardScaler().fit_transform(clean)
                pca = SklearnPCA(n_components=2)
                pcs = pca.fit_transform(scaled)
                ax.scatter(pcs[:,0], pcs[:,1], c=range(len(pcs)), cmap='viridis', alpha=0.7, edgecolors='white')
                
                report_lines = []
                report_lines.append(f"📊 PCA Results:")
                report_lines.append(f"• PC1 explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
                report_lines.append(f"• PC2 explains {pca.explained_variance_ratio_[1]*100:.1f}% of variance")
                report_lines.append(f"• Cumulative: {(pca.explained_variance_ratio_.sum())*100:.1f}%")
                
                if self.cb_show_pca.isChecked():
                    loadings = pca.components_.T
                    scale = np.max(np.abs(pcs))*0.8
                    for i, var in enumerate(nd.columns[:15]):
                        ax.arrow(0,0, loadings[i,0]*scale, loadings[i,1]*scale,
                                head_width=scale*0.04, head_length=scale*0.04, fc='red', ec='red', alpha=0.8, linewidth=1.5)
                        ax.text(loadings[i,0]*scale*1.15, loadings[i,1]*scale*1.15, var,
                               fontsize=8, color='darkred', weight='bold')
                    
                    pc1_importance = np.abs(loadings[:,0])
                    top_pc1_idx = np.argsort(pc1_importance)[-5:][::-1]
                    top_pc1 = [f"{nd.columns[i]} ({loadings[i,0]:.2f})" for i in top_pc1_idx]
                    report_lines.append(f"• Top PC1 contributors: {', '.join(top_pc1)}")
                    
                    pc2_importance = np.abs(loadings[:,1])
                    top_pc2_idx = np.argsort(pc2_importance)[-5:][::-1]
                    top_pc2 = [f"{nd.columns[i]} ({loadings[i,1]:.2f})" for i in top_pc2_idx]
                    report_lines.append(f"• Top PC2 contributors: {', '.join(top_pc2)}")
                
                ax.axhline(0, color='grey', linestyle='--', alpha=0.3)
                ax.axvline(0, color='grey', linestyle='--', alpha=0.3)
                ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
                ax.set_title("PCA Biplot with Loading Vectors")
                report = "\n".join(report_lines)
                
            except Exception as e:
                ax.text(0.5,0.5, f"PCA failed:\n{str(e)}", ha='center', va='center', transform=ax.transAxes)
                report = f"PCA Error: {str(e)}"
                
        elif atype == "Scatter Plot":
            ax = self.ana_fig.add_subplot(111)
            if nd.shape[1] >= 2:
                colors = nd.iloc[:,2].values if nd.shape[1]>=3 else range(len(nd))
                ax.scatter(nd.iloc[:,0], nd.iloc[:,1], c=colors, cmap='viridis', alpha=0.7, edgecolors='white')
                ax.set_xlabel(nd.columns[0])
                ax.set_ylabel(nd.columns[1])
                ax.set_title(f"{nd.columns[0]} vs {nd.columns[1]}")
                corr_val = nd.iloc[:,0].corr(nd.iloc[:,1])
                report = f"📈 Scatter Plot:\n• {nd.columns[0]} vs {nd.columns[1]}\n• Pearson r = {corr_val:.4f}\n• R² = {corr_val**2:.4f}"
                
        self.ana_fig.tight_layout()
        self.ana_canvas.draw()
        self.ana_report.setText(report)
        self.last_analysis_report = f"{atype}:\n{report}"
        self.update_statistics()

    def export_analysis_plot(self):
        path, _ = QFileDialog.getSaveFileName(self,"Save","analysis.png","PNG (*.png);;PDF (*.pdf)")
        if path: self.ana_fig.savefig(path,dpi=300,bbox_inches='tight')

    def update_statistics(self):
        if self.df is None: return
        adf = self.get_analysis_df(); stats = EnhancedStatisticsEngine.compute(adf) if adf is not None else pd.DataFrame()
        if stats.empty: return
        self.stats_table.setRowCount(len(stats)); self.stats_table.setColumnCount(len(stats.columns)); self.stats_table.setHorizontalHeaderLabels(stats.columns.tolist())
        for i in range(len(stats)):
            for j in range(len(stats.columns)): self.stats_table.setItem(i,j,QTableWidgetItem(str(stats.iloc[i,j])))
        self.stats_table.resizeColumnsToContents()

    def export_statistics(self):
        if self.df is None: return
        stats = EnhancedStatisticsEngine.compute(self.get_analysis_df())
        if stats.empty: return
        path, _ = QFileDialog.getSaveFileName(self,"Save","statistics.csv","CSV (*.csv)")
        if path: stats.to_csv(path,index=False)

    # --- SVM Methods ---
    def configure_svm(self):
        if self.df is None: return
        dlg = SVMDialog(self.df.columns.tolist(), self)
        if dlg.exec_() == QDialog.Accepted: self.svm_params = dlg.get_params(); self.log("SVM configured")

    def run_svm(self):
        if self.svm_params is None or self.df is None: QMessageBox.warning(self,"Error","Configure SVM first"); return
        params = self.svm_params
        if not params['features']: QMessageBox.warning(self,"Error","No features selected"); return
        
        try:
            numeric_cols = self.get_numeric_cols(); valid_features = [f for f in params['features'] if f in numeric_cols]
            if not valid_features: QMessageBox.warning(self,"Error","No valid numeric features"); return
            
            df = self.df[[params['target']]+valid_features].copy(); df[params['target']] = df[params['target']].astype(str)
            for col in valid_features: df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            if len(df) < 10: QMessageBox.warning(self,"Error",f"Only {len(df)} complete cases"); return
            
            X = df[valid_features].values; y_raw = df[params['target']].values
            
            unique_classes, class_counts = np.unique(y_raw, return_counts=True); small_mask = class_counts < 3
            if small_mask.any():
                small_classes = unique_classes[small_mask]; y_clean = y_raw.copy()
                for sc in small_classes: y_clean[y_clean == sc] = 'Other'
                self.log(f"Merged {len(small_classes)} small classes into 'Other'")
            else: y_clean = y_raw
            
            le = LabelEncoder(); y = le.fit_transform(y_clean); all_class_names = [str(c) for c in le.classes_]
            if len(all_class_names) < 2: QMessageBox.warning(self,"Error","Need at least 2 classes"); return
            
            scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
            try: X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=params['test_size'],random_state=42,stratify=y)
            except: X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=params['test_size'],random_state=42)
            
            model = SVC(kernel=params['kernel'],C=params['C'],degree=params['degree'] if params['kernel']=='poly' else 3,random_state=42)
            model.fit(X_train,y_train); y_pred = model.predict(X_test)
            
            test_classes = np.unique(np.concatenate([y_test,y_pred])); test_class_names = [all_class_names[i] for i in test_classes if i < len(all_class_names)]
            
            acc = accuracy_score(y_test,y_pred); cm = confusion_matrix(y_test,y_pred,labels=test_classes)
            report = classification_report(y_test,y_pred,labels=test_classes,target_names=test_class_names,zero_division=0)
            
            self.last_svm_data = {'accuracy':acc,'report':report,'class_names':test_class_names}
            
            self.svm_fig.clear(); pca = SklearnPCA(n_components=2); X_pca = pca.fit_transform(X_scaled); X_test_pca = pca.transform(X_test)
            ax1 = self.svm_fig.add_subplot(121); ax2 = self.svm_fig.add_subplot(122)
            
            x_min,x_max = X_pca[:,0].min()-1,X_pca[:,0].max()+1; y_min,y_max = X_pca[:,1].min()-1,X_pca[:,1].max()+1
            xx,yy = np.meshgrid(np.linspace(x_min,x_max,100),np.linspace(y_min,y_max,100))
            b_model = SVC(kernel=params['kernel'],C=params['C'],degree=params['degree'] if params['kernel']=='poly' else 3,random_state=42)
            b_model.fit(X_pca,y); Z = b_model.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
            
            n_classes = len(test_classes)
            if n_classes <= 10: ax1.contourf(xx,yy,Z,alpha=0.3,cmap=plt.cm.RdYlBu,levels=n_classes)
            else: ax1.contourf(xx,yy,Z,alpha=0.3,cmap=plt.cm.RdYlBu)
            
            scatter = ax1.scatter(X_test_pca[:,0],X_test_pca[:,1],c=y_test,cmap=plt.cm.RdYlBu,edgecolors='black',s=60)
            ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"); ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            ax1.set_title(f"SVM Decision Boundary\nAccuracy: {acc:.2%}")
            if n_classes <= 10: ax1.legend(*scatter.legend_elements(),title="Classes",loc='upper right',fontsize=7)
            
            if cm.shape[0] <= 15: sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax2,xticklabels=test_class_names,yticklabels=test_class_names,annot_kws={'size':7})
            else: sns.heatmap(cm,cmap='Blues',ax=ax2)
            ax2.set_title("Confusion Matrix"); ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
            self.svm_fig.tight_layout(); self.svm_canvas.draw()
            
            self.svm_accuracy_table.setRowCount(1); self.svm_accuracy_table.setColumnCount(4); self.svm_accuracy_table.setHorizontalHeaderLabels(["Accuracy","Precision","Recall","F1"])
            try: prec = precision_score(y_test,y_pred,average='weighted',zero_division=0,labels=test_classes); rec = recall_score(y_test,y_pred,average='weighted',zero_division=0,labels=test_classes); f1 = f1_score(y_test,y_pred,average='weighted',zero_division=0,labels=test_classes)
            except: prec = precision_score(y_test,y_pred,average='weighted',zero_division=0); rec = recall_score(y_test,y_pred,average='weighted',zero_division=0); f1 = f1_score(y_test,y_pred,average='weighted',zero_division=0)
            for j,v in enumerate([acc,prec,rec,f1]): self.svm_accuracy_table.setItem(0,j,QTableWidgetItem(f"{v:.2%}"))
            self.svm_accuracy_table.resizeColumnsToContents()
            self.svm_report.setText(f"Classification Report:\n\n{report}")
            self.log(f"SVM accuracy: {acc:.3f}"); self.set_status(f"SVM Acc: {acc:.2%}")
        except Exception as e: QMessageBox.critical(self,"SVM Error",f"{str(e)}\n\n{traceback.format_exc()}"); self.log(f"SVM error: {e}")

    def visualize_overfitting(self):
        if self.svm_params is None or self.df is None: return
        params = self.svm_params
        if not params['features']: return
        valid_features = [f for f in params['features'] if f in self.get_numeric_cols()]
        if not valid_features: return
        df = self.df[[params['target']]+valid_features].copy()
        for col in valid_features: df[col] = pd.to_numeric(df[col],errors='coerce')
        df = df.dropna()
        if len(df)==0: return
        X = df[valid_features].values; y_raw = df[params['target']].astype(str).values; le = LabelEncoder(); y = le.fit_transform(y_raw)
        scaler = StandardScaler(); X_scaled = scaler.fit_transform(X); X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=params['test_size'],random_state=42)
        C_range = np.logspace(-2,3,10); train_scores,test_scores = [],[]
        for C in C_range: model = SVC(kernel=params['kernel'],C=C,degree=params['degree'] if params['kernel']=='poly' else 3,random_state=42); model.fit(X_train,y_train); train_scores.append(model.score(X_train,y_train)); test_scores.append(model.score(X_test,y_test))
        self.svm_fig.clear(); ax = self.svm_fig.add_subplot(111); ax.semilogx(C_range,train_scores,'o-',label='Train',markersize=8); ax.semilogx(C_range,test_scores,'s-',label='Test',markersize=8); ax.set_xlabel('C'); ax.set_ylabel('Accuracy'); ax.set_title('Overfitting Analysis'); ax.legend(); ax.grid(True,alpha=0.3); self.svm_fig.tight_layout(); self.svm_canvas.draw()

    def export_svm_figure(self):
        path, _ = QFileDialog.getSaveFileName(self,"Save","svm_plot.png","PNG (*.png);;PDF (*.pdf)")
        if path: self.svm_fig.savefig(path,dpi=300,bbox_inches='tight')

    def export_current_figure(self):
        idx = self.stack.currentIndex()
        if idx==2: self.export_distribution_plot()
        elif idx==3: self.export_analysis_plot()
        elif idx==5: self.export_cluster_figure()
        elif idx==6: self.export_svm_figure()

    def export_data_csv(self):
        if self.df is None: return
        path, _ = QFileDialog.getSaveFileName(self,"Save","data.csv","CSV (*.csv);;Excel (*.xlsx)")
        if path:
            if path.endswith('.xlsx'): self.df.to_excel(path,index=False)
            else: self.df.to_csv(path,index=False)

    def export_geochem_excel(self):
        path, _ = QFileDialog.getSaveFileName(self,"Save","geochem_results.xlsx","Excel (*.xlsx)")
        if path:
            with pd.ExcelWriter(path,engine='openpyxl') as w:
                for k,data in self.last_geochem_data.items():
                    if data: df = pd.DataFrame.from_dict(data,orient='index') if isinstance(next(iter(data.values())),dict) else pd.DataFrame(list(data.items()),columns=['Param','Value']); df.to_excel(w,sheet_name=k[:31],index=False)
            self.log(f"Saved: {path}")

    def generate_full_report(self):
        if self.df is None: QMessageBox.warning(self,"No Data","Upload data first"); return
        adf = self.get_analysis_df(); stats_df = EnhancedStatisticsEngine.compute(adf)
        html = f"""<html><head><title>{APPNAME} Report</title><style>body{{font-family:'Segoe UI',sans-serif;margin:40px;color:#333;}}h1{{color:{Palette.PRIMARY};border-bottom:3px solid {Palette.PRIMARY};}}h2{{color:{Palette.PRIMARYDARK};margin-top:30px;}}table{{border-collapse:collapse;width:100%;margin:10px 0;}}th,td{{border:1px solid #ddd;padding:8px;}}th{{background-color:{Palette.PRIMARYLIGHT};}}.footer{{margin-top:50px;font-size:12px;color:#999;text-align:center;}}</style></head>
        <body><h1>{APPNAME} v{APPVERSION} - Report</h1><p>File:{self.filename}<br>Generated:{datetime.now().strftime('%Y-%m-%d %H:%M')}<br>{DEVELOPER} | {EMAIL}</p>
        <h2>Data Summary</h2><table><tr><td>Rows</td><td>{len(self.df)}</td></tr><tr><td>Cols</td><td>{len(self.df.columns)}</td></tr><tr><td>Numeric</td><td>{len(self.df.select_dtypes(include=[np.number]).columns)}</td></tr></table>
        <h2>Statistics</h2>{stats_df.to_html(index=False) if not stats_df.empty else '<p>No data</p>'}
        <h2>Analysis</h2><pre>{self.last_analysis_report or 'Not performed'}</pre>
        <h2>Geochemistry</h2>{self.dict_to_html('Alteration',self.last_geochem_data.get('alterationindices',{}))}{self.dict_to_html('Ratios',self.last_geochem_data.get('elementratios',{}))}{self.dict_to_html('Grades',self.last_geochem_data.get('gradestats',{}),True)}
        <h2>Clustering</h2><p>{f"Method:{self.last_cluster_data.get('method','N/A')}, Clusters:{self.last_cluster_data.get('n_clusters','N/A')}" if self.last_cluster_data else 'Not performed'}</p>
        <h2>SVM</h2>{f"<p>Accuracy:{self.last_svm_data.get('accuracy','N/A')}</p><pre>{self.last_svm_data.get('report','')}</pre>" if self.last_svm_data else '<p>Not performed</p>'}
        <div class='footer'><p>{APPNAME} v{APPVERSION} | {DEVELOPER} | {EMAIL}</p></div></body></html>"""
        d = QDialog(self); d.setWindowTitle("Report"); d.resize(900,700); l = QVBoxLayout(d); v = QTextEdit(); v.setHtml(html); v.setReadOnly(True); l.addWidget(v)
        btns = QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Close); btns.rejected.connect(d.reject); btns.button(QDialogButtonBox.Save).clicked.connect(lambda: self.save_report(html)); l.addWidget(btns); d.exec_()

    def save_report(self, html):
        path, _ = QFileDialog.getSaveFileName(self, "Save", f"report_{os.path.splitext(self.filename)[0]}.html" if self.filename else "report.html", "HTML (*.html)")
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html)
            self.log(f"Report saved: {path}")

    def clear_log(self):
        self.log_text.clear()

    def export_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save", "log.txt", "Text (*.txt)")
        if path:
            with open(path, 'w') as f:
                f.write(self.log_text.toPlainText())

    def get_stylesheet(self):
        P = Palette
        return f"""
        QMainWindow{{background-color:{P.BG};}}QWidget{{font-family:"Segoe UI";font-size:12px;color:{P.TEXTPRIMARY};}}
        #Sidebar{{background-color:{P.SIDEBARBG};color:{P.SIDEBARTEXT};}}#BrandLabel{{color:{P.SIDEBARTEXTACTIVE};font-size:20px;font-weight:bold;padding:20px;}}
        QListWidget{{background:transparent;border:none;}}QListWidget::item{{padding:12px 20px;color:{P.SIDEBARTEXT};}}QListWidget::item:selected{{background-color:{P.PRIMARY};color:white;border-radius:4px;}}
        #Toolbar{{background-color:{P.SURFACE};padding:8px;border-bottom:1px solid {P.BORDER};}}#StatusPanel{{background-color:{P.PRIMARYLIGHT};color:{P.PRIMARY};padding:8px;font-weight:bold;}}
        QPushButton{{padding:8px 16px;border-radius:6px;border:1px solid {P.BORDER};background:{P.SURFACE};}}QPushButton[kind="primary"]{{background:{P.PRIMARY};color:white;border:none;}}QPushButton[kind="warning"]{{background:{P.WARNING};color:white;border:none;}}QPushButton[kind="success"]{{background:{P.SUCCESS};color:white;border:none;}}QPushButton[kind="danger"]{{background:{P.DANGER};color:white;border:none;}}
        QTableWidget{{background:white;gridline-color:#eee;border:1px solid {P.BORDER};}}QHeaderView::section{{background:#f9fafb;padding:6px;border:1px solid #eee;font-weight:bold;}}
        QGroupBox{{font-weight:bold;border:1px solid {P.BORDER};border-radius:6px;margin-top:10px;padding-top:10px;}}QGroupBox::title{{subcontrol-origin:margin;left:10px;padding:0 5px;}}
        """

def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APPNAME)
    w = ElementalSocietyAnalysis()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()