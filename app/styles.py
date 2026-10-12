APP_STYLESHEET = """
/* DaVinci Resolve–inspired dark NLE chrome */
QWidget {
    background-color: #1b1b1b;
    color: #e6e6e6;
    font-family: "Segoe UI", "Arial";
    font-size: 12px;
}
QMainWindow, QDialog {
    background-color: #1b1b1b;
}
QLabel#BrandTitle {
    font-size: 16px;
    font-weight: 700;
    color: #f2f2f2;
    letter-spacing: 0.4px;
    padding: 2px 0;
}
QLabel#Subtitle {
    color: #8a8a8a;
    font-size: 11px;
}
QLabel#PanelTitle {
    font-size: 11px;
    font-weight: 700;
    color: #c8c8c8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 2px 0 6px 0;
}
QFrame#Toolbar {
    background-color: #242424;
    border-bottom: 1px solid #0f0f0f;
}
QFrame#Panel {
    background-color: #222222;
    border: 1px solid #0f0f0f;
}
QPushButton {
    background-color: #2e2e2e;
    border: 1px solid #3a3a3a;
    border-radius: 3px;
    padding: 7px 12px;
    color: #e6e6e6;
}
QPushButton:hover {
    background-color: #3a3a3a;
    border-color: #505050;
}
QPushButton:pressed {
    background-color: #1f1f1f;
}
QPushButton#Primary {
    background-color: #c4541a;
    border-color: #e06a2c;
    font-weight: 600;
    color: #fff;
}
QPushButton#Primary:hover {
    background-color: #d46322;
}
QPushButton#Success {
    background-color: #2f6b3a;
    border-color: #3d8a4a;
    color: #fff;
}
QPushButton#Success:hover {
    background-color: #388548;
}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit {
    background-color: #141414;
    border: 1px solid #333;
    border-radius: 3px;
    padding: 5px 7px;
    selection-background-color: #c4541a;
}
QListWidget, QTableWidget, QTreeWidget {
    background-color: #191919;
    border: 1px solid #0f0f0f;
    border-radius: 2px;
    outline: none;
}
QListWidget#MediaPool {
    background-color: #161616;
    padding: 6px;
}
QListWidget#MediaPool::item {
    background-color: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    color: #d0d0d0;
    padding: 4px;
}
QListWidget#MediaPool::item:selected {
    background-color: #3a2818;
    border: 1px solid #c4541a;
}
QListWidget#ProjectBin::item {
    padding: 8px 10px;
    border-bottom: 1px solid #2a2a2a;
}
QListWidget::item, QTableWidget::item {
    padding: 6px;
}
QListWidget::item:selected, QTableWidget::item:selected {
    background-color: #3a2818;
    color: #fff;
}
QHeaderView::section {
    background-color: #242424;
    border: none;
    padding: 6px;
    font-weight: 600;
}
QTabWidget::pane {
    border: 1px solid #0f0f0f;
    background: #1b1b1b;
    top: -1px;
}
QTabBar::tab {
    background: #242424;
    border: 1px solid #0f0f0f;
    padding: 9px 16px;
    margin-right: 2px;
    color: #b0b0b0;
}
QTabBar::tab:selected {
    background: #c4541a;
    color: #fff;
    border-color: #e06a2c;
}
QTabBar::tab:hover:!selected {
    background: #303030;
    color: #fff;
}
QProgressBar {
    border: 1px solid #333;
    border-radius: 2px;
    text-align: center;
    background: #141414;
    height: 16px;
}
QProgressBar::chunk {
    background-color: #c4541a;
}
QSlider::groove:horizontal {
    height: 5px;
    background: #333;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    width: 12px;
    margin: -4px 0;
    border-radius: 6px;
    background: #e06a2c;
}
QGroupBox {
    border: 1px solid #333;
    border-radius: 3px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    padding: 0 4px;
    color: #9a9a9a;
}
QStatusBar {
    background: #141414;
    color: #8a8a8a;
    border-top: 1px solid #0f0f0f;
}
QSplitter::handle {
    background: #0f0f0f;
}
QSplitter::handle:horizontal {
    width: 3px;
}
QSplitter::handle:vertical {
    height: 3px;
}
QScrollBar:vertical {
    background: #1b1b1b;
    width: 10px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #3a3a3a;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
"""
