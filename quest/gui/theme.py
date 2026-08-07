"""Modern dark and light Qt stylesheets (QSS) for QuEst desktop application.

Provides rich visual aesthetics including glassmorphism cards, modern typography,
rounded controls, styled table widgets, pyqtgraph integration, and vibrant accents.
"""

from __future__ import annotations

DARK_THEME_QSS = """
/* QuEst Modern Dark Theme */
QMainWindow, QDialog {
    background-color: #0f172a;
    color: #f8fafc;
}

QWidget {
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
}

/* ToolBar & Menus */
QToolBar {
    background-color: #1e293b;
    border-bottom: 1px solid #334155;
    padding: 4px;
    spacing: 6px;
}

QToolButton {
    background-color: #334155;
    color: #f8fafc;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}

QToolButton:hover {
    background-color: #475569;
    border-color: #6366f1;
}

QToolButton:pressed {
    background-color: #6366f1;
    color: #ffffff;
}

QMenuBar {
    background-color: #1e293b;
    color: #e2e8f0;
    border-bottom: 1px solid #334155;
}

QMenuBar::item:selected {
    background-color: #334155;
    border-radius: 4px;
}

QMenu {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 4px;
}

QMenu::item:selected {
    background-color: #6366f1;
    color: #ffffff;
    border-radius: 4px;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #334155;
    border-radius: 8px;
    background-color: #1e293b;
    top: -1px;
}

QTabBar::tab {
    background-color: #0f172a;
    color: #94a3b8;
    border: 1px solid #334155;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #1e293b;
    color: #6366f1;
    border-color: #6366f1;
    border-bottom: 2px solid #6366f1;
}

QTabBar::tab:hover:!selected {
    background-color: #1e293b;
    color: #cbd5e1;
}

/* Collapsible Panel & Frame */
QFrame, QGroupBox {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    margin-top: 4px;
}

QPushButton#CollapsibleToggle {
    background-color: #334155;
    color: #f8fafc;
    border: 1px solid #475569;
    border-radius: 6px;
    padding: 8px;
    text-align: left;
    font-weight: 700;
    font-size: 13px;
}

QPushButton#CollapsibleToggle:hover {
    background-color: #475569;
    border-color: #6366f1;
}

/* Buttons */
QPushButton {
    background-color: #6366f1;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #4f46e5;
}

QPushButton:pressed {
    background-color: #4338ca;
}

QPushButton:disabled {
    background-color: #334155;
    color: #64748b;
}

QPushButton#PrimaryButton {
    background-color: #10b981;
}

QPushButton#PrimaryButton:hover {
    background-color: #059669;
}

QPushButton#DangerButton {
    background-color: #ef4444;
}

QPushButton#DangerButton:hover {
    background-color: #dc2626;
}

/* Input Fields */
QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #0f172a;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: #6366f1;
}

QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {
    border-color: #6366f1;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #1e293b;
    color: #f8fafc;
    selection-background-color: #6366f1;
    border: 1px solid #334155;
    border-radius: 6px;
}

/* Checkbox */
QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 1px solid #475569;
    border-radius: 4px;
    background-color: #0f172a;
}

QCheckBox::indicator:checked {
    background-color: #6366f1;
    border-color: #6366f1;
}

/* Table Widget */
QTableWidget {
    background-color: #0f172a;
    gridline-color: #334155;
    border: 1px solid #334155;
    border-radius: 6px;
    color: #f8fafc;
}

QHeaderView::section {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 6px;
    border: none;
    border-right: 1px solid #334155;
    border-bottom: 1px solid #334155;
    font-weight: 600;
}

QTableWidget::item:selected {
    background-color: #334155;
    color: #6366f1;
}

/* Scroll Area & Bars */
QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background-color: #0f172a;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #334155;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Status Bar */
QStatusBar {
    background-color: #1e293b;
    color: #94a3b8;
    border-top: 1px solid #334155;
}

QProgressBar {
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
    background-color: #0f172a;
    color: #f8fafc;

}

QProgressBar::chunk {
    background-color: #6366f1;
    border-radius: 3px;
}
"""

LIGHT_THEME_QSS = """
/* QuEst Modern Light Theme */
QMainWindow, QDialog {
    background-color: #f8fafc;
    color: #0f172a;
}

QWidget {
    color: #1e293b;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 13px;
}

QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e2e8f0;
    padding: 4px;
    spacing: 6px;
}

QToolButton {
    background-color: #f1f5f9;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 6px 12px;
    font-weight: 600;
}

QToolButton:hover {
    background-color: #e2e8f0;
    border-color: #6366f1;
}

QTabWidget::pane {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background-color: #ffffff;
}

QTabBar::tab {
    background-color: #f1f5f9;
    color: #64748b;
    border: 1px solid #e2e8f0;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background-color: #ffffff;
    color: #4f46e5;
    border-color: #4f46e5;
    border-bottom: 2px solid #4f46e5;
}

QPushButton#CollapsibleToggle {
    background-color: #f1f5f9;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 8px;
    text-align: left;
    font-weight: 700;
}

QPushButton {
    background-color: #4f46e5;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 6px 14px;
    font-weight: 600;
}

QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {
    background-color: #ffffff;
    color: #0f172a;
    border: 1px solid #cbd5e1;
    border-radius: 6px;
    padding: 5px 8px;
}
"""


def apply_theme(app_or_widget, dark: bool = True) -> None:
    """Apply modern dark or light stylesheet to QApplication or QWidget."""
    qss = DARK_THEME_QSS if dark else LIGHT_THEME_QSS
    app_or_widget.setStyleSheet(qss)
