'''
FILE: main_application.py

DESCRIPTION:
---------------
Main application entry point and UI orchestrator for the PPE Surveillance AI system.

This module:
- Initializes the Qt application
- Creates the main window container
- Manages page navigation using a stacked widget
- Acts as the central controller between Home, Mode Selection,
  and Monitoring pages

This file is intentionally lightweight and only coordinates high-level
application flow. Heavy processing and business logic are delegated to
other modules.
'''

import sys
import cv2
import time
import numpy as np
from ultralytics import YOLO

# PyQt6 core UI components
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QLineEdit, QTextEdit, QScrollArea,
    QProgressBar, QComboBox, QFileDialog ,QSizePolicy
)

# PyQt6 core utilities
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor

# Application-specific UI pages
from Gui_screens import MonitoringPage
from Gui_components import ModeSelectionPage
from Gui_components import HomePage
from Gui_responsive_wrapper import apply_proportional_scaling



# =========================================================
# APPLICATION ORCHESTRATOR
# =========================================================
class PPEMonitoringApp(QMainWindow):
    """
    Master controller class for the PPE Surveillance AI application.

    Responsibilities:
    ------------------
    - Owns the main application window
    - Maintains a central QStackedWidget for page navigation
    - Creates and switches between high-level application screens
    - Acts as the entry point for launching monitoring sessions

    This class intentionally does NOT contain:
    - AI inference logic
    - Video processing logic
    - UI component rendering details

    Those responsibilities are delegated to dedicated modules.
    """

    def __init__(self):
        """
        Initializes the main application window and registers core pages.
        """
        super().__init__()
        #self.setup_ui()   # your existing UI code

        

        # Window metadata
        self.setWindowTitle("PPE Surveillance AI")
        #self.resize(1500, 900)
        #self.showFullScreen()


        # Global application background styling
        self.setStyleSheet("background-color: #0f172a;")

        # Central stacked widget enables page-based navigation
        # Only one page is visible at a time
        self.page_switching = QStackedWidget()
        self.setCentralWidget(self.page_switching)

        # Preload static application pages
        # These pages persist for the entire application lifetime
        self.app_pages = {
            "HOME": HomePage(self),
            "MODE_SELECTION": ModeSelectionPage(self)
        }

        # Register pages with the stacked widget
        for page in self.app_pages.values():
            self.page_switching.addWidget(page)

        # Default landing page
        self.navigate_to_page("HOME")

    def navigate_to_page(self, page_id):
        """
        Switches the currently visible page.

        Args:
            page_id (str): Key identifying the target page
                           (e.g., 'HOME', 'MODE_SELECTION')
        """
        self.page_switching.setCurrentWidget(self.app_pages[page_id])

    def initialize_monitoring_session(self, mode_id):
        """
        Dynamically creates and launches a monitoring session.

        This method:
        - Instantiates a MonitoringPage for the selected mode
        - Registers it with the stacked widget
        - Switches the UI to the monitoring screen

        Monitoring pages are created at runtime because they depend
        on the selected operational mode (camera, video, vision, etc.).

        Args:
            mode_id (str): Identifier for the selected monitoring mode
        """
        monitoring_page = MonitoringPage(self, mode_id)
        self.page_switching.addWidget(monitoring_page)
        self.page_switching.setCurrentWidget(monitoring_page)


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================
if __name__ == "__main__":
    """
    Application bootstrap sequence.

    Steps:
    ------
    1. Create QApplication instance
    2. Initialize the main application window
    3. Display the UI
    4. Enter the Qt event loop
    """

    application = QApplication(sys.argv)
    main_app_window = PPEMonitoringApp()
    apply_proportional_scaling(main_app_window)
    main_app_window.show()
    #main_app_window.showFullScreen()


    # Start Qt event loop and exit cleanly on close
    sys.exit(application.exec())
