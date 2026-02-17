"""
FILE: Gui_components.py

DESCRIPTION:
Reusable GUI components for the PPE Monitoring application.
Contains navigation-related pages only (Home & Mode Selection).
Safe for direct production deployment.
"""

import sys
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame ,QSizePolicy
)
from PyQt6.QtCore import Qt


# =========================================================
# HOME / LANDING PAGE
# =========================================================
class HomePage(QWidget):
    """
    Initial landing page of the application.
    Displays branding and primary navigation actions.
    """

    def __init__(self, main_controller):
        """
        :param main_controller:
            Reference to the main application controller.
            Used for page navigation and app lifecycle control.
        """
        super().__init__()

        self.main_controller = main_controller

        # Root layout centered on screen
        root_layout = QVBoxLayout(self)
        root_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Card-style container
        container_card = QFrame()
        container_card.setSizePolicy(
        QSizePolicy.Policy.Preferred,
        QSizePolicy.Policy.Preferred
        )
        container_card.setMaximumWidth(600)
        container_card.setStyleSheet(
            "background-color: #1e293b;"
            "border: 2px solid #334155;"
            "border-radius: 15px;"
        )

        card_layout = QVBoxLayout(container_card)

        # Application title
        title_label = QLabel("PPE MONITORING HUB")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(
            "color: #38bdf8;"
            "font-size: 28px;"
            "font-weight: bold;"
        )

        # Start application button
        start_application_button = QPushButton(" ▶ START APPLICATION")
        start_application_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        start_application_button.setStyleSheet(
            "background-color: #0ea5e9;"
            "color: white;"
            "font-size: 18px;"
            "border-radius: 10px;"
        )
        start_application_button.clicked.connect(
            lambda: self.main_controller.navigate_to_page("MODE_SELECTION")
        )

        # Exit application button
        exit_application_button = QPushButton(" ⊘ EXIT")
        exit_application_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred
        )
        exit_application_button.setStyleSheet(
            "background-color: #ef4444;"
            "color: white;"
            "font-size: 18px;"
            "border-radius: 10px;"
        )
        exit_application_button.clicked.connect(sys.exit)

        # Assemble card
        card_layout.addWidget(title_label)
        card_layout.addWidget(start_application_button)
        card_layout.addWidget(exit_application_button)

        root_layout.addWidget(container_card)


# =========================================================
# MODE SELECTION PAGE
# =========================================================
class ModeSelectionPage(QWidget):
    """
    Allows the user to select the monitoring mode:
    - Vision Audit (static / low FPS)
    - Live Video Stream
    - Recorded Video File
    """

    def __init__(self, main_controller):
        super().__init__()

        self.main_controller = main_controller

        # Root layout
        root_layout = QVBoxLayout(self)
        root_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # -------------------------
        # HEADER (HOME NAVIGATION)
        # -------------------------
        header_layout = QHBoxLayout()

        home_navigation_button = QPushButton("⌂ HOME MENU")
        home_navigation_button.setStyleSheet(
            "background-color: #1e293b;"
            "color: white;"
            "padding: 8px 15px;"
        )
        home_navigation_button.clicked.connect(
            lambda: self.main_controller.navigate_to_page("HOME")
        )

        header_layout.addWidget(home_navigation_button)
        header_layout.addStretch()

        root_layout.addLayout(header_layout)

        # -------------------------
        # PAGE TITLE
        # -------------------------
        page_title_label = QLabel("CHOOSE MONITORING MODE")
        page_title_label.setStyleSheet(
            "color: white;"
            "font-size: 24px;"
            "font-weight: bold;"
        )

        root_layout.addWidget(
            page_title_label,
            alignment=Qt.AlignmentFlag.AlignCenter
        )

        # -------------------------
        # MODE SELECTION BUTTONS
        # -------------------------
        mode_buttons_layout = QHBoxLayout()

        mode_buttons_layout.addWidget(
            self.create_mode_option_button(
                label_text="VISION AUDIT\n(Static)",
                accent_color="#0ea5e9",
                mode_identifier="VISION"
            )
        )

        mode_buttons_layout.addWidget(
            self.create_mode_option_button(
                label_text="VIDEO STREAM\n(Live)",
                accent_color="#a78bfa",
                mode_identifier="VIDEO"
            )
        )

        mode_buttons_layout.addWidget(
            self.create_mode_option_button(
                label_text="RECORDED VIDEO\n(File Processing)",
                accent_color="#f59e0b",
                mode_identifier="RECORDED"
            )
        )

        root_layout.addLayout(mode_buttons_layout)

    # =====================================================
    # UI FACTORY METHODS
    # =====================================================
    def create_mode_option_button(self, label_text, accent_color, mode_identifier):
        """
        Creates a standardized mode-selection button.

        :param label_text:
            Display text shown on the button.

        :param accent_color:
            Highlight color for border and text.

        :param mode_identifier:
            Internal mode ID passed to the controller.
        """
        mode_button = QPushButton(label_text)
        mode_button.setMinimumSize(180, 180)
        mode_button.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        #mode_buttons_layout.addWidget(button1, 1)
        #mode_buttons_layout.addWidget(button2, 1)
        #mode_buttons_layout.addWidget(button3, 1)

        mode_button.setStyleSheet(
            f"background: #1e293b;"
            f"color: {accent_color};"
            f"border: 3px solid {accent_color};"
            f"border-radius: 20px;"
            f"font-weight: bold;"
        )

        mode_button.clicked.connect(
            lambda: self.main_controller.initialize_monitoring_session(mode_identifier)
        )

        return mode_button
