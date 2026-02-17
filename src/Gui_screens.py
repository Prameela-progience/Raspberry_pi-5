'''
FILE: Gui_screens.py

DESCRIPTION:
-------------
Defines the MonitoringPage UI used in the PPE Surveillance AI application.
'''

import sys
import cv2
import time
import numpy as np
from ultralytics import YOLO

# PyQt6 UI components
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QLineEdit, QTextEdit, QScrollArea,
    QProgressBar, QComboBox, QFileDialog ,QSizePolicy
)

# PyQt6 core utilities
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor

# Background media processing thread
from object_detection_inference_engine import MediaProcessorThread_TimeBased
from rag_handler import RAGWorker
from rag_logic import rag_retriever


class MonitoringPage(QWidget):
    def __init__(self, main_controller, mode_type):
        super().__init__()
        self.main_controller = main_controller
        self.mode_type = mode_type
        self.media_processor_thread = None
        self._build_user_interface()
        self.recent_violations_context = []
    
    def _build_user_interface(self):
        combo_box_style = (
            "QComboBox { color: white; background-color: #1e293b; "
            "border: 1px solid #334155; padding: 2px 5px; min-width: 100px; }"
        )

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(5, 5, 5, 5) # Added tight margins per sketch
        root_layout.setSpacing(5)

        # =================================================
        # HEADER BAR (Home, Backend, FPS)
        # =================================================
        header_layout = QHBoxLayout()
        home_nav_button = self._create_styled_button("⌂ HOME MENU", "#1e293b")
        home_nav_button.clicked.connect(lambda: self.main_controller.navigate_to_page("HOME"))
        header_layout.addWidget(home_nav_button)
        header_layout.addStretch()

        header_layout.addWidget(QLabel("Backend:", styleSheet="color: white; font-weight: bold;"))
        self.backend_selector = QComboBox()
        self.backend_selector.setStyleSheet(combo_box_style)
        self.backend_selector.addItems(["V4L2", "FFMPEG", "GSTREAMER"])
        header_layout.addWidget(self.backend_selector)

        fps_label_text = "Vision FPS:" if self.mode_type == "VISION" else "Video FPS:"
        header_layout.addWidget(QLabel(fps_label_text, styleSheet="color: white; font-weight: bold;"))

        self.inference_fps_selector = QComboBox()
        self.inference_fps_selector.setStyleSheet(combo_box_style)
        self.inference_fps_selector.addItems(["1", "5", "10", "30"])
        self.inference_fps_selector.currentTextChanged.connect(self.handle_fps_adjustment)
        header_layout.addWidget(self.inference_fps_selector)
        root_layout.addLayout(header_layout)

        # =================================================
        # MAIN LAYOUT (Sidebar Buttons + Center Grid + Report)
        # =================================================
        main_content_layout = QHBoxLayout()

        # --- SIDEBAR (ACTIVATE, RUN, HALT) ---
        sidebar_button_stack = QVBoxLayout()
        sidebar_button_stack.setSpacing(10)
        
        self.stream_toggle_button = self._create_styled_button("ACTIVATE CAMERA", "#334155")
        self.stream_toggle_button.clicked.connect(self.handle_stream_activation)
        
        self.start_analytics_button = self._create_styled_button(f"RUN AUDIT", "#065f46")
        self.start_analytics_button.clicked.connect(lambda: self.toggle_analytics_processing(True))
        
        self.stop_analytics_button = self._create_styled_button("HALT ANALYTICS", "#991b1b")
        self.stop_analytics_button.clicked.connect(lambda: self.toggle_analytics_processing(False))

        sidebar_button_stack.addWidget(self.stream_toggle_button)
        sidebar_button_stack.addWidget(self.start_analytics_button)
        sidebar_button_stack.addWidget(self.stop_analytics_button)
        sidebar_button_stack.addStretch() # Push buttons to top per sketch
        
        main_content_layout.addLayout(sidebar_button_stack, 1) # Sidebar takes least width

        # --- CENTER COLUMN (Query + 4 Boxes) ---
        center_column = QVBoxLayout()
        
        # Thinner Query box aligned to grid
        self.audit_search_input = QLineEdit()
        self.audit_search_input.setPlaceholderText("Enter query...")
        self.audit_search_input.setFixedHeight(25) 
        self.audit_search_input.setStyleSheet("background-color: #1e293b; color: white; border: 1px solid #334155;")
        center_column.addWidget(self.audit_search_input)

        self.audit_search_input.returnPressed.connect(self.handle_rag_request)

        # 4-Box Grid
        grid_layout = QVBoxLayout()
        
        top_row = QHBoxLayout()
        self.raw_stream_view = self._create_video_frame("SYSTEM OFFLINE", "#334155")
        self.sampled_stream_view = self._create_video_frame("VISION FEED", "#a78bfa")
        top_row.addWidget(self.raw_stream_view)
        top_row.addWidget(self.sampled_stream_view)
        
        bottom_row = QHBoxLayout()
        self.detection_panel = self._create_status_panel("OBJECT DETECTION", "#0ea5e9")
        self.violation_panel = self._create_status_panel("VIOLATION ISOLATION", "#ef4444")
        bottom_row.addWidget(self.detection_panel)
        bottom_row.addWidget(self.violation_panel)
        
        grid_layout.addLayout(top_row, 1)    # Top row slightly smaller per earlier request
        grid_layout.addLayout(bottom_row, 1) # Bottom row (AI) larger
        center_column.addLayout(grid_layout)
        
        main_content_layout.addLayout(center_column, 6) # Center grid takes middle width
        '''
        # --- RIGHT SIDEBAR (Report) ---
        report_column = QVBoxLayout()
        report_column.addWidget(QLabel("SAFETY AUDITOR REPORT...", styleSheet="color: white; font-weight: bold;"))
        self.audit_report_textbox = QTextEdit(styleSheet="background: black; color: #38bdf8; border: 1px solid #334155;")
        report_column.addWidget(self.audit_report_textbox)
        
        main_content_layout.addLayout(report_column, 3) # Right sidebar width
        '''
        # --- RIGHT SIDEBAR (Report) ---
        report_column = QVBoxLayout()
        report_column.setSpacing(5) 
        
        report_column.addWidget(QLabel("SAFETY AUDITOR REPORT...", styleSheet="color: white; font-weight: bold;"))

        # Violations
        report_column.addWidget(QLabel("Violations:", styleSheet="color: #ef4444; font-weight: bold; font-size: 10px;"))
        self.violations_report_textbox = QTextEdit()
        self.violations_report_textbox.setStyleSheet("background: black; color: #ef4444; border: 1px solid #334155;")
        self.violations_report_textbox.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        report_column.addWidget(self.violations_report_textbox, 1) 

        # RAG Output
        report_column.addWidget(QLabel("RAG Output:", styleSheet="color: #38bdf8; font-weight: bold; font-size: 10px;"))
        self.rag_output_textbox = QTextEdit()
        self.rag_output_textbox.setStyleSheet("background: black; color: #38bdf8; border: 1px solid #334155;")
        self.rag_output_textbox.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        report_column.addWidget(self.rag_output_textbox, 1) 
        
        main_content_layout.addLayout(report_column, 3)
        # Add the Centered Wrapper to handle 7-inch "squeeze"
        centered_wrapper = QHBoxLayout()
        centered_wrapper.addStretch(1)
        centered_wrapper.addLayout(main_content_layout, 12)
        centered_wrapper.addStretch(1)

        root_layout.addLayout(centered_wrapper)

        
    # =====================================================
    # UI HELPERS
    # =====================================================
    # In your _create_video_frame helper:
    def _create_video_frame(self, placeholder_text, border_color):
        label = QLabel(placeholder_text, alignment=Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(f"background: black; border: 2px solid {border_color}; color: white;")
        
        # Use a standard minimum size for all four boxes
        label.setMinimumSize(100, 80) 
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return label

    def _create_status_panel(self, title_text, theme_color):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        title = QLabel(title_text, styleSheet="color: white; font-weight: bold; font-size: 10px;")
        progress_bar = QProgressBar()
        progress_bar.setFixedHeight(4) # Keep this very thin
        
        video_canvas = self._create_video_frame("IDLE", theme_color)
        
        layout.addWidget(title)
        layout.addWidget(progress_bar)
        layout.addWidget(video_canvas, 1) # The '1' ensures the video frame expands to fill space
    
        container.canvas = video_canvas
        return container
    def _create_styled_button(self, text, color):
        btn = QPushButton(text)
        btn.setFixedHeight(25) # Thinner buttons to save vertical space
        btn.setStyleSheet(f"background-color: {color}; color: white; font-weight: bold; font-size: 10px;")
        return btn

    # =====================================================
    # THREAD & DISPLAY UPDATES
    # =====================================================
    def toggle_analytics_processing(self, run_state):
        if self.media_processor_thread:
            self.media_processor_thread.toggle_object_detection(run_state)
            if not run_state:
                self.detection_panel.canvas.setPixmap(QPixmap())
                self.detection_panel.canvas.setText("IDLE")

    def handle_fps_adjustment(self, fps_text):
        if self.media_processor_thread and self.media_processor_thread.isRunning():
            try:
                self.media_processor_thread.update_inference_fps(int(fps_text))
            except: pass

    def handle_stream_activation(self):
        if self.media_processor_thread and self.media_processor_thread.isRunning():
            self.media_processor_thread.terminate_thread()
            self.stream_toggle_button.setText("OPEN VIDEO FILE" if self.mode_type == "RECORDED" else "ACTIVATE CAMERA")
            return

        source = 0
        if self.mode_type == "RECORDED":
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Source", "", "Videos (*.mp4 *.avi *.mov)")
            if not file_path: return
            source = file_path

        self.media_processor_thread = MediaProcessorThread_TimeBased(
            media_source=source,
            target_processing_fps=int(self.inference_fps_selector.currentText())
        )

        self.media_processor_thread.signal_raw_stream.connect(self.refresh_raw_display)
        self.media_processor_thread.signal_sampled_audit_stream.connect(self.refresh_sampled_display)
        self.media_processor_thread.signal_object_detection_result_stream.connect(self.refresh_inference_display)
        # 2. Connect the "Violation Isolation" feed (Filtered image)
        self.media_processor_thread.signal_violation_image.connect(self.refresh_violation_display)
        
        self.media_processor_thread.signal_violation_alert.connect(self.update_violations_report)

        self.media_processor_thread.start()
        self.stream_toggle_button.setText("STOP FEED")

    def refresh_raw_display(self, pixmap):
        self.raw_stream_view.setPixmap(pixmap.scaled(self.raw_stream_view.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def refresh_sampled_display(self, pixmap):
        self.sampled_stream_view.setPixmap(pixmap.scaled(self.sampled_stream_view.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def refresh_inference_display(self, pixmap):
        canvas = self.detection_panel.canvas
        canvas.setPixmap(pixmap.scaled(canvas.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def refresh_violation_display(self, pixmap):
        # self.violation_panel is the container, .canvas is the actual QLabel
        canvas = self.violation_panel.canvas
        
        # Scale the image to fit the box while keeping it clear
        scaled_pixmap = pixmap.scaled(
            canvas.size(), 
            Qt.AspectRatioMode.KeepAspectRatio, 
            Qt.TransformationMode.SmoothTransformation
        )
        canvas.setPixmap(scaled_pixmap)
         
    # Update your existing update_violations_report method
    def update_violations_report(self, alert_text):

        # 1️ Show violations in UI
        #self.violations_textbox.append(alert_text)
        self.violations_report_textbox.append(alert_text)


        # 2️ Store context for RAG
        if not hasattr(self, "recent_violations_context"):
            self.recent_violations_context = []

        self.recent_violations_context.append(alert_text)

        # 3️ Auto-trigger RAG only for FINAL SUMMARY
        if "FINAL SUMMARY" in alert_text:

            #print("Data being sent to RAG:")
            #print(self.recent_violations_context)

            self.rag_output_textbox.clear()

            self.rag_thread = RAGWorker(
                "",  # No user query
                self.recent_violations_context,
                rag_retriever
            )

            self.rag_thread.signal_response_ready.connect(
                self.display_rag_output
            )

            self.rag_thread.start()



    # 2. Add this new method to MonitoringPage:
    def handle_rag_request(self):
        query = self.audit_search_input.text().strip()
        if not query:
            return

        # Display status in the GUI
        self.rag_output_textbox.append(f"<b>User:</b> {query}")
        self.rag_output_textbox.append("<i>Analyzing violations and generating response...</i>")
        self.audit_search_input.clear()


        # Start the thread
        self.rag_thread = RAGWorker(query, self.recent_violations_context, rag_retriever)
        self.rag_thread.signal_response_ready.connect(self.display_rag_output)
        self.rag_thread.start()

    def display_rag_output(self, response):
        # Print response to the RAG Output box
        self.rag_output_textbox.append(f"<span style='color: #38bdf8;'><b>AI Auditor:</b> {response}</span>")
        self.rag_output_textbox.append("-" * 30)