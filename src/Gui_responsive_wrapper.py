"""
Uniform proportional GUI scaler
Maintains exact design proportions across devices.
"""

from PyQt6.QtGui import QGuiApplication
from PyQt6.QtCore import QSize

BASE_WIDTH = 1920
BASE_HEIGHT = 1080

def apply_proportional_scaling(main_window):
    screen = QGuiApplication.primaryScreen()
    screen_size = screen.size()

    current_width = screen_size.width()
    current_height = screen_size.height()

    scale_w = current_width / BASE_WIDTH
    scale_h = current_height / BASE_HEIGHT
    
    # Use the smaller scale to ensure nothing clips
    scale_factor = min(scale_w, scale_h)

    # Resize window to fit the physical screen
    main_window.resize(current_width, current_height)

    # Scale fonts with a 'Safety Floor' (minimum 9pt)
    font = main_window.font()
    new_font_size = max(9.0, font.pointSizeF() * scale_factor * 1.5) # Slight boost for small screens
    font.setPointSizeF(new_font_size)
    main_window.setFont(font)
