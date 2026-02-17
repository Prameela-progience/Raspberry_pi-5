''' 
FILE:object_detection.py
'''
import sys
import cv2
import time
import numpy as np
from ultralytics import YOLO
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QStackedWidget, QWidget,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QLineEdit, QTextEdit, QScrollArea,
    QProgressBar, QComboBox, QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QColor


import os
from ultralytics import YOLO



def load_yolo_model():
    # Get src folder path
    src_dir = os.path.dirname(os.path.abspath(__file__))
    #print(f"src_dir:{src_dir}")
    # Go to project root (one level up)
    project_root = os.path.dirname(src_dir)
    #print(f"project_root:{project_root}")
    # Go into models folder
    models_dir = os.path.join(project_root, "models")
    #print(f"models_dir:{models_dir}")

    custom_model_path = os.path.join(models_dir, "best.pt")
    fallback_model_path = os.path.join(models_dir, "yolov8n.pt")

    if os.path.exists(custom_model_path):
        print(f"[INFO] Loading custom model: {custom_model_path}")
        model = YOLO(custom_model_path)

    elif os.path.exists(fallback_model_path):
        print(f"[INFO] Custom model not found → loading fallback model")
        model = YOLO(fallback_model_path)

    else:
        raise FileNotFoundError(
            "No model file found in models folder!"
        )

    return model


