"""
FILE: object_detection_inference_engine.py

DESCRIPTION:
Media processing threads for video capture, sampling, and AI inference using YOLO.
Supports:
- Time-based sampling
- Frame-based sampling
- Quantitative instrumentation (FPS, frame drop, validation)
- Dynamic validation duration
- Live camera feeds and video files
- PyQt6 GUI integration
"""

import cv2
import time
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from object_detection import load_yolo_model


# =========================================================
# TIME-BASED SAMPLING THREAD
# =========================================================
class MediaProcessorThread_TimeBased(QThread):
    """
    Thread for capturing frames and performing time-based sampling.
    Includes instrumentation for FPS, total frames, and validation report.
    """

    # Signals
    signal_raw_stream = pyqtSignal(QPixmap)
    signal_sampled_audit_stream = pyqtSignal(QPixmap)
    signal_object_detection_result_stream = pyqtSignal(QPixmap)
    signal_error = pyqtSignal(str)
    signal_violation_image = pyqtSignal(QPixmap)
    signal_violation_alert=pyqtSignal(str)


    def __init__(self, media_source=0, target_processing_fps=5, validation_duration_sec=30.0):
        super().__init__()

        # Media source
        self.media_source = media_source

        # Sampling configuration
        self.target_processing_fps = target_processing_fps
        self.sampling_interval_sec = 1.0 / target_processing_fps

        # Thread control
        self.thread_active_status = True
        self.object_detection_enabled = False

        # YOLO model (loaded once)
        self.detection_model = load_yolo_model()

        # Time control
        self.last_sample_timestamp = 0.0

        # Instrumentation
        self.raw_frame_count_total = 0
        self.sampled_frame_count_total = 0
        self.raw_frame_count_1s = 0
        self.sampled_frame_count_1s = 0
        self.fps_window_start_time = 0.0
        self.validation_start_time = 0.0
        self.validation_duration_sec = validation_duration_sec

        # Video info
        self.video_reported_duration_sec = 0.0
        self.video_end_reached = False
        self.actual_processing_duration_sec = 0.0

        # ---- Alert suppression (1 second per class) ----
        self.violation_last_alert_time = {}   # {label: last_alert_timestamp}
        self.violation_suppression_sec = 1.0  # 1 second rule

        # ---- Store unique violations for final summary ----
        self.detected_violation_types = set()


    # ------------------------- PUBLIC METHODS -------------------------
    def update_inference_fps(self, new_fps: int):
        """Dynamically update sampling FPS."""
        if new_fps > 0:
            self.inference_target_fps = new_fps
            self.sampling_interval_sec = 1.0 / new_fps

    def toggle_object_detection(self, activation_state: bool):
        """Enable or disable AI inference."""
        self.object_detection_enabled = activation_state

    def terminate_thread(self):
        """Gracefully stop the thread."""
        self.thread_active_status = False

    # ------------------------- THREAD EXECUTION ------------------------
    def run(self):
        video_capture = cv2.VideoCapture(self.media_source)

        # Detect video file duration if applicable
        if isinstance(self.media_source, str):
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps > 0 and frame_count > 0:
                self.video_reported_duration_sec = frame_count / fps

        self.last_sample_timestamp = time.perf_counter()
        self.fps_window_start_time = self.last_sample_timestamp
        self.validation_start_time = self.last_sample_timestamp

        while self.thread_active_status:
            success, frame_bgr = video_capture.read()
            current_time = time.perf_counter()

            if not success:
                if isinstance(self.media_source, int):  # Camera
                    self.signal_error.emit("Camera disconnected. Attempting reconnect...")
                    video_capture.release()
                    time.sleep(1)

                    video_capture = cv2.VideoCapture(self.media_source)

                    if not video_capture.isOpened():
                        self.signal_error.emit("Reconnection failed. Stopping thread.")
                        break
                    else:
                        self.signal_error.emit("Camera reconnected successfully.")
                        continue
                else:
                    self.video_end_reached = True
                    break


            # RAW FRAME PATH
            self.raw_frame_count_total += 1
            self.raw_frame_count_1s += 1
            raw_pixmap = self.convert_cv_to_qpixmap(frame_bgr)
            self.signal_raw_stream.emit(raw_pixmap)

            # TIME-BASED SAMPLING
            if (current_time - self.last_sample_timestamp) >= self.sampling_interval_sec:
                self.last_sample_timestamp += self.sampling_interval_sec
                self.sampled_frame_count_total += 1
                self.sampled_frame_count_1s += 1
                self.signal_sampled_audit_stream.emit(raw_pixmap)

                if self.object_detection_enabled:
                    results = self.detection_model.predict(frame_bgr, conf=0.4, verbose=False)
                    frame_with_detections = results[0].plot()
                    frame_with_detections_pixmap = self.convert_cv_to_qpixmap(frame_with_detections)
                    self.signal_object_detection_result_stream.emit(frame_with_detections_pixmap)
                    # 1. Create a fresh copy for the "Violations Only" view
                    violation_display_frame = frame_bgr.copy()
                    violation_detected_this_frame = False
                    
                    violation_list = ["NO-Mask", "NO-Hardhat", "NO-Safety Vest"] 
                        
                    for result in results:
                        for box in result.boxes:
                            class_id = int(box.cls[0])
                            label = self.detection_model.names[class_id]
                            confidence = float(box.conf[0])

                            if label in violation_list:
                                violation_detected_this_frame = True
                                
                                # --- Extract coordinates ---
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                
                                # --- Draw Custom Box (No Conf Score) ---
                                # Red Box
                                cv2.rectangle(violation_display_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                # Label background
                                cv2.rectangle(violation_display_frame, (x1, y1 - 25), (x1 + 150, y1), (0, 0, 255), -1)
                                # Label text only
                                cv2.putText(violation_display_frame, label, (x1 + 5, y1 - 8), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                                # --- Send to Text Box ---
                                current_alert_time = time.perf_counter()

                                # Check last alert time for this violation class
                                last_time = self.violation_last_alert_time.get(label, 0)

                                if (current_alert_time - last_time) >= self.violation_suppression_sec:

                                    # Update last alert time
                                    self.violation_last_alert_time[label] = current_alert_time

                                    # Store unique violation type
                                    self.detected_violation_types.add(label)

                                    # Emit alert
                                    timestamp = time.strftime("%H:%M:%S")
                                    alert_text = f"[{timestamp}] VIOLATION: {label}"
                                    self.signal_violation_alert.emit(alert_text)


                    # 2. If a violation was found, emit the specific violation image
                    if violation_detected_this_frame:
                        violation_pixmap = self.convert_cv_to_qpixmap(violation_display_frame)
                        self.signal_violation_image.emit(violation_pixmap)

            # 1-second FPS window
            if (current_time - self.fps_window_start_time) >= 1.0:
                '''
                print(
                    f"[FPS] Input_frame_count_sec={self.raw_frame_count_1s} | "
                    f"Sampled_frame_count_sec={self.sampled_frame_count_1s} | "
                    f"user_selected_fps={self.target_processing_fps}"
                )
                '''
                self.raw_frame_count_1s = 0
                self.sampled_frame_count_1s = 0
                self.fps_window_start_time = current_time

            # Validation time limit
            if (current_time - self.validation_start_time) >= self.validation_duration_sec:
                break

        video_capture.release()
        self.actual_processing_duration_sec = current_time - self.validation_start_time

        # Frame drop ratio
        if self.video_end_reached:
            drop_ratio = 0.0
        elif self.raw_frame_count_total > 0:
            drop_ratio = 1.0 - (self.sampled_frame_count_total / self.raw_frame_count_total)
        else:
            drop_ratio = 0.0

        # ---- FINAL VIOLATION SUMMARY (No counts) ----
        if self.detected_violation_types:

            unique_string = ", ".join(self.detected_violation_types)

            final_summary = (
                "\nFINAL SUMMARY:\n"
                f"All violations are: {unique_string}"
            )

            #print(final_summary)

            # Emit to GUI
            self.signal_violation_alert.emit(final_summary)

        # Final report
        print("\n===== SAMPLING VALIDATION REPORT =====")
        print(f"Configured duration : {self.validation_duration_sec:.1f} sec")
        print(f"Actual duration     : {self.actual_processing_duration_sec:.1f} sec")
        if self.video_reported_duration_sec > 0:
            print(f"Video file duration : {self.video_reported_duration_sec:.1f} sec")
        print(f"Total raw frames    : {self.raw_frame_count_total}")
        print(f"Total sampled frames: {self.sampled_frame_count_total}")
        print(f"Frame drop ratio    : {drop_ratio:.2f}")
        print(
            "Termination reason  : Video file ended" 
            if self.video_end_reached else
            "Termination reason  : Validation time completed"
        )
        print("=====================================\n")

    # ------------------------- IMAGE CONVERSION ------------------------
    def convert_cv_to_qpixmap(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qt_image)

'''
# =========================================================
# FRAME-BASED SAMPLING THREAD
# =========================================================
class MediaProcessorThread_FrameBased(QThread):
    """
    Thread for capturing frames and performing frame-based sampling.
    """

    signal_raw_stream = pyqtSignal(QPixmap)
    signal_sampled_audit_stream = pyqtSignal(QPixmap)
    signal_inference_result_stream = pyqtSignal(QPixmap)

    def __init__(self, media_source=0, target_processing_fps=5):
        super().__init__()

        self.media_source = media_source
        self.inference_target_fps = target_processing_fps
        self.thread_active_status = True
        self.detection_enabled = False
        self.source_native_fps = 30
        self.frame_count = 0
        self.detection_model = load_yolo_model()
        self.frame_step = 1

    def update_inference_fps(self, new_fps):
        self.inference_target_fps = new_fps
        self.frame_step = max(1, int(self.source_native_fps / new_fps))

    def toggle_inference_engine(self, activation_state: bool):
        self.detection_enabled = activation_state

    def terminate_thread(self):
        self.thread_active_status = False

    def run(self):
        cap = cv2.VideoCapture(self.media_source)
        if isinstance(self.media_source, str):
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps > 0:
                self.source_native_fps = int(fps)
        self.frame_step = max(1, int(self.source_native_fps / self.inference_target_fps))

        while self.thread_active_status:
            ret, frame = cap.read()
            if not ret:
                break

            self.frame_count += 1
            pixmap = self.convert_cv_to_qpixmap(frame)
            self.signal_raw_stream.emit(pixmap)

            if self.frame_count % self.frame_step == 0:
                self.signal_sampled_audit_stream.emit(pixmap)
                if self.detection_enabled:
                    results = self.detection_model.predict(frame, conf=0.4, verbose=False)
                    frame_with_detections = results[0].plot()
                    self.signal_inference_result_stream.emit(
                        self.convert_cv_to_qpixmap(frame_with_detections)
                    )

            time.sleep(1 / self.source_native_fps)

        cap.release()

    def convert_cv_to_qpixmap(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qt_image)
'''