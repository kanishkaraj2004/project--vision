import tkinter as tk
from tkinter import ttk, font as tkfont, messagebox
import pyttsx3
import threading
import speech_recognition as sr
import cv2
import numpy as np
import os
import subprocess
import sys
from PIL import Image, ImageTk
import onnxruntime

class VisionIRLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Vision IRL")
        self.root.geometry("900x650")
        
        # Initialize variables
        self.VOICE_REGISTRATION_FILE = "voice_open.wav"
        self.KNOWN_FACE_ENCODING_FILE = "known_face_encoding.npy"
        self.ort_session = None
        self.MODEL_PATH = "face_det_lite-lightweight-face-detection-float (3).onnx"
        
        try:
            self.setup_ui()
            self.initialize_onnx_model()
            
            # Start authentication flow in a separate thread
            self.root.after(100, lambda: threading.Thread(target=self.run_authentication_flow, daemon=True).start())
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Failed to initialize application: {str(e)}")
            self.root.destroy()
            sys.exit(1)
        
    def initialize_onnx_model(self):
        """Initialize the ONNX face detection model"""
        try:
            # First try the specified path
            if not os.path.exists(self.MODEL_PATH):
                # If not found, try relative paths
                model_path = os.path.join(os.path.dirname(__file__), "facedetect.onnx")
                if not os.path.exists(model_path):
                    model_path = "facedetect.onnx"  # Try current directory
            
            if not os.path.exists(self.MODEL_PATH):
                raise FileNotFoundError(f"facedetect.onnx model file not found at {self.MODEL_PATH}")
                
            # Verify the model file is valid
            if os.path.getsize(self.MODEL_PATH) == 0:
                raise ValueError("Model file is empty")
                
            # Set ONNX session options
            so = onnxruntime.SessionOptions()
            so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            self.ort_session = onnxruntime.InferenceSession(self.MODEL_PATH, sess_options=so)
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_names = [output.name for output in self.ort_session.get_outputs()]
            self.add_message("ONNX model loaded successfully", "system")
        except Exception as e:
            self.add_message(f"Failed to load ONNX model: {str(e)}", "system")
            raise

    def setup_ui(self):
        """Set up the user interface"""
        # Create gradient background canvas
        self.canvas = tk.Canvas(self.root, width=900, height=650, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.draw_gradient("#FF6D00", "#FFC400")  # Orange to yellow gradient
        
        # Main container frame
        self.main_frame = tk.Frame(self.canvas, bg="#FFFFFF", bd=0)
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center", width=800, height=550)
        
        # Load fonts
        self.load_fonts()
        
        # Setup header with logo
        self.setup_header()
        
        # Setup message display
        self.setup_message_display()
        
        # Setup status bar
        self.setup_status_bar()
        
        # Add initial system message
        self.add_message("System initialized...", "system")

    def draw_gradient(self, color1, color2):
        """Draw gradient background"""
        for i in range(650):
            ratio = i / 650
            r = int((1-ratio) * int(color1[1:3], 16) + ratio * int(color2[1:3], 16))
            g = int((1-ratio) * int(color1[3:5], 16) + ratio * int(color2[3:5], 16))
            b = int((1-ratio) * int(color1[5:7], 16) + ratio * int(color2[5:7], 16))
            color = f"#{r:02x}{g:02x}{b:02x}"
            self.canvas.create_line(0, i, 900, i, fill=color)

    def load_fonts(self):
        """Load custom fonts"""
        try:
            self.title_font = tkfont.Font(family="Helvetica", size=24, weight="bold")
            self.subtitle_font = tkfont.Font(family="Helvetica", size=14)
            self.message_font = tkfont.Font(family="Consolas", size=12)
        except:
            self.title_font = tkfont.Font(family="Arial", size=24, weight="bold")
            self.subtitle_font = tkfont.Font(family="Arial", size=14)
            self.message_font = tkfont.Font(family="Courier", size=12)

    def setup_header(self):
        """Setup application header"""
        header_frame = tk.Frame(self.main_frame, bg="#FFFFFF")
        header_frame.pack(fill="x", padx=20, pady=10)
        
        # Logo and title
        logo_label = tk.Label(header_frame, text="VISION IRL", font=self.title_font, 
                            bg="#FFFFFF", fg="#333333")
        logo_label.pack(side="left")
        
        # Version label
        version_label = tk.Label(header_frame, text="v1.0", font=self.subtitle_font,
                                bg="#FFFFFF", fg="#888888")
        version_label.pack(side="right")

    def setup_message_display(self):
        """Setup message display area"""
        message_frame = tk.Frame(self.main_frame, bg="#FFFFFF")
        message_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # Message display text widget
        self.message_text = tk.Text(message_frame, wrap="word", font=self.message_font,
                                  bg="#F8F8F8", fg="#333333", padx=10, pady=10,
                                  state="disabled", height=15)
        scrollbar = ttk.Scrollbar(message_frame, command=self.message_text.yview)
        self.message_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.message_text.pack(fill="both", expand=True)

    def setup_status_bar(self):
        """Setup status bar at bottom"""
        status_frame = tk.Frame(self.main_frame, bg="#FFFFFF")
        status_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Ready", font=self.subtitle_font,
                                   bg="#FFFFFF", fg="#666666", anchor="w")
        self.status_label.pack(fill="x")

    def add_message(self, message, msg_type="system"):
        """Add message to display"""
        self.message_text.configure(state="normal")
        
        # Set color based on message type
        if msg_type == "system":
            self.message_text.insert("end", "[SYSTEM] ", "system")
            self.message_text.insert("end", f"{message}\n", "message")
        elif msg_type == "error":
            self.message_text.insert("end", "[ERROR] ", "error")
            self.message_text.insert("end", f"{message}\n", "message")
        else:
            self.message_text.insert("end", f"{message}\n", "message")
            
        self.message_text.configure(state="disabled")
        self.message_text.see("end")
        
    def speak(self, text):
        """Convert text to speech"""
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            self.add_message(f"Text-to-speech error: {str(e)}", "error")

    def voice_unlock(self):
        """Voice recognition authentication"""
        self.add_message("Voice authentication started...", "system")
        self.speak("Please say the unlock phrase")
        
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source, timeout=5)
                
            try:
                text = r.recognize_google(audio)
                self.add_message(f"You said: {text}", "system")
                
                # Check if the recognized text matches the expected phrase
                if "open" in text.lower():
                    self.add_message("Voice authentication successful", "system")
                    return True
                else:
                    self.add_message("Voice authentication failed", "system")
                    return False
                    
            except sr.UnknownValueError:
                self.add_message("Could not understand audio", "error")
                return False
            except sr.RequestError as e:
                self.add_message(f"Speech recognition error: {str(e)}", "error")
                return False
                
        except Exception as e:
            self.add_message(f"Voice unlock error: {str(e)}", "error")
            return False

    def preprocess_image(self, image):
        """Preprocess image according to ONNX model input"""

        # Convert to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Get the exact input shape expected by ONNX model
        input_shape = self.ort_session.get_inputs()[0].shape

        # Model format: [batch, channels, height, width]
        model_height = int(input_shape[2])
        model_width = int(input_shape[3])

        print("ONNX input shape:", input_shape)
        print("Resizing to:", model_width, "x", model_height)

        # OpenCV resize takes (width, height)
        image = cv2.resize(image, (model_width, model_height))

        # Normalize
        image = image.astype(np.float32) / 255.0

        # HWC -> CHW
        image = np.transpose(image, (2, 0, 1))

        # Add batch dimension
        image = np.expand_dims(image, axis=0)

        print("Final image shape:", image.shape)

        return image

    def detect_face(self, image):
        """Detect face using Qualcomm FaceDetLite ONNX model"""

        if self.ort_session is None:
            self.add_message("ONNX model not loaded", "system")
            return False

        try:
            # Preprocess image
            input_image = self.preprocess_image(image)

            # Run ONNX inference
            outputs = self.ort_session.run(
                self.output_names,
                {self.input_name: input_image}
            )

            if len(outputs) < 3:
                self.add_message(
                    "Face model returned unexpected outputs",
                    "error"
                )
                return False

            # Qualcomm FaceDetLite outputs:
            # 0 = heatmap
            # 1 = bounding boxes
            # 2 = landmarks
            hm = np.asarray(outputs[0])
            box = np.asarray(outputs[1])
            landmark = np.asarray(outputs[2])

            print("Heatmap shape:", hm.shape)
            print("Box shape:", box.shape)
            print("Landmark shape:", landmark.shape)

            # Apply sigmoid to heatmap
            hm = 1.0 / (1.0 + np.exp(-hm))

            # Remove batch/channel dimensions
            hm_2d = np.squeeze(hm)

            # Find highest-confidence face location
            max_index = np.unravel_index(
                np.argmax(hm_2d),
                hm_2d.shape
            )

            cy, cx = max_index
            confidence = float(hm_2d[cy, cx])

            print("Face confidence:", confidence)

            # Qualcomm reference uses 0.55 threshold
            if confidence < 0.55:
                return False

            # Optional: decode bounding box
            box_data = np.squeeze(box)

            # Expected shape after squeeze: (4, H, W)
            if box_data.ndim == 3 and box_data.shape[0] >= 4:

                x, y, r, b = box_data[:, cy, cx][:4]

                stride = 8

                left = (cx - x) * stride
                top = (cy - y) * stride
                right = (cx + r) * stride
                bottom = (cy + b) * stride

                print(
                    f"Face detected: confidence={confidence:.3f}, "
                    f"box=({left:.0f}, {top:.0f}, "
                    f"{right:.0f}, {bottom:.0f})"
                )

            return True

        except Exception as e:
            self.add_message(
                f"Face detection error: {str(e)}",
                "error"
            )
            return False

    def face_verification(self):
        """Face verification process"""
        self.add_message("Starting face verification...", "system")
        self.speak("Please look at the camera")
        
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Could not open video capture device")
                
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read frame from camera")
                
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face
            if self.detect_face(rgb_frame):
                self.add_message("Face verification successful", "system")
                cap.release()
                return True
            else:
                self.add_message("Face not detected", "system")
                cap.release()
                return False
                
        except Exception as e:
            self.add_message(f"Face verification error: {str(e)}", "error")
            if 'cap' in locals() and cap.isOpened():
                cap.release()
            return False

    def ask_to_repeat(self):
        """Ask user if they want to retry authentication"""
        self.add_message("Would you like to try again?", "system")
        self.speak("Authentication failed. Would you like to try again?")
        
        # In a real implementation, you would add UI elements for this
        # For now, we'll just restart the authentication flow after a delay
        self.root.after(3000, lambda: threading.Thread(target=self.run_authentication_flow, daemon=True).start())
    def launch_vision(self):
        """Launch the main object detection system"""

        self.add_message("Launching Vision System...", "system")
        self.speak("Vision System ready")

        try:
            # Get the folder where main.py is located
            backend_dir = os.path.dirname(os.path.abspath(__file__))

            # Path to obj.py
            obj_path = os.path.join(backend_dir, "obj.py")

            # Check whether obj.py exists
            if not os.path.exists(obj_path):
                raise FileNotFoundError(
                    f"obj.py not found at:\n{obj_path}"
                )

            self.add_message(
                "Starting object detection system...",
                "system"
            )

            # Launch obj.py using the SAME Python/venv
            subprocess.Popen(
                [sys.executable, obj_path],
                cwd=backend_dir
            )

            self.add_message(
                "Vision System is now running",
                "system"
            )

        except Exception as e:
            self.add_message(
                f"Failed to launch Vision System: {str(e)}",
                "error"
            )

    def run_authentication_flow(self):
        """Main authentication flow"""
        try:
            self.add_message("Starting authentication process...", "system")
            if self.ort_session is None:
                raise RuntimeError("Face detection model not available")
                
            if self.voice_unlock():
                if self.face_verification():
                    self.add_message("Authentication successful.", "system")
                    self.speak("Proceeding to Vision System...")
                    self.launch_vision()
                else:
                    self.add_message("Face verification failed", "system")
                    self.ask_to_repeat()
            else:
                self.add_message("Voice authentication failed", "system")
                self.ask_to_repeat()
        except Exception as e:
            self.add_message(f"Authentication failed: {str(e)}", "system")
            self.speak("Authentication failed. Please try again.")
            self.ask_to_repeat()

def main():
    root = tk.Tk()
    try:
        app = VisionIRLApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Fatal Error", f"Application failed to start: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()