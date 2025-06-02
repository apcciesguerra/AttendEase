import face_recognition
import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import time
from collections import defaultdict
import threading

# --- Global Variables ---
video_capture = None
camera_on = False
known_face_encodings = []
known_face_names = []

# UI Elements
root = None
status_label = None
toggle_button = None
video_label = None
faces_detected_label = None
recognized_names_display_label = None

# Enhanced tracking variables
face_tracker = {}  # Dictionary to store face tracking data
next_face_id = 0
TRACKING_THRESHOLD = 0.6  # Face recognition confidence threshold
TRACKING_FRAMES = 100  # Number of frames to keep tracking without detection (approx 5s at 30fps)
FACE_DISTANCE_THRESHOLD = 100  # Maximum pixel distance for face tracking

# Processing optimization
process_this_frame = True 
frame_count = 0
last_detection_time = time.time()

# Reference image path
REFERENCE_IMAGE_PATH = "photos/christian_esguerra.jpg"

# --- Color Palette ---
APC_GOLD = "#D1A134"
APC_BLUE = "#002B5C"
APC_WHITE = "#FFFFFF"
APC_LIGHT_GOLD = "#E6BE5C" # For hover/active states
APC_DARK_RED_ERROR = "#A00000" # For error messages
APC_STATUS_GREEN = "#006400" # Dark green for positive status 

# OpenCV BGR Colors
CV_APC_GOLD = (0x34, 0xA1, 0xD1) # BGR for #D1A134
CV_APC_BLUE = (0x5C, 0x2B, 0x00) # BGR for #002B5C
CV_APC_WHITE = (255, 255, 255)
CV_APC_GREEN_CONFIRMED = (0, 100, 0) # Dark Green for confirmed
CV_APC_YELLOW_TENTATIVE = (0, 191, 255) # A bright yellow
CV_APC_RED_UNKNOWN = (0,0,139) # Dark Red for unknown

# --- Enhanced Face Tracking Class ---
class FaceTracker:
    def __init__(self, face_id, name, location, encoding=None):
        self.id = face_id
        self.name = name
        self.location = location  # (top, right, bottom, left)
        self.encoding = encoding
        self.last_seen = time.time()
        self.confidence_history = []
        self.missed_frames = 0
        self.is_confirmed = False  # True if we're confident this is the right person
        
    def update_location(self, new_location, confidence=None):
        self.location = new_location
        self.last_seen = time.time()
        self.missed_frames = 0
        if confidence is not None:
            self.confidence_history.append(confidence)
            if len(self.confidence_history) > 5:
                self.confidence_history.pop(0)
            # Mark as confirmed if we have consistent good matches
            if len(self.confidence_history) >= 3 and all(c > 0.4 for c in self.confidence_history):
                self.is_confirmed = True
    
    def increment_missed_frames(self):
        self.missed_frames += 1
        
    def is_expired(self):
        return self.missed_frames > TRACKING_FRAMES
    
    def get_center(self):
        top, right, bottom, left = self.location
        return ((left + right) // 2, (top + bottom) // 2)

# --- Core Functions ---

def load_reference_data():
    """Loads the reference image and extracts face encodings."""
    global known_face_encodings, known_face_names
    print("📸 Loading reference image...")
    try:
        christian_image = face_recognition.load_image_file(REFERENCE_IMAGE_PATH)
        print("✓ Reference image loaded successfully.")
        
        face_encodings_list = face_recognition.face_encodings(christian_image)
        
        if not face_encodings_list:
            error_msg = f"❌ Error: No faces found in the reference image: {REFERENCE_IMAGE_PATH}\n   Please ensure it contains a clear, visible face."
            print(error_msg)
            messagebox.showerror("Reference Image Error", error_msg)
            return False
        
        christian_face_encoding = face_encodings_list[0]
        known_face_encodings = [christian_face_encoding]
        known_face_names = ["Christian Esguerra"]
        
        print(f"✓ Face encoding extracted. {len(known_face_encodings)} known face(s) configured: {known_face_names}")
        return True
        
    except FileNotFoundError:
        error_msg = f"❌ Error: Reference image not found at '{REFERENCE_IMAGE_PATH}'.\n   Please check the file path."
        print(error_msg)
        messagebox.showerror("File Not Found", error_msg)
        return False
    except Exception as e:
        error_msg = f"❌ Unexpected error loading reference data: {e}"
        print(error_msg)
        messagebox.showerror("Load Error", error_msg)
        return False

def calculate_distance(loc1, loc2):
    """Calculate Euclidean distance between two face locations."""
    center1 = ((loc1[1] + loc1[3]) // 2, (loc1[0] + loc1[2]) // 2)
    center2 = ((loc2[1] + loc2[3]) // 2, (loc2[0] + loc2[2]) // 2)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def match_faces_to_trackers(face_locations, face_encodings):
    """Match detected faces to existing trackers or create new ones."""
    global face_tracker, next_face_id, known_face_encodings, known_face_names
    
    # Scale face locations back to full size
    scaled_locations = []
    for (top, right, bottom, left) in face_locations:
        scaled_locations.append((top * 4, right * 4, bottom * 4, left * 4))
    
    # First, try to match with existing trackers based on proximity
    matched_trackers = set()
    new_detections = []
    
    for i, location in enumerate(scaled_locations):
        best_tracker = None
        min_distance = float('inf')
        
        # Find closest existing tracker
        for tracker_id, tracker in face_tracker.items():
            if tracker_id in matched_trackers:
                continue
            distance = calculate_distance(location, tracker.location)
            if distance < FACE_DISTANCE_THRESHOLD and distance < min_distance:
                min_distance = distance
                best_tracker = tracker_id
        
        if best_tracker is not None:
            # Update existing tracker
            name = "Unknown"
            confidence = None
            
            # Perform face recognition for known faces
            if i < len(face_encodings) and known_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encodings[i], tolerance=TRACKING_THRESHOLD)
                face_distances = face_recognition.face_distance(known_face_encodings, face_encodings[i])
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
            
            # Update tracker
            face_tracker[best_tracker].update_location(location, confidence)
            face_tracker[best_tracker].name = name
            matched_trackers.add(best_tracker)
        else:
            # New face detection
            new_detections.append((location, face_encodings[i] if i < len(face_encodings) else None))
    
    # Create new trackers for unmatched detections
    for location, encoding in new_detections:
        name = "Unknown"
        confidence = None
        
        # Perform face recognition for new faces
        if encoding is not None and known_face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=TRACKING_THRESHOLD)
            face_distances = face_recognition.face_distance(known_face_encodings, encoding)
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
        
        # Create new tracker
        tracker = FaceTracker(next_face_id, name, location, encoding)
        if confidence is not None:
            tracker.update_location(location, confidence)
        face_tracker[next_face_id] = tracker
        next_face_id += 1
    
    # Increment missed frames for unmatched trackers
    for tracker_id in list(face_tracker.keys()):
        if tracker_id not in matched_trackers:
            face_tracker[tracker_id].increment_missed_frames()
            # Remove expired trackers
            if face_tracker[tracker_id].is_expired():
                del face_tracker[tracker_id]

def check_webcam_status():
    """Checks initial webcam connectivity and updates status_label."""
    global status_label
    print("📹 Checking webcam status...")
    cap_test = cv2.VideoCapture(0)
    if cap_test.isOpened():
        status_label.config(text="Webcam: Connected and Ready", fg=APC_STATUS_GREEN)
        print("✓ Webcam connected and ready.")
        cap_test.release()
        return True
    else:
        status_label.config(text="Webcam: Not Detected / Error", fg=APC_DARK_RED_ERROR)
        print("❌ Error: Could not access webcam for initial check.")
        return False

def update_gui_frame(frame_to_display):
    """Converts an OpenCV frame to a Tkinter PhotoImage and updates the video_label."""
    global video_label
    try:
        cv2image = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk, text="")
    except Exception as e:
        print(f"Error updating GUI frame: {e}")

def recognize_and_display_video():
    """Enhanced face recognition with tracking."""
    global video_capture, camera_on, process_this_frame, video_label
    global known_face_encodings, known_face_names, frame_count, face_tracker, faces_detected_label
    global recognized_names_display_label

    if not camera_on or video_capture is None or not video_capture.isOpened():
        return

    ret, frame = video_capture.read()
    if not ret:
        print("❌ Error: Lost connection to webcam or cannot read frame.")
        if camera_on:
            toggle_camera()
        status_label.config(text="Webcam: Error reading frame", fg=APC_DARK_RED_ERROR)
        return

    frame_count += 1
    
    # Process face detection every 3rd frame for better performance
    if frame_count % 3 == 0:
        try:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            current_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            # Update trackers
            match_faces_to_trackers(face_locations, current_face_encodings)
            
        except Exception as e:
            print(f"⚠️ Warning: Error during face recognition: {e}")
    else:
        # On non-processing frames, just increment missed frames for existing trackers
        for tracker in face_tracker.values():
            tracker.increment_missed_frames()
    
    # Draw all active trackers
    for tracker_id, tracker in list(face_tracker.items()):
        if tracker.is_expired():
            continue
            
        top, right, bottom, left = tracker.location
        
        # Choose color based on recognition status
        if tracker.name != "Unknown" and tracker.is_confirmed:
            color = CV_APC_GREEN_CONFIRMED
            thickness = 3
        elif tracker.name != "Unknown":
            color = CV_APC_YELLOW_TENTATIVE
            thickness = 2
        else:
            color = CV_APC_RED_UNKNOWN
            thickness = 2
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, thickness)
        
        # Prepare label text
        label = tracker.name
        if tracker.confidence_history:
            avg_confidence = np.mean(tracker.confidence_history)
            label += f" ({avg_confidence:.2f})"
        
        # Draw label background
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        # Draw label text
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, CV_APC_WHITE, 1)
        
        # Add tracking ID for debugging
        cv2.putText(frame, f"ID:{tracker_id}", (left, top - 10), font, 0.4, color, 1)
    
    # Add frame info
    info_text = f"Frames: {frame_count} | Active Trackers: {len(face_tracker)}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, CV_APC_WHITE, 2, cv2.LINE_AA)
    
    update_gui_frame(frame)
    
    # Update faces detected label
    if faces_detected_label:
        num_active_trackers = len(face_tracker)
        faces_detected_label.config(text=f"Faces Detected: {num_active_trackers}")

    # Update recognized names display label
    if recognized_names_display_label:
        current_recognized_names = sorted(list(set(
            tracker.name for tracker_id, tracker in face_tracker.items()
            if tracker.name != "Unknown"
        )))
        if current_recognized_names:
            recognized_names_display_label.config(text=", ".join(current_recognized_names))
        else:
            recognized_names_display_label.config(text="None")

    if camera_on:
        video_label.after(33, recognize_and_display_video)  # ~30 FPS

def toggle_camera():
    """Turns the webcam ON or OFF and updates the UI accordingly."""
    global camera_on, video_capture, toggle_button, status_label, video_label, face_tracker, faces_detected_label
    global recognized_names_display_label
    
    target_camera_state = not camera_on
    
    if target_camera_state:
        print("📹 Turning camera ON...")
        video_capture = cv2.VideoCapture(0)
        if video_capture.isOpened():
            # Set camera properties for better performance
            video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            video_capture.set(cv2.CAP_PROP_FPS, 30)
            
            camera_on = True
            toggle_button.config(text="Turn Off Camera", bg=APC_LIGHT_GOLD, fg=APC_BLUE)
            status_label.config(text="Webcam: Active", fg=APC_BLUE)
            print("✓ Webcam activated.")
            
            # Clear tracking data
            face_tracker.clear()
            
            video_label.config(image='', text="")
            if faces_detected_label:
                faces_detected_label.config(text="Faces Detected: 0")
            if recognized_names_display_label:
                recognized_names_display_label.config(text="None")
            recognize_and_display_video()
        else:
            print("❌ Error: Could not access webcam to turn ON.")
            messagebox.showerror("Webcam Error", "Could not access webcam. Make sure it's not in use by another application.")
            camera_on = False
            if video_capture:
                video_capture.release()
            video_capture = None
            toggle_button.config(text="Turn On Camera", bg=APC_GOLD, fg=APC_BLUE)
            status_label.config(text="Webcam: Not Detected/Error", fg=APC_DARK_RED_ERROR)
    else:
        print("📹 Turning camera OFF...")
        camera_on = False
        if video_capture:
            video_capture.release()
            video_capture = None
        
        # Clear tracking data
        face_tracker.clear()
        
        toggle_button.config(text="Turn On Camera", bg=APC_GOLD, fg=APC_BLUE)
        status_label.config(text="Webcam: Off", fg=APC_BLUE)
        if faces_detected_label:
            faces_detected_label.config(text="Faces Detected: 0")
        if recognized_names_display_label:
            recognized_names_display_label.config(text="None")
        
        placeholder_img = Image.new('RGB', (640, 480), color=APC_GOLD)
        imgtk = ImageTk.PhotoImage(image=placeholder_img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk, text="Camera Off", compound=tk.CENTER, fg=APC_BLUE, bg=APC_GOLD)
        print("✓ Webcam deactivated.")

def on_closing_application():
    """Handles application cleanup when the window is closed."""
    global root, camera_on, video_capture
    print("👋 Closing application...")
    if camera_on:
        if video_capture:
            video_capture.release()
        camera_on = False
    
    if root:
        root.destroy()
    print("✅ Application shut down successfully!")

# --- UI Setup ---
def create_main_ui():
    """Creates and configures the main Tkinter UI."""
    global root, status_label, toggle_button, video_label, faces_detected_label
    global recognized_names_display_label

    print("🔧 Initializing Enhanced AttendEase UI...")
    
    if not load_reference_data():
        print("❌ Critical error: Could not load reference data. Face recognition will not work.")
    
    root = tk.Tk()
    root.title("AttendEase v0.1.0")
    root.geometry("800x700")
    root.configure(bg=APC_WHITE)

    # Header
    title_label = tk.Label(root, text="AttendEase (DEMO)", font=("Helvetica", 16, "bold"), bg=APC_WHITE, fg=APC_BLUE)
    title_label.pack(pady=(10,0))

    # Main content frame for horizontal layout
    main_horizontal_frame = tk.Frame(root, bg=APC_WHITE)
    main_horizontal_frame.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)

    # Left side: Video feed
    video_container_frame = tk.Frame(main_horizontal_frame, bg=APC_WHITE)
    video_container_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 10))

    video_frame = tk.Frame(video_container_frame, bg=APC_BLUE, bd=2, relief=tk.SUNKEN)
    video_frame.pack(expand=True, fill=tk.BOTH)
    
    video_label = tk.Label(video_frame, bg=APC_GOLD)
    placeholder_img = Image.new('RGB', (640, 480), color=APC_GOLD)
    imgtk = ImageTk.PhotoImage(image=placeholder_img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk, text="Camera Off", compound=tk.CENTER, fg=APC_BLUE, bg=APC_GOLD)
    video_label.pack(expand=True, fill=tk.BOTH)

    # Right side: Controls
    controls_frame = tk.Frame(main_horizontal_frame, bg=APC_WHITE)
    controls_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

    toggle_button = tk.Button(controls_frame, text="Turn On Camera", command=toggle_camera,
                              font=("Helvetica", 12), width=20, height=2,
                              bg=APC_GOLD, fg=APC_BLUE, activebackground=APC_LIGHT_GOLD, activeforeground=APC_BLUE,
                              relief=tk.FLAT, borderwidth=0)
    toggle_button.pack(pady=(0,5))

    status_label = tk.Label(controls_frame, text="Webcam: Initializing...", 
                           font=("Helvetica", 10), bg=APC_WHITE, fg=APC_BLUE)
    status_label.pack(pady=(0,10))

    faces_detected_label = tk.Label(controls_frame, text="Faces Detected: 0", 
                                   font=("Helvetica", 12), bg=APC_WHITE, fg=APC_BLUE)
    faces_detected_label.pack(pady=5)
    
    students_label = tk.Label(controls_frame, text="Recognized Students:",
                              font=("Helvetica", 10, "bold"), bg=APC_WHITE, fg=APC_BLUE)
    students_label.pack(pady=(10,0))

    recognized_names_display_label = tk.Label(controls_frame, text="None",
                                             font=("Helvetica", 10), bg=APC_WHITE, fg=APC_BLUE, wraplength=180, justify=tk.LEFT)
    recognized_names_display_label.pack(pady=(0,10))

    info_label = tk.Label(controls_frame, text="by Vector Four", 
                         font=("Helvetica", 10), fg=APC_BLUE, bg=APC_WHITE)
    info_label.pack(pady=(20,0), side=tk.BOTTOM)
    
    check_webcam_status()
    root.protocol("WM_DELETE_WINDOW", on_closing_application)
    
    print("🚀 Enhanced UI Ready. Features: Face tracking, confidence scoring, stable recognition")
    root.mainloop()

# --- Main Execution ---
if __name__ == "__main__":
    create_main_ui() 