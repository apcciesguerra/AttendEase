import face_recognition
import cv2
import numpy as np
import sys
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

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

# Processing flag from original code
process_this_frame = True 

# Reference image path (ensure this path is correct)
REFERENCE_IMAGE_PATH = "photos/christian_esguerra.jpg"

# --- Core Functions ---

def load_reference_data():
    """Loads the reference image and extracts face encodings."""
    global known_face_encodings, known_face_names
    print("📸 Loading reference image...")
    try:
        # Ensure the 'photos' directory and image exist or adjust the path.
        # For example, if running from the same directory as the script,
        # and 'photos' is a subdirectory: "photos/christian_esguerra.jpg"
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
        known_face_names = ["Christian Esguerra"] # You can make this more dynamic if needed
        
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

def check_webcam_status():
    """Checks initial webcam connectivity and updates status_label."""
    global status_label
    print("📹 Checking webcam status...")
    cap_test = cv2.VideoCapture(0) # Try to open the default camera
    if cap_test.isOpened():
        status_label.config(text="Webcam: Connected and Ready", fg="green")
        print("✓ Webcam connected and ready.")
        cap_test.release()
        return True
    else:
        status_label.config(text="Webcam: Not Detected / Error", fg="red")
        print("❌ Error: Could not access webcam for initial check.")
        return False

def update_gui_frame(frame_to_display):
    """Converts an OpenCV frame to a Tkinter PhotoImage and updates the video_label."""
    global video_label
    try:
        # Convert frame from BGR (OpenCV default) to RGB
        cv2image = cv2.cvtColor(frame_to_display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        
        video_label.imgtk = imgtk  # Keep a reference to avoid garbage collection
        video_label.configure(image=imgtk, text="") # Clear any placeholder text
    except Exception as e:
        print(f"Error updating GUI frame: {e}")


def recognize_and_display_video():
    """Captures a frame, performs face recognition, and schedules the next update."""
    global video_capture, camera_on, process_this_frame, video_label
    global known_face_encodings, known_face_names

    if not camera_on or video_capture is None or not video_capture.isOpened():
        # This check ensures we don't proceed if camera was turned off or failed
        return

    ret, frame = video_capture.read()
    if not ret:
        print("❌ Error: Lost connection to webcam or cannot read frame.")
        # Attempt to turn off camera gracefully through the toggle function
        # to update UI state correctly.
        if camera_on: # only toggle if it was supposed to be on
            toggle_camera() 
        status_label.config(text="Webcam: Error reading frame", fg="red")
        return

    # Process the frame for face recognition
    # Make a copy for processing if you modify it before drawing final boxes
    # frame_for_processing = frame.copy() 
    
    # Local lists for detected faces in the current frame
    face_locations_detected = []
    face_names_detected = []

    # Original logic: Only process every other frame of video to save time
    if process_this_frame:
        try:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (OpenCV uses) to RGB color (face_recognition uses)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find all the faces and face encodings in the current frame of video
            face_locations_detected = face_recognition.face_locations(rgb_small_frame)
            current_face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations_detected)
            
            for face_encoding in current_face_encodings:
                name = "Unknown"
                if known_face_encodings: # Check if there are known faces to compare against
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    
                    if len(face_distances) > 0: # Ensure distances were calculated
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]
                            # Optional: Add confidence display
                            # confidence = 1 - face_distances[best_match_index]
                            # name += f" ({confidence:.2f})"
                face_names_detected.append(name)
        except Exception as e:
            print(f"⚠️ Warning: Error during face recognition in frame: {e}")
            # Reset detected faces for this frame if error occurs
            face_locations_detected = []
            face_names_detected = []
            
    process_this_frame = not process_this_frame

    # Display the results (drawing on the original `frame`)
    for (top, right, bottom, left), name in zip(face_locations_detected, face_names_detected):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
    
    update_gui_frame(frame) # Update the video feed in the UI
    
    # Schedule the next frame processing if the camera is still supposed to be on
    if camera_on:
        video_label.after(15, recognize_and_display_video) # Adjust delay for desired FPS (15ms ~66fps)

def toggle_camera():
    """Turns the webcam ON or OFF and updates the UI accordingly."""
    global camera_on, video_capture, toggle_button, status_label, video_label
    
    target_camera_state = not camera_on # Desired state after toggle
    
    if target_camera_state: # Try to turn ON
        print("📹 Turning camera ON...")
        # Attempt to initialize the webcam (0 is usually the default)
        video_capture = cv2.VideoCapture(0) 
        if video_capture.isOpened():
            camera_on = True # Successfully turned on
            toggle_button.config(text="Turn Off Camera")
            status_label.config(text="Webcam: Active", fg="blue")
            print("✓ Webcam activated.")
            # Clear placeholder and start video stream
            video_label.config(image='', text="") 
            recognize_and_display_video() # Start the video processing loop
        else:
            print("❌ Error: Could not access webcam to turn ON.")
            messagebox.showerror("Webcam Error", "Could not access webcam. Make sure it's not in use by another application.")
            # Ensure camera_on remains False, video_capture is None
            camera_on = False 
            if video_capture: video_capture.release()
            video_capture = None
            toggle_button.config(text="Turn On Camera") # Revert button text
            status_label.config(text="Webcam: Not Detected/Error", fg="red") # Update status
    else: # Turn OFF
        print("📹 Turning camera OFF...")
        camera_on = False # Set state to off
        if video_capture:
            video_capture.release()
            video_capture = None
        toggle_button.config(text="Turn On Camera")
        status_label.config(text="Webcam: Off", fg="black")
        
        # Display a placeholder image when camera is off
        placeholder_img = Image.new('RGB', (640, 480), color='lightgray') # Default size
        # Or use video_label current size if available and non-zero
        # width = video_label.winfo_width()
        # height = video_label.winfo_height()
        # if width > 1 and height > 1:
        #     placeholder_img = Image.new('RGB', (width, height), color='lightgray')
            
        imgtk = ImageTk.PhotoImage(image=placeholder_img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk, text="Camera Off", compound=tk.CENTER, fg="black", bg="lightgray")
        print("✓ Webcam deactivated.")

def on_closing_application():
    """Handles application cleanup when the window is closed."""
    global root, camera_on, video_capture
    print("👋 Closing application...")
    if camera_on: # If camera is on, turn it off gracefully
        if video_capture:
            video_capture.release()
        camera_on = False # Ensure video loop stops
    
    if root:
        root.destroy() # Close the Tkinter window
    print("✅ Application shut down successfully!")
    # cv2.destroyAllWindows() # Generally not needed if Tkinter manages windows and capture is released.

# --- UI Setup ---
def create_main_ui():
    """Creates and configures the main Tkinter UI."""
    global root, status_label, toggle_button, video_label

    print("🔧 Initializing AttendEase UI...")
    
    # Load reference face data first. If it fails, the app might be non-functional.
    if not load_reference_data():
        print("❌ Critical error: Could not load reference data. Face recognition will not work.")
        # Decide if UI should still launch or exit. For now, it will launch with error shown.
        # Alternatively, could add: return
    
    root = tk.Tk()
    root.title("AttendEase - Face Recognition v0.1.0")
    root.geometry("720x650") # Adjusted size for better layout

    # --- Header/Title Label (Optional) ---
    title_label = tk.Label(root, text="AttendEase System", font=("Helvetica", 16, "bold"))
    title_label.pack(pady=(10,0))

    # --- Status Label for Webcam ---
    status_label = tk.Label(root, text="Webcam: Initializing...", font=("Helvetica", 12))
    status_label.pack(pady=(5,10))

    # --- Toggle Camera Button ---
    toggle_button = tk.Button(root, text="Turn On Camera", command=toggle_camera, 
                              font=("Helvetica", 12), width=20, height=2, 
                              bg="#4CAF50", fg="white", activebackground="#45a049")
    toggle_button.pack(pady=10)

    # --- Video Display Label ---
    video_frame = tk.Frame(root, bg="black", bd=2, relief=tk.SUNKEN) # Frame to hold video
    video_frame.pack(pady=10, padx=10, expand=True, fill=tk.BOTH)
    
    video_label = tk.Label(video_frame, bg="lightgray") # Video frames will be shown here
    # Set an initial placeholder text/image
    placeholder_img = Image.new('RGB', (640, 480), color='lightgray')
    imgtk = ImageTk.PhotoImage(image=placeholder_img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk, text="Camera Off", compound=tk.CENTER, fg="black")
    video_label.pack(expand=True, fill=tk.BOTH)
    
    # Perform initial webcam check and update status label
    check_webcam_status()

    # Handle window close event
    root.protocol("WM_DELETE_WINDOW", on_closing_application)
    
    print("🚀 UI Ready. Press 'Turn On Camera' to start recognition.")
    root.mainloop()

# --- Main Execution ---
if __name__ == "__main__":
    # The old safe_face_recognition() function is replaced by create_main_ui()
    create_main_ui() 