import face_recognition
import cv2
import numpy as np
import sys

def safe_face_recognition():
    """
    Safer version of face recognition with error handling and debugging.
    """
    print("🔧 Starting AttendEase Face Recognition System...")
    
    try:
        # Step 1: Load and verify the reference image
        print("📸 Loading reference image...")
        christian_image = face_recognition.load_image_file("photos/christian_esguerra.jpg")
        print("✓ Image loaded successfully")
        
        # Step 2: Extract face encoding
        print("🔍 Analyzing face in reference image...")
        face_encodings = face_recognition.face_encodings(christian_image)
        
        if len(face_encodings) == 0:
            print("❌ Error: No faces found in the reference image!")
            print("   Make sure the image has a clear, visible face.")
            return
        
        christian_face_encoding = face_encodings[0]
        print(f"✓ Face encoding extracted successfully ({len(face_encodings)} face(s) found)")
        
        # Step 3: Set up known faces
        known_face_encodings = [christian_face_encoding]
        known_face_names = ["Christian Esguerra"]
        print(f"✓ Known faces setup complete: {known_face_names}")
        
        # Step 4: Test webcam access
        print("📹 Initializing webcam...")
        video_capture = cv2.VideoCapture(0)
        
        if not video_capture.isOpened():
            print("❌ Error: Could not access webcam!")
            print("   Make sure no other applications are using the camera.")
            return
        
        # Test if we can read a frame
        ret, test_frame = video_capture.read()
        if not ret:
            print("❌ Error: Could not read from webcam!")
            video_capture.release()
            return
        
        print(f"✓ Webcam initialized successfully (Frame size: {test_frame.shape})")
        
        # Step 5: Start face recognition
        print("🚀 Starting real-time face recognition...")
        print("   - Press 'q' to quit")
        print("   - Press 'Esc' to quit")
        print("   - Close the window to quit")
        
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True
        frame_count = 0
        
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            
            if not ret:
                print("❌ Error: Lost connection to webcam!")
                break
            
            frame_count += 1
            
            # Only process every other frame of video to save time
            if process_this_frame:
                try:
                    # Resize frame of video to 1/4 size for faster face recognition processing
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    
                    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    
                    # Find all the faces and face encodings in the current frame of video
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                    
                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                        name = "Unknown"
                        
                        # Use the known face with the smallest distance to the new face
                        if known_face_encodings:
                            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                            best_match_index = np.argmin(face_distances)
                            if matches[best_match_index]:
                                name = known_face_names[best_match_index]
                                confidence = 1 - face_distances[best_match_index]
                                name += f" ({confidence:.2f})"
                        
                        face_names.append(name)
                    
                    # Debug output every 30 frames
                    if frame_count % 30 == 0:
                        print(f"Frame {frame_count}: Found {len(face_locations)} face(s)")
                        
                except Exception as e:
                    print(f"⚠️ Warning: Error processing frame {frame_count}: {e}")
                    face_locations = []
                    face_names = []
            
            process_this_frame = not process_this_frame
            
            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
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
            
            # Display the resulting image
            cv2.imshow('AttendEase - Face Recognition (Press Q to quit)', frame)
            
            # Check for quit commands
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:  # 'q', 'Q', or Esc
                print("👋 Quitting...")
                break
                
            # Check if window was closed
            if cv2.getWindowProperty('AttendEase - Face Recognition (Press Q to quit)', cv2.WND_PROP_VISIBLE) < 1:
                print("👋 Window closed, quitting...")
                break
        
        # Cleanup
        video_capture.release()
        cv2.destroyAllWindows()
        print("✅ Face recognition system shut down successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find the reference image: {e}")
        print("   Make sure 'photos/christian_esguerra.jpg' exists in the project directory.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("   Please check the error details above.")
    finally:
        # Make sure everything is cleaned up
        try:
            cv2.destroyAllWindows()
        except:
            pass

if __name__ == "__main__":
    safe_face_recognition() 