# AttendEase

A Smart Attendance Checker with Facial Recognition for classrooms

Version 0.1.0
- This is to test the first ideas and to have a demo up and running

Project by Group: Vector Four for SNTSDEV

## Installation Instructions (Windows 10)

### Prerequisites
Make sure you have Python 3.11 installed. You can check your Python version with:
```bash
python --version
```

### Installing CMake and dlib

#### Step 1: Install CMake
CMake is required to build dlib from source on Windows.

**Option A: Using Windows Package Manager (Recommended)**
```bash
winget install Kitware.CMake
```

**Option B: Manual Installation**
1. Download CMake from [cmake.org](https://cmake.org/download/)
2. Run the installer and make sure to check "Add CMake to the system PATH"

#### Step 2: Refresh Environment Variables
After installing CMake, you may need to refresh your PowerShell session or run:
```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
```

#### Step 3: Verify CMake Installation
```bash
cmake --version
```
You should see something like: `cmake version 4.0.2`

#### Step 4: Install dlib
With CMake installed, you can now install dlib:
```bash
python -m pip install dlib
```

#### Step 5: Verify dlib Installation
```bash
python -c "import dlib; print('dlib imported successfully!')"
```

### Python Environment Notes
If you have multiple Python installations, make sure to use `python -m pip` instead of just `pip` to ensure you're installing packages to the correct Python version.

To check which Python and pip you're using:
```bash
python --version
python -m pip --version
```

#### Step 6: Install Additional Dependencies
Install the remaining packages required for facial recognition:
```bash
python -m pip install opencv-python face-recognition numpy
```

This will install:
- **opencv-python**: For webcam access and image processing
- **face-recognition**: High-level facial recognition library (includes additional models)
- **numpy**: For numerical operations
- **Pillow**: Image processing library (installed automatically with face-recognition)
- **Click**: Command-line interface utilities (installed automatically with face-recognition)

#### Step 7: Verify Complete Installation
Test that all packages are working:
```bash
python -c "import cv2, face_recognition, numpy as np; print('All packages imported successfully!')"
```

### Quick Install (All Dependencies)
If you have CMake already installed, you can install all dependencies at once using the requirements file:
```bash
python -m pip install -r requirements.txt
```

## Testing the System

### Basic Test
After installation, you can test if everything is working with:
```bash
python -c "import cv2, face_recognition, numpy as np; print('✓ All packages imported successfully!')"
```

### Running the Demo
1. The project includes a sample face recognition script (`attendease-v-0-1-0.py`)
2. To test with your own images, add `.jpg`, `.jpeg`, or `.png` files to the project directory
3. The system will automatically detect and learn faces from these reference images
4. Run the demo to test real-time face recognition with your webcam

### Project Structure
```
AttendEase/
├── README.md
├── requirements.txt
├── attendease-v-0-1-0.py    # Sample face recognition demo
└── [your-reference-images]   # Add .jpg/.png files here
```

## Troubleshooting
- If CMake is not recognized after installation, restart your terminal or refresh environment variables
- If you get compilation errors, ensure you have the latest version of pip: `python -m pip install --upgrade pip`
- On some systems, you may need Visual Studio Build Tools for C++ compilation
- If webcam access fails, ensure no other applications are using the camera
- For face recognition issues, ensure reference images have clear, well-lit faces
