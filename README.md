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

### Additional Dependencies
For a complete facial recognition setup, you may also want to install:
```bash
python -m pip install opencv-python face-recognition numpy
```

### Troubleshooting
- If CMake is not recognized after installation, restart your terminal or refresh environment variables
- If you get compilation errors, ensure you have the latest version of pip: `python -m pip install --upgrade pip`
- On some systems, you may need Visual Studio Build Tools for C++ compilation
