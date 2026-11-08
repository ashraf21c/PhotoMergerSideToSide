@echo off
setlocal
cd /d "%~dp0"

py -c "import streamlit, streamlit_cropper, PIL, numpy, cv2" >nul 2>nul
if errorlevel 1 (
    echo Required packages are not installed.
    echo Run: py -m pip install -r "%~dp0requirements.txt"
    pause
    exit /b 1
)

py -m streamlit run "%~dp0BookStyleImageMerger.py"
exit /b %errorlevel%
