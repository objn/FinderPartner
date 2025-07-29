@echo off
echo 🚀 FinderPartner CLIP Training Pipeline Demo
echo ============================================================

echo.
echo 📋 Step 1: Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    echo Please run: python -m venv venv
    echo Then run: venv\Scripts\activate
    echo Then run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo 📋 Step 2: Checking environment...
python scripts\check_env.py
if errorlevel 1 (
    echo ❌ Environment check failed
    echo Please install requirements: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo 🎯 Step 3: Starting CLIP training...
echo Press Ctrl+C to stop training early if needed
echo.
python src\train_clip.py --config configs\clip_base.yaml

echo.
echo ✅ Training completed or stopped
pause
