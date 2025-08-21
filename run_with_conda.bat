@echo off
echo 🚀 Activating MAIN environment and starting Blood Cell AI...
call conda activate MAIN
echo 🔬 Environment activated: %CONDA_DEFAULT_ENV%
echo 📁 Current directory: %CD%
echo 🔍 Checking Python and PyTorch...
python -c "import torch; print('✅ PyTorch available:', torch.__version__)"
echo 🤖 Starting Flask application...
python main.py
pause
