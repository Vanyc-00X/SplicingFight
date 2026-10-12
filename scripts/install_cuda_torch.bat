@echo off
REM Install CUDA PyTorch into project venv on D: (C: often too full)
cd /d "%~dp0\.."
set TMP=%CD%\.tmp_pip
set TEMP=%CD%\.tmp_pip
mkdir "%TMP%" 2>nul
if not exist ".venv\Scripts\python.exe" python -m venv .venv
".venv\Scripts\python.exe" -m pip install -U pip
".venv\Scripts\pip.exe" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
".venv\Scripts\pip.exe" install -r requirements.txt
".venv\Scripts\python.exe" -c "import torch; print('cuda', torch.cuda.is_available(), torch.__version__)"
pause
