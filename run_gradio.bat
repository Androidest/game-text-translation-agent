cd /d "%~dp0"
call conda activate ai
cls
call python main_gradio.py
pause