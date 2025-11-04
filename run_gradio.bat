cd /d "%~dp0"
call conda activate ai
cls
call python __main_gradio__.py
pause