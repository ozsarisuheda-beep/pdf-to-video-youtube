@echo off
chcp 65001 >nul
echo ====================================================
echo  PDF + Ses -^> Video Uretici Baslatiliyor...
echo ====================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
streamlit run app.py
pause
