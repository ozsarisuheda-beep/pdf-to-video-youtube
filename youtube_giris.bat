@echo off
chcp 65001 >nul
echo ====================================================
echo  YouTube Oturumu Acma (1 Kerelik Yetkilendirme)
echo ====================================================
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python auth_youtube.py
