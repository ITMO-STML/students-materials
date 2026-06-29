@echo off
chcp 65001 >nul
echo ============================================
echo   NIR - Run Sanity Check
echo ============================================

:: Переменные окружения
set HF_HOME=W:\huggingface_cache

:: Проверяем что Ollama отвечает
echo Проверяем Ollama...
curl -s http://localhost:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo  *** ОШИБКА: Ollama не запущен! ***
    echo  Сначала запусти start_env.bat в другом терминале.
    echo.
    pause
    exit /b 1
)
echo OK - Ollama доступен

:: Запуск
echo.
echo Запускаем sanity check...
python W:\Jupyter\NIR_2\scripts\00_sanity_qwen.py %*

echo.
echo Готово! Результаты в W:\Jupyter\NIR_2\outputs\sanity\
pause
