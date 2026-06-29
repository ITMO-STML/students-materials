@echo off
chcp 65001 >nul
echo ============================================
echo   NIR Environment Startup
echo ============================================
echo.

:: 1. Убиваем фоновый Ollama (он стартует с кириллическим путём)
echo [1/3] Останавливаем фоновый Ollama...
taskkill /F /IM "ollama app.exe" >nul 2>&1
taskkill /F /IM "ollama.exe" >nul 2>&1
taskkill /F /IM "ollama_llama_server.exe" >nul 2>&1
timeout /t 3 /nobreak >nul

:: 2. Устанавливаем путь к моделям БЕЗ кириллицы
set OLLAMA_MODELS=W:\ollama_home\models
set HF_HOME=W:\huggingface_cache

:: 3. Запускаем Ollama serve в этом же окне
echo [2/3] Запускаем Ollama с путём: %OLLAMA_MODELS%
echo [3/3] Сервер будет слушать на http://localhost:11434
echo.
echo  *** НЕ ЗАКРЫВАЙ ЭТО ОКНО ***
echo  *** Скрипты запускай во ВТОРОМ терминале ***
echo.

ollama serve
