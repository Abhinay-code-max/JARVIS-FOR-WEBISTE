@echo off
setlocal EnableExtensions EnableDelayedExpansion

title JARVIS-XL System Launcher
chcp 65001 >nul

cd /d "%~dp0"

echo ===============================================================================
echo                   J.A.R.V.I.S - XL SYSTEM INITIALIZATION
echo ===============================================================================
echo.

REM ----------------------------------------------------------------------------
REM 1. Check Python Environment
REM ----------------------------------------------------------------------------
echo [*] [1/4] Checking Python environment...

set "PYTHON_EXE="

if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
    echo     Using virtual environment: .venv
) else if exist "%~dp0venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0venv\Scripts\python.exe"
    echo     Using virtual environment: venv
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        set "PYTHON_EXE=python"
    ) else (
        where py >nul 2>&1
        if !errorlevel! equ 0 (
            set "PYTHON_EXE=py"
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [!] ERROR: Python was not found in PATH or in a local virtual environment.
    echo     Please install Python 3.11+ and add it to your PATH: https://www.python.org/
    echo.
    pause
    exit /b 1
)

for /f "delims=" %%v in ('"%PYTHON_EXE%" --version 2^>^&1') do set "PY_VER=%%v"
echo     Python detected: %PY_VER%

REM ----------------------------------------------------------------------------
REM 2. Check and Start Ollama LLM Service
REM ----------------------------------------------------------------------------
echo.
echo [*] [2/4] Checking Ollama LLM service...

set "OLLAMA_EXE="
where ollama >nul 2>&1
if %errorlevel% equ 0 (
    set "OLLAMA_EXE=ollama"
) else if exist "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" (
    set "OLLAMA_EXE=%LOCALAPPDATA%\Programs\Ollama\ollama.exe"
    set "PATH=%LOCALAPPDATA%\Programs\Ollama;!PATH!"
) else if exist "%ProgramFiles%\Ollama\ollama.exe" (
    set "OLLAMA_EXE=%ProgramFiles%\Ollama\ollama.exe"
    set "PATH=%ProgramFiles%\Ollama;!PATH!"
)

REM Quick health check with curl
curl -s -f http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel% equ 0 (
    echo     Ollama service is active and responding on http://127.0.0.1:11434.
) else (
    if not "%OLLAMA_EXE%"=="" (
        echo     Ollama not running. Starting 'ollama serve' in background...
        start "" /min "%OLLAMA_EXE%" serve
        timeout /t 3 /nobreak >nul
        curl -s -f http://127.0.0.1:11434/api/tags >nul 2>&1
        if !errorlevel! equ 0 (
            echo     Ollama service started successfully.
        ) else (
            echo     Ollama service starting up in background...
        )
    ) else (
        echo [!] WARNING: Ollama executable not found. Make sure your LLM server is running.
    )
)

REM ----------------------------------------------------------------------------
REM 3. Display Configured & Available Models
REM ----------------------------------------------------------------------------
echo.
echo [*] [3/4] Checking Available LLM Models...
if not "%OLLAMA_EXE%"=="" (
    "%OLLAMA_EXE%" list
) else (
    echo     (External / Remote LLM endpoint configured)
)

REM ----------------------------------------------------------------------------
REM 4. Launch JARVIS-XL System (UI, Whisper STT, TTS, Wake-Word, Actions)
REM ----------------------------------------------------------------------------
echo.
echo ===============================================================================
echo [*] [4/4] Launching JARVIS-XL System...
echo           * PyQt6 HUD Interface
echo           * STT Speech Recognition (Whisper / faster-whisper)
echo           * TTS Speech Synthesis (EdgeTTS / Kokoro)
echo           * Wake-Word Detection ("Hey JARVIS")
echo           * Face & Voice Identification Systems
echo           * Autonomous Tool Execution Engine
echo ===============================================================================
echo.

"%PYTHON_EXE%" main.py %*

set "EXIT_CODE=%errorlevel%"
if %EXIT_CODE% neq 0 (
    echo.
    echo ===============================================================================
    echo [!] JARVIS-XL stopped with exit code %EXIT_CODE%.
    echo ===============================================================================
    pause
) else (
    echo.
    echo [*] JARVIS-XL session ended.
)
