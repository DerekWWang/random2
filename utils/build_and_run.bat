@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
call conda activate deepl
nvcc conv1d_main.cu -o conv1d_test.exe
if errorlevel 1 (echo Build failed & exit /b 1)
conv1d_test.exe
