:: Change the current directory to where the script is located
@echo off
cd /d "%~dp0"  

:: Delete all .png files in the current directory
del *.png /q

echo All .png files have been deleted.