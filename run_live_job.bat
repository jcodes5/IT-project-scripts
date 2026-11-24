@echo off
rem ============================================================
rem Daily live job for the Lokoja flood project.
rem ============================================================

rem EDIT 1: the folder that holds run_live_job.py and the data folder
set PROJECT_DIR=C:\Users\jatto\Research_project

rem EDIT 2: Earth Engine Cloud project ID
set EE_PROJECT_ID=nigeriagovhub-ef38e

rem EDIT 3: the public\web_data folder inside your web app (check this path is right)
set APP_PUBLIC=C:\Users\jatto\lokoja-flood-app\public\web_data

call C:\Users\jatto\miniconda3\condabin\conda.bat activate flood2
set PATH=%CONDA_PREFIX%;%CONDA_PREFIX%\Library\bin;%CONDA_PREFIX%\Scripts;C:\WINDOWS\system32;C:\WINDOWS

cd /d "%PROJECT_DIR%"
if not exist logs mkdir logs

echo ===== Started %DATE% %TIME% ===== >> logs\scheduler.txt
python run_live_job.py --project %EE_PROJECT_ID% --app-public "%APP_PUBLIC%" >> logs\scheduler.txt 2>&1
echo ===== Finished %DATE% %TIME% (exit code %ERRORLEVEL%) ===== >> logs\scheduler.txt