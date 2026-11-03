@echo off
rem Fresh trial of the web desk, entirely in Docker (deploy/compose.yaml): PostgreSQL, Ollama and the model,
rem the MCP server, the API, the web UI and the observability stack. Needs only Docker Desktop running.
rem The reset deletes every transaction in the demo store and ends every session.
rem   start_demo.bat          uses the NVIDIA GPU when Docker has the NVIDIA runtime, else the CPU
rem   start_demo.bat --cpu    never uses the GPU
rem The first run needs the internet and takes a while: it builds the images (about 13 GB) and pulls the
rem model (about 4.6 GB). Later runs work offline. Stop everything with:
rem   docker compose -f deploy/compose.yaml stop
setlocal
cd /d "%~dp0"
set "COMPOSE=docker compose -f deploy\compose.yaml"

docker info >nul 2>&1
if errorlevel 1 (
    echo Docker is not running. Start Docker Desktop, then run this again.
    goto :failed
)

rem The ports, as deploy/.env sets them (the defaults otherwise).
set "UI_PORT=5173"
set "POSTGRES_PORT=55432"
set "GRAFANA_PORT=13000"
set "TEMPO_PORT=13200"
set "PROMETHEUS_PORT=19090"
if exist "deploy\.env" for /f "usebackq eol=# tokens=1,* delims==" %%a in ("deploy\.env") do (
    for %%k in (UI_PORT POSTGRES_PORT GRAFANA_PORT TEMPO_PORT PROMETHEUS_PORT) do if /i "%%a"=="%%k" set "%%k=%%b"
)

rem The GPU overlay only when Docker can give containers the GPU.
set "GPU=CPU only"
if /i not "%~1"=="--cpu" (
    docker info --format "{{json .Runtimes}}" | findstr /C:"nvidia" >nul
    if not errorlevel 1 (
        set "COMPOSE=%COMPOSE% -f deploy\compose.gpu.yaml"
        set "GPU=NVIDIA GPU"
    )
)

rem If this stack isn't up yet, its ports must be free: the native stack (docker-compose.yml,
rem observability/compose.yaml, npm run dev) uses the same ones.
set "running="
for /f %%i in ('%COMPOSE% ps -q net 2^>nul') do set "running=1"
set "busy="
if not defined running for %%p in (%UI_PORT% %POSTGRES_PORT% %GRAFANA_PORT% %TEMPO_PORT% %PROMETHEUS_PORT%) do (
    call :is_listening %%p && (echo Port %%p is already in use. & set "busy=1")
)
if defined busy (
    echo Stop what uses it first. For the native stack: docker compose stop postgres
    echo and docker compose -f observability/compose.yaml stop, and close the native UI and API.
    echo Or choose other ports in deploy\.env ^(see deploy\.env.example^).
    goto :failed
)

echo [1/4] Building the images (%GPU%)...
%COMPOSE% build
if errorlevel 1 (
    echo The build failed. The first build needs the internet.
    goto :failed
)

echo [2/4] Starting PostgreSQL and the model server, with the API and UI stopped...
rem The README's order: no API may hold sessions while the store is reset.
%COMPOSE% stop ui api >nul 2>&1
%COMPOSE% up -d postgres ollama-pull
if errorlevel 1 (
    echo PostgreSQL or the model server did not start. See: %COMPOSE% logs postgres ollama ollama-pull
    goto :failed
)

echo [3/4] Ending old sessions and resetting the store to the seed data...
rem On a new database the session registry doesn't exist yet, and cleanup fails without it.
%COMPOSE% exec -T postgres psql -U retail -d retail -tAc "SELECT to_regclass('retail_ui_registry.sessions')" | findstr /C:"sessions" >nul
if not errorlevel 1 (
    %COMPOSE% run --rm tools python -m app.api.cleanup --older-than-hours 0
    if errorlevel 1 (
        echo Session cleanup failed, so the store was not reset.
        goto :failed
    )
)
%COMPOSE% run --rm tools python -m app.db.demo_store --reset
if errorlevel 1 (
    echo The store reset failed.
    goto :failed
)

echo [4/4] Starting the MCP server, the API, the UI and the observability stack...
rem Returns once the API is healthy and the UI has started (the model download, if any, comes first).
%COMPOSE% up -d
if errorlevel 1 (
    echo The stack did not start. See: %COMPOSE% ps   and   %COMPOSE% logs api
    goto :failed
)

echo Waiting for the desk to load its models...
set /a waited=0
:wait_ready
curl -s "http://127.0.0.1:%UI_PORT%/health/ready" 2>nul | findstr /C:"\"state\":\"ready\"" >nul
if not errorlevel 1 goto :ready
set /a waited+=1
if %waited% geq 150 (
    echo The desk is not ready after 5 minutes. See: %COMPOSE% logs api
    goto :failed
)
ping -n 3 127.0.0.1 >nul
goto :wait_ready

:ready
echo.
echo Ready (%GPU%).
echo UI:      http://127.0.0.1:%UI_PORT%   (sign in as C100 or C200)
echo API:     http://127.0.0.1:%UI_PORT%/docs
echo Grafana: http://127.0.0.1:%GRAFANA_PORT%
echo Stop:    %COMPOSE% stop
start "" "http://127.0.0.1:%UI_PORT%"
exit /b 0

:failed
echo.
pause
exit /b 1

rem Sets errorlevel 0 if the TCP port has a listener.
:is_listening
netstat -ano -p TCP | findstr /R /C:":%~1 .*LISTENING" >nul
exit /b %errorlevel%
