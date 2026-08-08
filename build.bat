@echo off
chcp 65001 > nul
setlocal
cd /d "%~dp0"

REM ====== 拼豆像素画助手 - 一键打包脚本 ======
REM 前置：项目 .venv 已存在且依赖已安装（pip install -r requirements.txt）
REM 产物：dist\PingDou\PingDou.exe（双击即弹 UAC 提权后启动）
REM 分发：把整个 dist\PingDou 文件夹用 7z/zip 压缩发给别人，解压双击 exe 即可

if not exist ".venv\Scripts\python.exe" (
  echo [错误] 未找到 .venv，请先在项目根目录创建虚拟环境并安装依赖
  echo         ".venv\Scripts\python.exe" -m venv .venv
  echo         ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  exit /b 1
)

echo === 安装 PyInstaller（如尚未安装） ===
".venv\Scripts\python.exe" -m pip install --no-cache-dir --no-clean pyinstaller > nul
if errorlevel 1 (
  echo [错误] PyInstaller 安装失败
  exit /b 1
)

echo === 清理旧产物 ===
if exist build rmdir /s /q build 2>nul
if exist dist rmdir /s /q dist 2>nul
if exist PingDou.spec del /q PingDou.spec 2>nul

echo === 开始打包（onedir 模式，启动快） ===
".venv\Scripts\python.exe" -m PyInstaller ^
  --noconfirm ^
  --windowed ^
  --uac-admin ^
  --name PingDou ^
  --add-data "known_palette.json;." ^
  --collect-submodules keyboard ^
  --exclude-module tkinter ^
  --exclude-module unittest ^
  --exclude-module pytest ^
  --exclude-module matplotlib ^
  --exclude-module IPython ^
  --exclude-module notebook ^
  --exclude-module PyQt5 ^
  --exclude-module PyQt6 ^
  main.py

if errorlevel 1 (
  echo.
  echo [错误] 打包失败，请查看上方日志
  exit /b 1
)

echo.
echo ========================================
echo   打包成功
echo ========================================
echo   产物目录：  dist\PingDou\
echo   启动方式：  双击 dist\PingDou\PingDou.exe（自动弹 UAC 提权）
echo   分发方式：  把整个 dist\PingDou 文件夹用 7z/zip 压缩发给别人
echo               对方解压后双击 PingDou.exe 即可，无需安装 Python
echo.
echo   首次启动会在 exe 旁边生成：
echo     - config.json  （保存框选区域等配置）
echo     - debug\        （检测叠加图等调试信息）
echo.
echo   如需单文件 exe（启动稍慢但分发更简单）：
echo     把上面 PyInstaller 命令加 --onefile 参数重新打包
echo.
pause
