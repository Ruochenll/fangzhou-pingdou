@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo === 推送到 GitHub (Ruochenll/fangzhou-pingdou) ===
echo.
git push -u origin main

if errorlevel 1 (
  echo.
  echo [推送失败]
  echo.
  echo 可能原因：
  echo   1. 首次推送需要登录 GitHub —— 会弹出浏览器/Git凭据管理窗口，
  echo      完成授权后重新双击本脚本即可（凭据会被记住，下次不再弹窗）
  echo   2. 若提示 non-fast-forward —— 说明远程仓库已有内容（如勾选了
  echo      README），需要先同步：运行下面两行再重试：
  echo         git pull --rebase origin main
  echo         git push -u origin main
  echo.
  pause
  exit /b 1
)

echo.
echo === 推送成功 ===
echo 仓库地址：https://github.com/Ruochenll/fangzhou-pingdou
echo.
pause
