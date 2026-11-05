@echo off
title LianLianKan Bot Launcher
chcp 65001 >nul

echo ============================================================
echo           LianLianKan Visual Solver Suite (連連看解題器)
echo ============================================================
echo.
echo [1] 啟動 Studio 建模工具 (框選盤面 ROI 並提取範本特徵)
echo [2] 啟動 Engine 執行引擎 (讀取 config 直接進行即時解題)
echo [3] 完整流程：先執行 Studio，完成後自動啟動 Engine
echo.
set /p choice="請選擇模式 [1/2/3] (預設 3): "

if "%choice%"=="1" (
    studio.exe
    goto end
)
if "%choice%"=="2" (
    if not exist "game_profile.json" (
        echo [錯誤] 找不到 game_profile.json！請先執行模式 [1] 建立設定。
        pause
        goto end
    )
    engine.exe --profile game_profile.json --templates templates
    goto end
)

:: 預設 3
echo 正在啟動 Studio 進行盤面建模...
studio.exe
if %ERRORLEVEL% equ 0 (
    echo.
    echo 建模完成！正在啟動 Engine 執行引擎...
    engine.exe --profile game_profile.json --templates templates
) else (
    echo 建模未完成或已被使用者取消。
)

:end
pause
