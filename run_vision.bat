@echo off
chcp 65001 >nul
echo ========================================
echo AirSim无人机视觉处理系统 - 快速启动
echo ========================================
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查依赖是否安装
echo [检查] 检查依赖包...
pip show airsim >nul 2>&1
if %errorlevel% neq 0 (
    echo [警告] 未检测到airsim包，正在安装依赖...
    pip install -r requirements.txt
)

REM 检查AirSim是否运行
echo [检查] 检查AirSim连接...
timeout /t 2 >nul

REM 创建必要目录
if not exist "logs" mkdir logs
if not exist "sensor_data" mkdir sensor_data
if not exist "visualizations" mkdir visualizations

echo.
echo ========================================
echo [启动] 启动视觉处理系统...
echo ========================================
echo.
echo 键盘控制说明:
echo   e - 捕获并处理两帧图像
echo   w - 前进
echo   s - 后退
echo   a - 左移
echo   d - 右移
echo   , - 上升
echo   . - 下降
echo   q - 退出程序
echo.
echo ========================================
echo.

python src/main.py

echo.
echo ========================================
echo [退出] 程序已退出
echo ========================================
pause