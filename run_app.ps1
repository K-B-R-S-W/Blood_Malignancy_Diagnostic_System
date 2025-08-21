# PowerShell script to run Blood Cell AI with proper conda environment
Write-Host "🚀 Starting Blood Cell AI Diagnostic System..." -ForegroundColor Green
Write-Host "📁 Current directory: $(Get-Location)" -ForegroundColor Cyan

# Check if conda is available
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    Write-Host "❌ Conda not found in PATH" -ForegroundColor Red
    exit 1
}

# Activate MAIN environment
Write-Host "🔧 Activating MAIN environment..." -ForegroundColor Yellow
& conda activate MAIN

# Check if activation worked by running Python with full path
$pythonPath = "C:\Users\DEATHSEC\anaconda3\envs\MAIN\python.exe"
if (Test-Path $pythonPath) {
    Write-Host "✅ Found Python at: $pythonPath" -ForegroundColor Green
    
    # Test PyTorch availability
    Write-Host "🔍 Testing PyTorch..." -ForegroundColor Yellow
    & $pythonPath -c "import torch; print('✅ PyTorch version:', torch.__version__)"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "🤖 Starting Flask application..." -ForegroundColor Green
        & $pythonPath main.py
    } else {
        Write-Host "❌ PyTorch test failed" -ForegroundColor Red
    }
} else {
    Write-Host "❌ Python not found at expected path: $pythonPath" -ForegroundColor Red
    Write-Host "📋 Available Python environments:" -ForegroundColor Yellow
    & conda env list
}

Write-Host "Press any key to continue..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
