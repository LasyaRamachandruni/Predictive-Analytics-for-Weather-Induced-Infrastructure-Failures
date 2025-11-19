# PowerShell script to permanently add Python to PATH
# Run this script as Administrator or it will add to User PATH

Write-Host "Adding Python to PATH permanently..." -ForegroundColor Green

$pythonPath = "C:\Users\Asus\AppData\Local\Programs\Python\Python312"
$pythonScripts = "C:\Users\Asus\AppData\Local\Programs\Python\Python312\Scripts"

# Get current User PATH
$currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

# Check if already in PATH
if ($currentPath -notlike "*$pythonPath*") {
    # Add Python paths
    $newPath = $currentPath + ";$pythonPath;$pythonScripts"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    Write-Host "✅ Python paths added to User PATH!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Paths added:" -ForegroundColor Yellow
    Write-Host "  - $pythonPath" -ForegroundColor Cyan
    Write-Host "  - $pythonScripts" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⚠️  IMPORTANT: Close and reopen your terminal for changes to take effect!" -ForegroundColor Yellow
} else {
    Write-Host "✅ Python is already in PATH!" -ForegroundColor Green
}

Write-Host ""
Write-Host "To verify, run: python --version" -ForegroundColor Cyan

