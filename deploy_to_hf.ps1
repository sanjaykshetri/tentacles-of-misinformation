# Deploy Streamlit app to Hugging Face Spaces
# Usage: .\deploy_to_hf.ps1 -HFUsername "your_hf_username" -SpaceName "misinformation-classifier"

param(
    [string]$HFUsername = "",
    [string]$SpaceName = "misinformation-classifier",
    [string]$WorkDir = "$env:USERPROFILE\OneDrive\Documents"
)

# Colors for output
$green = "Green"
$yellow = "Yellow"
$red = "Red"

function Write-Step {
    param([string]$Message)
    Write-Host "→ $Message" -ForegroundColor $green
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ $Message" -ForegroundColor $yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor $red
}

# Check prerequisites
Write-Host "`n🚀 Hugging Face Spaces Deployment Script`n" -ForegroundColor Cyan

if (-not $HFUsername) {
    Write-Error-Custom "HatF username is required"
    Write-Info "Usage: .\deploy_to_hf.ps1 -HFUsername 'your_username'"
    exit 1
}

Write-Step "Checking prerequisites..."

# Check git
try {
    git --version | Out-Null
    Write-Step "Git installed ✓"
} catch {
    Write-Error-Custom "Git not found. Please install Git first."
    exit 1
}

# Check if we're in the repo
$repoRoot = "c:\Users\sanja\OneDrive\Documents\GitHub\tentacles-of-misinformation"
if (-not (Test-Path "$repoRoot\.git")) {
    Write-Error-Custom "Not in main repo directory. Expected: $repoRoot"
    exit 1
}

Write-Step "Main repo found ✓"

# Create HF Space clone directory
$spaceDir = "$WorkDir\$SpaceName"
$hfUrl = "https://huggingface.co/spaces/$HFUsername/$SpaceName"

if (Test-Path $spaceDir) {
    Write-Info "Space directory already exists: $spaceDir"
    $response = Read-Host "Overwrite? (y/n)"
    if ($response -ne "y") {
        Write-Info "Deployment cancelled"
        exit 0
    }
    Remove-Item $spaceDir -Recurse -Force
}

Write-Step "Creating space directory..."
New-Item -ItemType Directory -Path $spaceDir -Force | Out-Null

# Copy files
Write-Step "Copying application files..."
Copy-Item "$repoRoot\dashboards\streamlit\app.py" "$spaceDir\app.py" -Force
Copy-Item "$repoRoot\dashboards\streamlit\requirements.txt" "$spaceDir\requirements.txt" -Force
Copy-Item "$repoRoot\dashboards\streamlit\.streamlit" "$spaceDir\.streamlit" -Recurse -Force

Write-Step "Copying model files..."
New-Item -ItemType Directory -Path "$spaceDir\models" -Force | Out-Null
Copy-Item "$repoRoot\models\*.joblib" "$spaceDir\models\" -Force

Write-Step "Updating app.py path for HF Space..."
$appPath = "$spaceDir\app.py"
$appContent = Get-Content $appPath -Raw
$appContent = $appContent -replace 'Path\(__file__\)\.parent\.parent\.parent / "models"', 'Path(__file__).parent / "models"'
Set-Content $appPath $appContent -NoNewline

# Git initialization
Write-Step "Initializing git repository..."
Push-Location $spaceDir

git init | Out-Null
git remote add huggingface $hfUrl | Out-Null
git add . | Out-Null

# First commit
Write-Step "Creating initial commit..."
git commit -m "Add Streamlit misinformation classifier app" | Out-Null

# Push to HF
Write-Step "Pushing to Hugging Face Spaces..."
try {
    git push -u huggingface main 2>&1 | ForEach-Object { Write-Info $_ }
    Write-Host "`n✓ Deployment successful!`n" -ForegroundColor Green
} catch {
    # Branch might be 'main' or 'master' - try master
    Write-Info "Trying push to master branch..."
    try {
        git branch -M master
        git push -u huggingface master 2>&1 | ForEach-Object { Write-Info $_ }
        Write-Host "`n✓ Deployment successful!`n" -ForegroundColor Green
    } catch {
        Write-Error-Custom "Push failed: $_"
        Pop-Location
        exit 1
    }
}

Pop-Location

# Summary
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "✅ Your app is live!" -ForegroundColor Green
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host ""
Write-Host "Space URL:" -ForegroundColor Yellow
Write-Host "👉 $hfUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Visit the URL above to see your live app"
Write-Host "2. Test with different text samples"
Write-Host "3. Share the link in your portfolio/resume"
Write-Host ""
Write-Host "To update your app:" -ForegroundColor Yellow
Write-Host "   cd $spaceDir"
Write-Host "   # Make changes to app.py"
Write-Host "   git add . && git commit -m 'Update' && git push huggingface main"
Write-Host ""
