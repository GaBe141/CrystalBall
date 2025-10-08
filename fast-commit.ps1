# Fast Git Commit Script
# This script optimizes git operations for better performance

param(
    [Parameter(Mandatory=$true)]
    [string]$Message,
    
    [switch]$All,
    [switch]$Push
)

Write-Host "🚀 Fast Git Commit Starting..." -ForegroundColor Green

# Measure commit time
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

try {
    if ($All) {
        Write-Host "📋 Adding all changes..." -ForegroundColor Yellow
        git add .
    }
    
    Write-Host "💾 Committing with message: '$Message'" -ForegroundColor Yellow
    git commit -m $Message
    
    if ($Push) {
        Write-Host "🌐 Pushing to remote..." -ForegroundColor Yellow
        git push
    }
    
    $stopwatch.Stop()
    Write-Host "✅ Commit completed in $($stopwatch.ElapsedMilliseconds)ms" -ForegroundColor Green
    
} catch {
    $stopwatch.Stop()
    Write-Host "❌ Error during commit: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}