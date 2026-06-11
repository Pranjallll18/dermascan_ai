$targetFile = "..\..\torch-2.6.0+cu124-cp311-cp311-win_amd64.whl"
$url = "https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp311-cp311-win_amd64.whl"

Write-Output "Starting PyTorch GPU wheel download with auto-resume..."

for ($i = 1; $i -le 30; $i++) {
    Write-Output "Attempt $i of 30..."
    curl.exe -L -C - $url -o $targetFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Download completed successfully!"
        exit 0
    }
    
    Write-Warning "Download interrupted. Retrying in 5 seconds..."
    Start-Sleep -Seconds 5
}

Write-Error "Failed to download PyTorch after 30 attempts."
exit 1
