# Dùng: . .\env.ps1  (PowerShell)
# Giữ MỌI cache / file tạm / cấu hình công cụ bên trong thư mục dự án — không ghi ra ổ C.
$C = Join-Path $PSScriptRoot '.cache'
$env:PIP_CACHE_DIR = "$C\pip"
$env:UV_CACHE_DIR = "$C\uv"
$env:YOLO_AUTOINSTALL = 'false'  # không để Ultralytics tự cài gói
$env:npm_config_cache = "$C\npm"
$env:TMP = "$C\tmp"; $env:TEMP = "$C\tmp"
$env:YOLO_CONFIG_DIR = "$C\ultralytics"
$env:TORCH_HOME = "$C\torch"
$env:HF_HOME = "$C\hf"
$env:MPLCONFIGDIR = "$C\matplotlib"
$env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
foreach ($d in 'uv','pip','npm','tmp','ultralytics','torch','hf','matplotlib') {
  New-Item -ItemType Directory -Force -Path (Join-Path $C $d) | Out-Null
}
