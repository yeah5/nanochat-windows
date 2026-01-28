# powershell
<#
PowerShell port of `speedrun.sh`.
Save as `speedrun.ps1` and run from PowerShell (preferably with developer tools installed).
This script tries to use native Windows installers when possible and falls back to Bash/WSL for Unix-only install scripts.
#>

function CommandExists($name) {
    return (Get-Command $name -ErrorAction SilentlyContinue) -ne $null
}

# -----------------------------------------------------------------------------
# Environment and base dir
$env:OMP_NUM_THREADS = "1"
$env:NANOCHAT_BASE_DIR = Join-Path $HOME ".cache\nanochat"
New-Item -ItemType Directory -Path $env:NANOCHAT_BASE_DIR -Force | Out-Null

# -----------------------------------------------------------------------------
# Python venv setup with uv
if (-not (CommandExists "uv")) {
    if (CommandExists "bash") {
        Write-Host "Installing uv via upstream install script using bash..."
        bash -lc "curl -LsSf https://astral.sh/uv/install.sh | sh"
        # ensure new uv is on PATH (may require a new shell)
    } else {
        Write-Warning "Command 'uv' not found and no bash available to run the installer. Install 'uv' manually: https://astral.sh/uv"
    }
}

if (-not (Test-Path ".venv")) {
    if (CommandExists "uv") {
        Write-Host "Creating .venv via uv..."
        & uv venv
    } else {
        Write-Warning "Skipping virtualenv creation because 'uv' is not available."
    }
}

if (CommandExists "uv") {
    Write-Host "Syncing dependencies (extra: gpu)..."
    & uv sync --extra gpu
}

# Activate venv: PowerShell activation path vs Unix
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-Host "Activating PowerShell venv..."
    . .\.venv\Scripts\Activate.ps1
} elseif (Test-Path ".venv/bin/activate" -PathType Leaf -ErrorAction SilentlyContinue -ErrorVariable ev) {
    if (CommandExists "bash") {
        Write-Host "Activating Unix venv via bash..."
        bash -lc "source .venv/bin/activate"
    } else {
        Write-Warning "Found Unix-style venv but no bash to source it."
    }
} else {
    Write-Warning "Virtual environment not found. Continuing with system Python."
}

# -----------------------------------------------------------------------------
# wandb setup
if (-not $env:WANDB_RUN -or $env:WANDB_RUN -eq "") {
    $env:WANDB_RUN = "dummy"
}

# -----------------------------------------------------------------------------
# Write report header
Write-Host "Resetting report..."
& python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer: Install Rust / Cargo
if (-not (CommandExists "rustc")) {
    if (CommandExists "bash") {
        Write-Host "Installing Rust via rustup (bash)..."
        bash -lc "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y"
    } else {
        Write-Host "Attempting to download rustup-init.exe for Windows..."
        $exeUrl = "https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe"
        $tmp = Join-Path $env:TEMP "rustup-init.exe"
        Invoke-WebRequest -Uri $exeUrl -OutFile $tmp -UseBasicParsing
        Start-Process -FilePath $tmp -ArgumentList "-y" -Wait
    }
}

# Ensure cargo bin is on PATH
$cargoBin = Join-Path $HOME ".cargo\bin"
if (Test-Path $cargoBin) {
    $env:PATH = "$cargoBin;$env:PATH"
}

# Build the rustbpe Tokenizer using maturin
if (CommandExists "uv") {
    Write-Host "Building rustbpe with maturin..."
    & uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
} else {
    Write-Warning "Skipping maturin build because 'uv' is not available. Ensure maturin is installed and run: maturin develop --release --manifest-path rustbpe/Cargo.toml"
}


# -----------------------------------------------------------------------------
# Dataset download & tokenizer training
Write-Host "Downloading initial dataset shards (8)..."
& python -m nanochat.dataset -n 8

Write-Host "Kicking off background dataset download (240 shards)..."
# Start background process and capture PID
$proc = Start-Process -FilePath "python" -ArgumentList "-m nanochat.dataset -n 240" -NoNewWindow -PassThru
$datasetPid = $proc.Id

Write-Host "Training tokenizer (vocab size 2^16) on ~2B chars..."
& python -m scripts.tok_train --max_chars=2000000000

Write-Host "Evaluating tokenizer..."
& python -m scripts.tok_eval

#-----------------------------------------------------------------------------
# Re-Installing torch and verifying CUDA
# This was necessary for the Windows port due to some uv sync issues

# use the venv python explicitly to avoid picking up a global "uv" Python
$venvPython = ".\.venv\Scripts\python.exe"

# check for pip; if missing, bootstrap it
& $venvPython -m pip --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip not found; bootstrapping with ensurepip..."
    & $venvPython -m ensurepip --upgrade
}

# upgrade pip and install packages
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128


# test torch and CUDA availability
& $venvPython -c "import torch; print('torch:', getattr(torch, '__version__', 'unknown')); print('cuda available:', torch.cuda.is_available())"

# -----------------------------------------------------------------------------
# Modify some torch distributed scripts for Windows compatibility
Write-Host "Patching torch distributed scripts for Windows compatibility..."
#Copying some modified scripts back over so the library syncs don't wipe out custom settings
Set-Location $PSScriptRoot
$env:PYTHONPATH = $PSScriptRoot
$Folder1 = $PSScriptRoot + "\.venv\Lib\site-packages\torch\distributed"
$Folder2 = $PSScriptRoot + "\.venv\Lib\site-packages\torch\distributed\elastic\rendezvous"
Write-Host $Folder1
$Item1 = $PSScriptRoot + "\rendezvous.py"
$Item2 = $PSScriptRoot + "\c10d_rendezvous_backend.py"
$Item3 = $PSScriptRoot + "\dynamic_rendezvous.py"
Copy-Item $Item1 -Destination $Folder1
Copy-Item $Item2 -Destination $Folder2
Copy-Item $Item3 -Destination $Folder2

# -----------------------------------------------------------------------------
# Base model (pretraining)
Write-Host "Waiting for dataset download to complete..."
if ($datasetPid) {
    Wait-Process -Id $datasetPid -ErrorAction SilentlyContinue
}

# Number of processes/GPUs to use
$env:NPROC_PER_NODE = "1"

# Evaluate or not
$env:EVAL_OR_NOT = "false"
$env:MEASURE_LOSS = "true"

Write-Host ""
Write-Host "    !!"
Write-Host "Starting base model pretraining..."
Write-Host "    !!"
Write-Host ""
& torchrun --standalone --nproc_per_node=$env:NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$env:WANDB_RUN

Write-Host ""
Write-Host "    !!"
Write-Host "Evaluating base model loss..."
Write-Host "    !!"
Write-Host ""
if ($env:EVAL_OR_NOT -eq "false") {
    Write-Host "Skipping base model loss as per EVAL_OR_NOT=$env:EVAL_OR_NOT"
    } else {
    & torchrun --standalone --nproc_per_node=$env:NPROC_PER_NODE -m scripts.base_loss
}

Write-Host ""
Write-Host "    !!"
Write-Host "Running base model core evals..."
Write-Host "    !!"
Write-Host ""
if ($env:EVAL_OR_NOT -eq "false") {
    Write-Host "Skipping base model core evals as per EVAL_OR_NOT=$env:EVAL_OR_NOT"
    } else {
    & torchrun --standalone --nproc_per_node=$env:NPROC_PER_NODE -m scripts.base_eval
}

# -----------------------------------------------------------------------------
# Midtraining
Write-Host ""
Write-Host "    !!"
Write-Host "Downloading synthetic identity conversations..."
Write-Host "    !!"
Write-Host ""
$identityPath = Join-Path $env:NANOCHAT_BASE_DIR "identity_conversations.jsonl"
Invoke-WebRequest -Uri "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl" -OutFile $identityPath -UseBasicParsing


Write-Host ""
Write-Host "    !!"
Write-Host "Running midtraining..."
Write-Host "    !!"
Write-Host ""
& torchrun --standalone --nproc_per_node=$env:NPROC_PER_NODE -m scripts.mid_train -- --run=$env:WANDB_RUN
if ($env:EVAL_OR_NOT -eq "false") {
    Write-Host "Skipping mid training core evals as per EVAL_OR_NOT=$env:EVAL_OR_NOT"
    } else {
    & torchrun --standalone --nproc_per_node=$env:NPROC_PER_NODE -m scripts.chat_eval -- -i mid
}

# -----------------------------------------------------------------------------
# Supervised Finetuning (SFT)
Write-Host ""
Write-Host "    !!"
Write-Host "Running supervised finetuning (SFT)..."
Write-Host "    !!"
Write-Host ""
& torchrun --standalone --nproc_per_node=$env:NPROC_PER_NODE -m scripts.chat_sft -- --run=$env:WANDB_RUN
if ($env:EVAL_OR_NOT -eq "false") {
    Write-Host "Skipping SFT evals as per EVAL_OR_NOT=$env:EVAL_OR_NOT"
    } else {
    & torchrun --standalone --nproc_per_node=$env:NPROC_PER_NODE -m scripts.chat_eval -- -i sft
}

# -----------------------------------------------------------------------------
# Optional Reinforcement Learning (commented out in original)
# & torchrun --standalone --nproc_per_node=$env:NPROC_PER_NODE -m scripts.chat_rl -- --run=$env:WANDB_RUN
# & torchrun --standalone --nproc_per_node=$env:NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# Generate final report
Write-Host ""
Write-Host "    !!"
Write-Host "Generating final report..."
Write-Host "    !!"
Write-Host ""
& python -m nanochat.report generate
Write-Host ""
Write-Host ""
Write-Host ""
Write-Host "All Done!"
 