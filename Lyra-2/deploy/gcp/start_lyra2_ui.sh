#!/usr/bin/env bash
set -euo pipefail

INSTALL_ROOT="${INSTALL_ROOT:-/opt/lyra}"
REPO_URL="${REPO_URL:-$(curl -fsS -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo-url' 2>/dev/null || true)}"
REPO_URL="${REPO_URL:-https://github.com/nv-tlabs/lyra.git}"
REPO_BRANCH="${REPO_BRANCH:-$(curl -fsS -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo-branch' 2>/dev/null || true)}"
REPO_BRANCH="${REPO_BRANCH:-main}"
HF_TOKEN="${HF_TOKEN:-$(curl -fsS -H 'Metadata-Flavor: Google' 'http://metadata.google.internal/computeMetadata/v1/instance/attributes/hf-token' 2>/dev/null || true)}"
CONDA_ENV="${CONDA_ENV:-lyra2}"
PORT="${PORT:-8080}"
ROOT_DIR="${ROOT_DIR:-${INSTALL_ROOT}/Lyra-2}"
SERVICE_SCRIPT="/usr/local/bin/lyra2-ui-service"

export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends git curl ca-certificates build-essential

mkdir -p "$INSTALL_ROOT"
if [[ ! -d "${INSTALL_ROOT}/.git" ]]; then
  git clone --recursive --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_ROOT"
else
  git -C "$INSTALL_ROOT" fetch origin "$REPO_BRANCH"
  git -C "$INSTALL_ROOT" checkout "$REPO_BRANCH"
  git -C "$INSTALL_ROOT" pull origin "$REPO_BRANCH"
  git -C "$INSTALL_ROOT" submodule update --init --recursive
fi

if ! command -v conda >/dev/null 2>&1; then
  curl -fsSLo /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p /opt/conda
  ln -sf /opt/conda/bin/conda /usr/local/bin/conda
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -n "$CONDA_ENV" python=3.10 pip cmake ninja libgl ffmpeg packaging -c conda-forge -y
fi
conda activate "$CONDA_ENV"
CONDA_BACKUP_CXX="" conda install gcc=13.3.0 gxx=13.3.0 eigen zlib -c conda-forge -y
conda install cuda -c nvidia/label/cuda-12.8.0 -y

export CUDA_HOME="$CONDA_PREFIX"
SITE="$CONDA_PREFIX/lib/python3.10/site-packages"
export CPATH="$CUDA_HOME/include:$SITE/nvidia/cudnn/include:$SITE/nvidia/nccl/include:${CPATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$SITE/torch/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cudnn/lib:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

cd "$ROOT_DIR"
pip install torch==2.7.1 torchvision==0.22.1 --extra-index-url https://download.pytorch.org/whl/cu128
pip install --no-deps -r requirements.txt
pip install "git+https://github.com/microsoft/MoGe.git"
pip install --no-build-isolation "transformer_engine[pytorch]"
ln -sf "$SITE/nvidia/cuda_runtime" "$SITE/nvidia/cudart"
MAX_JOBS="${MAX_JOBS:-16}" pip install --no-build-isolation --no-binary :all: flash-attn==2.6.3
USE_SYSTEM_EIGEN=1 pip install --no-build-isolation -e 'lyra_2/_src/inference/vipe'
pip install --no-build-isolation -e 'lyra_2/_src/inference/depth_anything_3[gs]'

if [[ ! -f checkpoints/text_encoder/negative_prompt.pt ]]; then
  if [[ -n "$HF_TOKEN" ]]; then
    huggingface-cli login --token "$HF_TOKEN"
  fi
  huggingface-cli download nvidia/Lyra-2.0 --include "checkpoints/*" --local-dir .
fi

cat > "$SERVICE_SCRIPT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
cd "$ROOT_DIR"
export CUDA_HOME="\$CONDA_PREFIX"
SITE="\$CONDA_PREFIX/lib/python3.10/site-packages"
export CPATH="\$CUDA_HOME/include:\$SITE/nvidia/cudnn/include:\$SITE/nvidia/nccl/include:\${CPATH:-}"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$SITE/torch/lib:\$SITE/nvidia/cuda_runtime/lib:\$SITE/nvidia/cudnn/lib:\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="$ROOT_DIR\${PYTHONPATH:+:\$PYTHONPATH}"
export LYRA2_ROOT="$ROOT_DIR"
export LYRA2_HOST="0.0.0.0"
export LYRA2_PORT="$PORT"
exec python ui/web_ui.py
EOF
chmod +x "$SERVICE_SCRIPT"

cat > /etc/systemd/system/lyra2-ui.service <<EOF
[Unit]
Description=Lyra 2 playground UI
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=$SERVICE_SCRIPT
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now lyra2-ui
