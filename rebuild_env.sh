#!/bin/bash
# ============================================================
# rebuild_env.sh
# Replaces the old Python-3.9 / PyTorch-1.12 environment with
# one that actually uses your Blackwell GPU (sm_120).
#
# Why bump Python 3.9 → 3.10?
#   PyTorch 2.7 cu128 (the first stable release shipping sm_120)
#   does not publish wheels for 3.9.  3.10 is the lowest version
#   that has full wheel coverage across torch / torchvision /
#   torchaudio / hydra / wandb — nothing in the BNT repo actually
#   uses any 3.9-only feature.
#
# Why PyTorch 2.7.0 specifically?
#   • 2.7.0 cu128 is the FIRST stable release confirmed to include
#     sm_120 kernels (see pytorch/pytorch #159207).
#   • Pinning to stable avoids the nightly-build flakiness that
#     people keep hitting on the forums.
# ============================================================

set -e

# ── 1. Drop the old environment entirely ──
echo "[1/5] Removing old 'bnt' environment …"
conda deactivate 2>/dev/null || true
conda env remove -n bnt -y 2>/dev/null || true

# ── 2. Create fresh env with Python 3.10 ──
echo "[2/5] Creating new 'bnt' env (Python 3.10) …"
conda create -n bnt python=3.10 -y

# ── 3. Activate ──
echo "[3/5] Activating …"
# Works in both bash and zsh; conda shell hook needed for activate
eval "$(conda shell.bash hook)"
conda activate bnt

# ── 4. Install PyTorch 2.7.0 + CUDA 12.8  (sm_120 included) ──
echo "[4/5] Installing PyTorch 2.7.0 cu128 …"
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# ── 5. Install everything else the BNT repo needs ──
echo "[5/5] Installing remaining BNT dependencies …"
pip install hydra-core==1.2.0 wandb scikit-learn pandas

# ── 6. Smoke test ──
echo ""
echo "============================================================"
echo " Smoke test"
echo "============================================================"
python -c "
import torch
print('PyTorch version :', torch.__version__)
print('CUDA available  :', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU             :', torch.cuda.get_device_name(0))
    print('Arch list       :', torch.cuda.get_arch_list())
    # tiny kernel launch — proves sm_120 kernels load
    x = torch.randn(4, 4, device='cuda')
    print('Kernel launch   : OK  (shape', x.shape, ')')
else:
    print('WARNING: CUDA not available — check driver / toolkit.')
"
echo "============================================================"
echo " Done.  Run:  conda activate bnt"
echo "============================================================"