#!/usr/bin/env bash
# install_vggt.sh
# Installs VGGT (Visual Geometry Grounded Transformer) from Facebook Research.
# Must be run BEFORE training Stage 2 with extractor_type="vggt".
#
# Usage: bash stage2/install_vggt.sh

set -e
echo "=== Installing VGGT ==="

# Clone into a sibling directory of stage2/
cd "$(dirname "$0")/.."
VGGT_DIR="vggt"

if [ -d "$VGGT_DIR" ]; then
    echo "VGGT directory already exists at $VGGT_DIR, pulling latest..."
    cd "$VGGT_DIR" && git pull && cd ..
else
    git clone git@github.com:facebookresearch/vggt.git "$VGGT_DIR"
fi

# Install VGGT requirements
cd "$VGGT_DIR"
pip install -r requirements.txt --break-system-packages

# Install VGGT itself as an editable package so imports work
pip install -e . --break-system-packages

cd ..
echo ""
echo "=== Verifying VGGT install ==="
python -c "
from vggt.models.vggt import VGGT
print('VGGT import successful.')
print('VGGT class:', VGGT)
"

echo ""
echo "=== VGGT installed successfully ==="
echo "Model weights will be downloaded automatically on first use:"
echo "  VGGT.from_pretrained('facebook/VGGT-1B')"
echo ""
echo "To pre-download weights now (recommended before renting GPU):"
echo "  python -c \"from vggt.models.vggt import VGGT; VGGT.from_pretrained('facebook/VGGT-1B')\""