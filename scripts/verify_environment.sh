#!/usr/bin/env bash
set -Eeuo pipefail
nvidia-smi
for e in h3_vanilla h3_fastv h3_sparsevlm h3_visionzip; do conda run -n "$e" python -c 'import torch; print(torch.__version__,torch.version.cuda,torch.cuda.is_available()); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0),torch.cuda.get_device_capability(0))'; done
conda run -n h3_fastv python -c 'from transformers import LlavaForConditionalGeneration; print("FastV Transformers OK")'
conda run -n h3_visionzip python -c 'import visionzip; print("VisionZip OK")'
