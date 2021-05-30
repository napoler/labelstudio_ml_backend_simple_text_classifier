#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.
set -euo pipefail

# 激活环境
conda activate  label-studio-ml-backend



exec label-studio-ml start /app