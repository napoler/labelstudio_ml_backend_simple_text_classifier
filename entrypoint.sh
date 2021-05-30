#!/bin/bash --login
# The --login ensures the bash configuration is loaded,
# enabling Conda.
set -euo pipefail

# 激活环境
conda activate  myenv

exec "$@"
# cd /
# exec label-studio-ml start /app