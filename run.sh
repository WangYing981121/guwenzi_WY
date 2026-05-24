#!/bin/bash
set -euo pipefail

echo "Starting YOLO + EfficientNet inference..."

echo "===== Runtime paths ====="
echo "PWD=$(pwd)"
echo "INPUT_DIR=${INPUT_DIR:-<default>}"
echo "OUTPUT_FILE=${OUTPUT_FILE:-<default>}"
for path in /saisdata /saisresult /app; do
  if [ -e "${path}" ]; then
    echo "${path}: present"
    ls -la "${path}" | head -n 40 || true
  else
    echo "${path}: missing"
  fi
done
echo "===== End runtime paths ====="

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${NVIDIA_VISIBLE_DEVICES:-}" ] \
  && [ "${NVIDIA_VISIBLE_DEVICES}" != "all" ] && [ "${NVIDIA_VISIBLE_DEVICES}" != "void" ]; then
  export CUDA_VISIBLE_DEVICES="${NVIDIA_VISIBLE_DEVICES}"
fi

mkdir -p /usr/local/cuda/lib64 || true
for lib in libcuda.so libnvidia-ml.so; do
  if [ ! -e "/usr/local/cuda/lib64/${lib}" ]; then
    target="$(ldconfig -p 2>/dev/null | awk -v name="${lib}.1" '$1 == name && $NF !~ "/usr/local/cuda/compat/" {print $NF; exit}' || true)"
    if [ -n "${target}" ]; then
      ln -sf "${target}" "/usr/local/cuda/lib64/${lib}" || true
    fi
  fi
done
ldconfig || true

echo "===== GPU diagnostics ====="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
echo "NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
ls -l /dev/nvidia* 2>/dev/null || echo "/dev/nvidia* not found"
ldconfig -p 2>/dev/null | grep -E 'libcuda|libnvidia-ml|libcudart|libcublas|libcudnn' || true

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi found: $(command -v nvidia-smi)"
  nvidia-smi || true
  nvidia-smi -L || true
else
  echo "nvidia-smi not found in container PATH"
fi

python3 - <<'PY' || true
import os
import ctypes
import ctypes.util

print("Python CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
print("Python NVIDIA_VISIBLE_DEVICES:", os.environ.get("NVIDIA_VISIBLE_DEVICES", "<unset>"))
print("ctypes find_library cuda:", ctypes.util.find_library("cuda"))
print("ctypes find_library cudart:", ctypes.util.find_library("cudart"))

for lib_name in ("libcuda.so.1", "libcuda.so", "libcudart.so", "libnvidia-ml.so.1"):
    try:
        ctypes.CDLL(lib_name)
        print(f"ctypes load {lib_name}: OK")
    except Exception as exc:
        print(f"ctypes load {lib_name}: {exc!r}")

try:
    cuda = ctypes.CDLL("libcuda.so.1")
    count = ctypes.c_int()
    init_ret = cuda.cuInit(0)
    count_ret = cuda.cuDeviceGetCount(ctypes.byref(count))
    print("CUDA driver cuInit return:", init_ret)
    print("CUDA driver cuDeviceGetCount return:", count_ret)
    print("CUDA driver device count:", count.value)
except Exception as exc:
    print("CUDA driver API check failed:", repr(exc))

try:
    import torch
    print("PyTorch CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print("PyTorch check failed:", repr(exc))
PY
echo "===== End GPU diagnostics ====="

if [ ! -d "/saisdata" ]; then
  echo "Warning: /saisdata not found; continuing so prediction.json can still be produced"
fi

if [ ! -d "/saisresult" ]; then
  echo "Warning: /saisresult not found; attempting to create it"
  mkdir -p /saisresult
fi

python3 /app/src/run_inference.py

PREDICTION_FILE="${OUTPUT_FILE:-/saisresult/prediction.json}"
if [ ! -f "${PREDICTION_FILE}" ]; then
  echo "Error: ${PREDICTION_FILE} not found"
  exit 1
fi

echo "Done!"
