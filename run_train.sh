
# 可选：指定用哪张 GPU
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "========== [1/2] build_index =========="
python build_index.py

echo "========== [2/2] train =========="
# 剩余参数全部透传给 train.py
python train.py "$@"
