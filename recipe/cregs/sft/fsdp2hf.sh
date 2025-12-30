set -e

fsdp_model_dir=$1
hf_model_dir=$fsdp_model_dir-hf/

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir $fsdp_model_dir \
    --target_dir $hf_model_dir