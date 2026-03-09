set -e

root_dir=$1

for fsdp_model_dir in ${root_dir}/*; do
    if [ -d "$fsdp_model_dir" ]; then
        hf_model_dir=${fsdp_model_dir}-hf/

        echo "Merging $fsdp_model_dir -> $hf_model_dir"

        python scripts/legacy_model_merger.py merge \
            --backend fsdp \
            --local_dir $fsdp_model_dir \
            --target_dir $hf_model_dir
    fi
done