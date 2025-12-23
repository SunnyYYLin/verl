set -e

fsdp_model_dir=/home/dataset-assist-0/models/verl/HybriDNA-300M-Instruct-train-32k-bs256_p8-lr1e-4-wd1e-5-gpu4/global_step_200/
hf_model_dir=${fsdp_model_dir/\/verl\//\/}

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir $fsdp_model_dir \
    --target_dir $hf_model_dir