set -x

# export nproc_per_node=1
# export max_prompt_length_by_k=4
# export response_length_by_k=4
# export batch_size=8
# export temperature=0.01
# export tp=1
# export pp=1
# export dp=1
# export dataset_dir=$DATASETS/verl/Gene-CRE
# export model_dir=$MODELS/GENERator-v2-eukaryote-1.2b-base

data_path=$dataset_dir/test.parquet
data_base=$(basename "$data_path")
data_name="${data_base%.*}"
model_base=$(basename "$model_dir")
model_name="${model_base%.*}"
timestamp=$(date +%Y%m%d_%H%M%S)

save_dir=${DATASETS}/verl/results
save_path="${save_dir}/${data_name}-${model_name}-${timestamp}.parquet"

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$nproc_per_node \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.batch_size=$batch_size \
    data.output_path=$save_path \
    +data.trust_remote_code=true \
    model.path=$model_dir \
    +model.trust_remote_code=true \
    rollout.name=vllm \
    rollout.mode=sync \
    rollout.temperature=$temperature \
    rollout.prompt_length=$(( max_prompt_length_by_k * 1024 )) \
    rollout.response_length=$(( response_length_by_k * 1024 )) \
    rollout.tensor_model_parallel_size=$tp \
    rollout.pipeline_model_parallel_size=$pp \
    rollout.data_parallel_size=$dp \
    rollout.gpu_memory_utilization=0.8
