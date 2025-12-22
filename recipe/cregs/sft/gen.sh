set -x

data_path=$DATASETS/verl/Gene-CRE/test.parquet
save_path=$DATASETS/verl/results/sft-mixed_random.parquet
model_path=$MODELS/HybriDNA-300M-base 

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=1 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=$save_path \
    +data.trust_remote_code=true \
    model.path=$model_path \
    +model.trust_remote_code=true \
    rollout.name=hf \
    rollout.temperature=0.6 \
    rollout.prompt_length=1024 \
    rollout.response_length=1024 \
    rollout.gpu_memory_utilization=0.8
