set -x

data_path=$DATASETS/verl/Gene-CRE/test.parquet
save_path=$DATASETS/verl/results/sft-mixed_random.parquet
model_path=$MODELS/HybriDNA-300M-Instruct-train-32k-bs256_p8-lr1e-4-wd1e-5-gpu4/global_step_500

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
    rollout.temperature=0.6 \
    rollout.prompt_length=32768 \
    rollout.response_length=32768 \
    rollout.gpu_memory_utilization=0.8
