set -x

data_path=$DATASETS/verl/Gene-CRE/test.parquet
save_path=$DATASETS/verl/results/sft-bychrom-test-32k.parquet
model_path=$MODELS/HybriDNA-300M-Instruct-bychrom

HYDRA_FULL_ERROR=1 python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.batch_size=24 \
    data.output_path=$save_path \
    +data.trust_remote_code=true \
    model.path=$model_path \
    +model.trust_remote_code=true \
    rollout.name=hf \
    rollout.temperature=0.6 \
    rollout.prompt_length=32768 \
    rollout.response_length=32768 \
    rollout.tensor_model_parallel_size=1 \
    rollout.pipeline_model_parallel_size=1 \
    rollout.data_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8
