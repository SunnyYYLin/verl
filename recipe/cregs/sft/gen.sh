set -x

export target="offline"
export eager="false"
export nproc_per_node=1
export max_prompt_length_by_k=4
export response_length=512
export batch_size=64
export n_samples=1
export sample_size=10
export temperature=0.01
export tp=1
export pp=1
export dp=1
export sft_ckpt_dir=/vepfs-mlp2/mlp-public/zhongcuiting/models
export dataset_dir=/vepfs-mlp2/mlp-public/zhongcuiting/verl_dataset/org_topK_sft_input/K562_abc_org_hg19_top0.01
export save_dir=/vepfs-mlp2/mlp-public/zhongcuiting/verl_output/K562_abc_org_hg19_top0.01_SFT_pure_base
export model_dir=$sft_ckpt_dir/HybriDNA-300M-instruct-pure_base
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))

#data.n_samples = 每个 prompt 生成几个结果
data_path=$dataset_dir/test.parquet

#可选：如果测试集很大，可以先用 sample_parquet.py 从测试集中采样一小部分进行测试，生成完成后再对整个测试集进行生成。
sample_path=/vepfs-mlp2/mlp-public/zhongcuiting/verl/tmp/test_sample_${sample_size}_$(date +%s).parquet
python scripts/sample_parquet_filteredsep.py \
    --input $data_path \
    --output $sample_path \
    --n $sample_size

data_path=$sample_path



data_base=$(basename "$data_path")
data_name="${data_base%.*}"
ckpt_name=$(basename "$model_dir")
model_name=$(basename "$(dirname "$model_dir")")
timestamp=$(date +%Y%m%d_%H%M%S)

prompt_length=$(( max_prompt_length_by_k * 1024 ))

max_num_batched_tokens=$(( $prompt_length + $response_length + 10240 ))

save_path="${save_dir}/${model_name}/${ckpt_name}-${timestamp}.parquet"

if [[ "$target" == *debug* ]]; then
     EXTRA_ENV='CUDA_LAUNCH_BLOCKING=1 NCCL_DEBUG=INFO PYTHONFAULTHANDLER=1 TORCH_DISTRIBUTED_DEBUG=DETAIL PYTHONUNBUFFERED=1 VERL_SFT_LOGGING_LEVEL=DEBUG TRANSFORMERS_VERBOSITY=debug HYDRA_FULL_ERROR=1'
else
     EXTRA_ENV=''
fi
export $EXTRA_ENV

python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$nproc_per_node \
    data.path=$data_path \
    data.prompt_key=prompt \
    data.batch_size=$batch_size \
    data.output_path=$save_path \
    data.n_samples=$n_samples \
    +data.trust_remote_code=true \
    +data.metadata_keys='["extra_info"]' \
    model.path=$model_dir \
    +model.trust_remote_code=true \
    rollout.name=vllm \
    rollout.mode=sync \
    rollout.log_prob_micro_batch_size_per_gpu=16 \
    rollout.temperature=$temperature \
    rollout.prompt_length=$prompt_length \
    rollout.response_length=$response_length \
    rollout.tensor_model_parallel_size=$tp \
    rollout.pipeline_model_parallel_size=$pp \
    rollout.data_parallel_size=$dp \
    rollout.gpu_memory_utilization=0.85 \
    rollout.enforce_eager=$eager \
    rollout.max_num_batched_tokens=$max_num_batched_tokens
