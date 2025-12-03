# Tested with 2 & 4 GPUs

set -x

nproc_per_node=1
DATASET_DIR=$DATASETS/verl/gsm8k
MODEL_DIR=$MODELS/HybriDNA-300M-base
SAVE_DIR=$MODELS/verl/debug

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$DATASET_DIR/train.parquet \
    data.val_files=$DATASET_DIR/test.parquet \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=1 \
    data.max_length=1024 \
    data.truncation=left \
    model.partial_pretrain=$MODEL_DIR \
    model.trust_remote_code=true \
    trainer.default_local_dir=$SAVE_DIR \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-gemma-2b-it \
    trainer.total_epochs=2 \
    trainer.logger='["console","swanlab"]' $@