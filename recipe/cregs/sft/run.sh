#!/bin/bash

# export target="train"
# export nproc_per_node=1
# export max_prompt_length_by_k=4
# export learning_rate=1e-4
# export weight_decay=1e-5
# export batch_size=128
# export batch_size_per_gpu=8
# export dataset_dir=$DATASETS/verl/Gene-CRE
# export model_dir=$MODELS/GENERator-v2-eukaryote-1.2b-base

set -x

EXPERIMENT_NAME="${target}-32k-bs${batch_size}_p${batch_size_per_gpu}-lr${learning_rate}-wd${weight_decay}-gpu${nproc_per_node}"
MODEL_BASE=$(basename "$model_dir")
SAVE_DIR=$MODELS/verl/$MODEL_BASE-$EXPERIMENT_NAME
MAX_PROMPT_LENGTH=$(( max_prompt_length_by_k * 1024 ))

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
     data.train_files=$dataset_dir/train.parquet \
     data.val_files=$dataset_dir/val.parquet \
     data.prompt_key=extra_info \
     data.response_key=extra_info \
     data.prompt_dict_keys=['gene_seq'] \
     +data.response_dict_keys=['cre_seq'] \
     data.train_batch_size=$batch_size \
     data.micro_batch_size_per_gpu=$batch_size_per_gpu \
     data.max_length=$MAX_PROMPT_LENGTH \
     data.truncation=right \
     model.partial_pretrain=$model_dir \
     model.trust_remote_code=true \
     model.fsdp_config.cpu_offload=false \
     model.fsdp_config.offload_params=false \
     model.enable_gradient_checkpointing=true \
     optim.lr=$learning_rate \
     optim.weight_decay=$weight_decay \
     optim.lr_scheduler=cosine \
     trainer.save_freq=50 \
     trainer.test_freq=50 \
     trainer.default_local_dir=$SAVE_DIR \
     trainer.project_name=$MODEL_BASE \
     trainer.experiment_name=$EXPERIMENT_NAME \
     trainer.total_epochs=2 \
     trainer.resume_mode=auto \
     trainer.nnodes=1 \
     trainer.n_gpus_per_node=$nproc_per_node \
     trainer.logger='["console","swanlab"]' $@