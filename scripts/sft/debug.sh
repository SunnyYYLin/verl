# Tested with 2 & 4 GPUs

set -x

nproc_per_node=1
DATASET_DIR=$DATASETS/verl/Gene-CRE
MODEL_DIR=$MODELS/HybriDNA-300M-base
SAVE_DIR=$MODELS/verl/HybriDNA-300M-Instruct-debug

HYDRA_FULL_ERROR=1 torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
     data.train_files=$DATASET_DIR/train.parquet \
     data.val_files=$DATASET_DIR/test.parquet \
     data.prompt_key=extra_info \
     data.response_key=extra_info \
     data.prompt_dict_keys=['gene_seq'] \
     +data.response_dict_keys=['cre_seq'] \
     data.train_batch_size=64 \
     data.micro_batch_size_per_gpu=4 \
     data.max_length=32768 \
     data.truncation=right \
     model.partial_pretrain=$MODEL_DIR \
     model.trust_remote_code=true \
     model.fsdp_config.cpu_offload=false \
     model.fsdp_config.offload_params=false \
     model.enable_gradient_checkpointing=true \
     optim.lr=1e-4 \
     optim.weight_decay=1e-5 \
     optim.lr_scheduler=cosine \
     trainer.save_freq=64 \
     trainer.test_freq=64 \
     trainer.default_local_dir=$SAVE_DIR \
     trainer.project_name=HybriDNA-300M-Instruct-debug \
     trainer.experiment_name=debug-32k-bs64_p4-lr1e-4-gpu1 \
     trainer.total_epochs=2 \
     trainer.resume_mode=disable \
     trainer.logger='["console","swanlab"]' $@