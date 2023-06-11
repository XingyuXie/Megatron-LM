#! /bin/bash

GPUS_PER_NODE=8
# Change for multinode config
# MASTER_ADDR=localhost
# MASTER_PORT=6000
NNODES=2
# NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# export DLWS_NUM_WORKER=${NNODES}
# export DLWS_NUM_GPU_PER_WORKER=${GPUS_PER_NODE}

# LLama 
NUM_LAYERS=40
HIDDEN_SIZE=5120
FFN_HIDDEN_SIZE=11008
NUM_ATTN_HEADS=40
INIT_STD=0.01275

# Megatron Model Parallelism
mp_size=2
# DeepSpeed Pipeline parallelism
pp_size=2

MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=512
TRAIN_ITERS=50000
LR=3.0e-4
MIN_LR=0.000005
MAX_GRAD_NORM=1.0
WD=1e-1
adam_beta1=0.9
adam_beta2=0.95
WARMUP=0.15
# h_g.add_(grad)
# exp_avg.lerp_(h_g, 1. - beta1)  # m_t
# exp_avg_sq.mul_(beta2).addcmul_(h_g, h_g, value=1. - beta2)


# lr_deacy=0.5
# export freeze_step=$(printf "%.0f" $(echo "$TRAIN_ITERS * $WARMUP + 1" | bc))

# # 输出结果
EXP_NAME=BP16_ori_allreduce_llama_13b_config$NUM

# EXP_NAME=zero1adam_gpt2_345m_ds_bs${GLOBAL_BATCH_SIZE}_iter${TRAIN_ITERS}_mp${mp_size}_pp${pp_size}_freeze_step${freeze_step}

DATA_PATH=/code/Megatron-DeepSpeed/data/webtext-gpt2_text_document
VOCAB_PATH=/code/Megatron-DeepSpeed/data/gpt2-vocab.json
MERGE_PATH=/code/Megatron-DeepSpeed/data/gpt2-merges.txt
OUT_DIR=checkpoints/$EXP_NAME
CHECKPOINT_PATH=$OUT_DIR/checkpoints


# script_path=$(realpath $0)
# script_dir=$(dirname $script_path)



## GPT-3 XL 1.3B
# MODEL_SIZE=1.3
# NUM_LAYERS=24
# HIDDEN_SIZE=2048
# NUM_ATTN_HEADS=16
# GLOBAL_BATCH_SIZE=512
# LR=2.0e-4
# MIN_LR=2.0e-5

## GPT-3 2.7B
# MODEL_SIZE=2.7
# NUM_LAYERS=32
# HIDDEN_SIZE=2560
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=512
# LR=1.6e-4
# MIN_LR=1.6e-5

## GPT-3 6.7B
# MODEL_SIZE=6.7
# NUM_LAYERS=32
# HIDDEN_SIZE=4096
# NUM_ATTN_HEADS=32
# GLOBAL_BATCH_SIZE=1024
# LR=1.2e-4
# MIN_LR=1.2e-5

## GPT-3 13B
# MODEL_SIZE=13
# NUM_LAYERS=48
# HIDDEN_SIZE=4224
# NUM_ATTN_HEADS=48
# GLOBAL_BATCH_SIZE=1024
# LR=1.0e-4
# MIN_LR=1.0e-5

## GPT-3 175B
# MODEL_SIZE=175
# NUM_LAYERS=96
# HIDDEN_SIZE=12288
# NUM_ATTN_HEADS=96
# GLOBAL_BATCH_SIZE=1536
# LR=0.6e-4
# MIN_LR=0.6e-5

LOGDIR="tensorboard_data/${EXP_NAME}_LM"

mkdir -p $OUT_DIR
config_json=$OUT_DIR/lm_config.json
# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()


gpt_options=" \
        --sequence-parallel \
        --init-method-std $INIT_STD \
        --recompute-granularity full \
        --recompute-method uniform \
        --attention-dropout 0.1 \
        --hidden-dropout 0.1 \
        --bf16 \
        --use-flash-attn \
        --tensor-model-parallel-size ${mp_size} \
        --pipeline-model-parallel-size ${pp_size} \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --ffn-hidden-size $FFN_HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTN_HEADS \
        --seq-length 2048  \
        --max-position-embeddings 2048 \
        --micro-batch-size $MICRO_BATCH_SIZE \
        --global-batch-size $GLOBAL_BATCH_SIZE \
        --train-iters $TRAIN_ITERS \
        --save $CHECKPOINT_PATH \
        --load $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_PATH \
        --merge-file $MERGE_PATH \
        --DDP-impl local \
        --split 949,50,1 \
        --distributed-backend nccl \
        --optimizer adam \
        --adam-beta1 $adam_beta1 \
        --adam-beta2 $adam_beta2 \
        --lr $LR \
        --lr-decay-style cosine \
        --min-lr $MIN_LR \
        --clip-grad $MAX_GRAD_NORM \
        --weight-decay $WD \
        --lr-warmup-fraction $WARMUP \
        --use-distributed-optimizer \
        --no-load-optim \
        --no-load-rng \
        --override-opt_param-scheduler \
        --log-interval 200 \
        --save-interval 100000 \
        --eval-interval 2000 \
        --eval-iters 50 \
        --tensorboard-dir $LOGDIR \
"


full_options="${gpt_options}"

export LAUNCHER="python -m torch.distributed.launch \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $RANK \
    "


run_cmd="$LAUNCHER pretrain_gpt.py ${full_options}"
echo ${run_cmd}
eval ${run_cmd} 2>&1 | tee $OUT_DIR/output.log

set +x
