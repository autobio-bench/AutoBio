# if EXP_NAME is not set, fail
if [ -z "$EXP_NAME" ]; then
    echo "Please set EXP_NAME"
    exit 1
fi

export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_7:1,mlx5_8:1,mlx5_9:1
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0
export NCCL_DEBUG=INFO
export NCCL_NVLS_ENABLE=0

export TEXT_ENCODER_NAME="$PWD/google/t5-v1_1-xxl"
export VISION_ENCODER_NAME="$PWD/google/siglip-so400m-patch14-384"
export OUTPUT_DIR="./checkpoints/$EXP_NAME"
export CFLAGS="-I/usr/include"
export LDFLAGS="-L/usr/lib/x86_64-linux-gnu"
export CUTLASS_PATH="$PWD/cutlass"

export WANDB_PROJECT="rdt"

if [ -d "$OUTPUT_DIR" ]; then
    echo "Folder '$OUTPUT_DIR' already exists"
    exit 1
fi

mkdir "$OUTPUT_DIR"

accelerate launch main.py \
    --deepspeed="$PWD/configs/zero2.json" \
    --pretrained_model_name_or_path="$PWD/rdt-1b" \
    --pretrained_text_encoder_name_or_path=$TEXT_ENCODER_NAME \
    --pretrained_vision_encoder_name_or_path=$VISION_ENCODER_NAME \
    --output_dir=$OUTPUT_DIR \
    --train_batch_size=32 \
    --sample_batch_size=64 \
    --max_train_steps=30000 \
    --checkpointing_period=500000 \
    --sample_period=500000 \
    --checkpoints_total_limit=1 \
    --lr_scheduler="constant" \
    --learning_rate=1e-4 \
    --mixed_precision="bf16" \
    --dataloader_num_workers=8 \
    --image_aug \
    --dataset_type="finetune" \
    --state_noise_snr=40 \
    --load_from_lerobot \
    --precomp_lang_embed \
    --report_to=wandb \
    --allow_tf32
