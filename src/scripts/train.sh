export DEBUG_MODE="true"


export DATA_PATH=../../data/ScreenSpot
export CKPT_PATH=./../ckpt/Qwen/Qwen2.5-VL-3B-Instruct
export SAVE_PATH=../../ckpt/your_model
export LOG_PATH=${SAVE_PATH}"/debug_log.txt"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    ../ui_r1/src/open_r1/grpo_json_action_coord.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --data_file_paths ../../data/ScreenSpot/example.json:../../data/AndroidControl/train.json\
    --image_folders ../../data/ScreenSpot/screenspot_imgs:../../data/AndroidControl/screenshots\
    --dataset_name ${DATA_PATH} \
    --deepspeed ../ui_r1/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 12845056 \
    --num_train_epochs 1 \
    --run_name GRPO_example \
    --save_strategy epoch \
    --save_only_model true \
    --num_generations 8
