export DEBUG_MODE="true"

export DATA_PATH=../../data/UI-R1-low
export CKPT_PATH=../../ckpt/Qwen/Qwen2.5-VL-3B-Instruct
export SAVE_PATH=../../ckpt/Qwen2.5-VL-UI-R1-ground-dast-4ep
export LOG_PATH=${SAVE_PATH}"/debug_log.txt"
export Train_PATH=${SAVE_PATH}"/train.log"
mkdir -p $SAVE_PATH
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    ../ui_r1/src/open_r1/grpo_json_action_coord-dast.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --data_file_paths ../../data/UI-R1-low/train_ground.json\
    --image_folders ../../data/UI-R1-low/train_imgs\
    --dataset_name ${DATA_PATH} \
    --deepspeed ../ui_r1/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to none \
    --max_completion_length 512 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 12845056 \
    --num_train_epochs 4 \
    --run_name GRPO_example \
    --save_strategy epoch \
    --save_only_model true \
    --num_generations 8 \
    >> $Train_PATH 2>&1

export DEBUG_MODE="true"

export DATA_PATH=../../data/UI-R1-low
export CKPT_PATH=../../ckpt/Qwen2.5-VL-UI-R1-ground-dast-4ep
export SAVE_PATH=../../ckpt/Qwen2.5-VL-UI-R1-ground-dast-nothink-8ep
export LOG_PATH=${SAVE_PATH}"/debug_log.txt"
export Train_PATH=${SAVE_PATH}"/train.log"
mkdir -p $SAVE_PATH
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12346" \
    ../ui_r1/src/open_r1/grpo_json_action_coord-nothink.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --data_file_paths ../../data/UI-R1-low/train_ground.json\
    --image_folders ../../data/UI-R1-low/train_imgs\
    --dataset_name ${DATA_PATH} \
    --deepspeed ../ui_r1/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to none \
    --max_completion_length 512 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 12845056 \
    --num_train_epochs 4 \
    --run_name GRPO_example \
    --save_strategy epoch \
    --save_only_model true \
    --num_generations 8 \
    >> $Train_PATH 2>&1
