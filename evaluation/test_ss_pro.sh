MODEL_PATH="../ckpt/Qwen2.5-VL-3B-UI-R1"
SS_PATH="../data/ScreenSpot-pro"
TASK_NAME="all"
TEST_NAME="ScreenSpot-pro-"${TASK_NAME}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_coord_ss_pro.py \
    --model_path ${MODEL_PATH} \
    --ss_path ${SS_PATH} \
    --task_name ${TASK_NAME} \
    --test_name ${TEST_NAME}