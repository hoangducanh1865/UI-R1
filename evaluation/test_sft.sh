MODEL_PATH="../ckpt/qwen2.5_vl-7b/lora-sft"
IMG_PATH="../data/ScreenSpot/screenspot_imgs"
TEST_JSON="./data/ScreenSpot/screenspot_web.json"
TEST_NAME="ScreenSpot-web"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python test_coord_lora_sft.py \
    --model_path ${MODEL_PATH} \
    --image_path ${IMG_PATH} \
    --test_json ${TEST_JSON} \
    --test_name ${TEST_NAME}
