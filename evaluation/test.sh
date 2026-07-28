MODEL_PATH="../../UI-R1-3B"

IMG_PATH="../../test_dataset/test_images"

TEST_JSON="../../test_dataset/test.json"

TEST_NAME="test"

CUDA_VISIBLE_DEVICES=0 python test_screenspot.py \
    --model_path ${MODEL_PATH} \
    --image_path ${IMG_PATH} \
    --test_json ${TEST_JSON} \
    --test_name ${TEST_NAME}
