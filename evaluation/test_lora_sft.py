from tqdm import tqdm
import os
import json
import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor,Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import sys
import re
import multiprocessing as mp
import logging
from multiprocessing import Pool
import functools

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

rank = 0

def extract_bbox(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        bbox_match = re.search(bbox_pattern, content_answer)
        if bbox_match:
            bbox = [int(bbox_match.group(1)), int(bbox_match.group(2)), int(bbox_match.group(3)), int(bbox_match.group(4))]
            return bbox, True
    return [0, 0, 0, 0], False
def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    coord_match = re.search(bbox_pattern, content)
    if coord_match:
        coord = [int(coord_match.group(1)), int(coord_match.group(2))]
        return coord, True
    else:
        coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
        coord_match = re.search(coord_pattern, content)
        if coord_match:
            coord = [int(coord_match.group(1)), int(coord_match.group(2))]
            return coord, True
    return [0, 0], False

def iou(box1, box2):
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[2]-1, box2[2]-1)
    inter_y2 = min(box1[3]-1, box2[3]-1)
    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter
    return float(inter)/union

logger = logging.getLogger(__name__)

def run(rank, world_size, args):
    qwen_model_path = args.qwen_model_path
    qwen_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        qwen_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cpu",
    )
    adapter_model_path = args.adapter_model_path
    model = PeftModel.from_pretrained(qwen_model,adapter_model_path)
    if args.ori_processor_path is None:
        ori_processor_path = qwen_model_path
    infer_dir = os.path.join(adapter_model_path,'infer')
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)
    output_file = os.path.join(infer_dir, f'prediction_results_{args.test_name}.jsonl')

    processor = AutoProcessor.from_pretrained(ori_processor_path) 
    model = model.to(torch.device(rank))
    model = model.eval()
    
    error_count = 0
    correct_count = 0
    pred_results = []
    
    tasks = ["mobile"]
    for task in tasks:
        dataset = args.test_json 
        data = json.load(open(dataset, "r"))
        
        data = data[rank::world_size]
        print(f"Process {rank} handling {len(data)} samples", flush=True)

        for j, item in tqdm(enumerate(data), total=len(data)):
            image_path = os.path.join(args.image_path, item["img_filename"])
            task_prompt = item["instruction"]

            question_template = (
                f"You are a GUI assistant. In this UI screenshot, I want to perform the command '{task_prompt}'"
                "Please provide the action to perform (enumerate in ['click', 'scroll']) and the coordinate if click is performed."
                "The output answer format should be as follows:\n"
                "{'action': str, 'coordinate': [x, y]}\n"
                "Please strictly follow the format. Do not include any other information."
            
            )
            query = '<image>\n' + question_template
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path}
                    ] + [{"type": "text", "text": query}],
                }
            ]
            
            try:
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )

                inputs = inputs.to(model.device)
                
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response = response[0]
                print(response)
                gt_bbox = item["bbox"]
                pred_coord, _ = extract_coord(response)
                success = gt_bbox[0] <= pred_coord[0] <= gt_bbox[2] and gt_bbox[1] <= pred_coord[1] <= gt_bbox[3]
                if success:
                    correct_count += 1
                else:
                    error_count += 1
                
                new_pred_dict = {
                    'image_id': item["img_filename"],
                    'gt_bbox': gt_bbox,
                    'pred_coord': pred_coord,
                    'response': response,
                    'pred_result': success
                }
                with open(output_file, 'a') as json_file:
                    json.dump(new_pred_dict, json_file)
                    json_file.write('\n')  # 保证换行
                pred_results.append(new_pred_dict)

            except Exception as e:
                print(f"Process {rank} error: {e}", flush=True)
                error_count += 1

    return [error_count, correct_count, pred_results]

def main(args):
    multiprocess = torch.cuda.device_count() >= 2
    mp.set_start_method('spawn')
    
    if multiprocess:
        logger.info('Started generation')
        n_gpus = torch.cuda.device_count()
        world_size = n_gpus

        with Pool(world_size) as pool:
            func = functools.partial(run, world_size=world_size, args=args)
            result_lists = pool.map(func, range(world_size))

        global_count_error = 0
        global_count_correct = 0
        global_results = []

        for i in range(world_size):
            global_count_error += int(result_lists[i][0])
            global_count_correct += int(result_lists[i][1])
            global_results.extend(result_lists[i][2])  # 修正拼接方式

        logger.info(f'Error number: {global_count_error}')  

        logger.info('Finished running')
    
    else:
        logger.info("Not enough GPUs")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen_model_path", type=str, required=True)
    parser.add_argument("--adapter_model_path", type=str, default=None)
    parser.add_argument("--ori_processor_path", type=str, default=None)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--test_name", type=str, required=True)
    args = parser.parse_args()
    main(args)