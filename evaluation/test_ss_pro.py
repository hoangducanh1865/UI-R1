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
def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        coord_match = re.search(bbox_pattern, content_answer)
        if coord_match:
            coord = [int(coord_match.group(1)), int(coord_match.group(2))]
            x, y = coord
            return coord, False
    return [0, 0, 0, 0], False


logger = logging.getLogger(__name__)

def run(rank, world_size, args):
    if "Qwen2.5" in args.model_path:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
    else:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="cpu",
        )
    if args.ori_processor_path is None:
        ori_processor_path = args.model_path
    infer_dir = os.path.join(args.model_path,'infer')
    if not os.path.exists(infer_dir):
        os.makedirs(infer_dir)
    output_file = os.path.join(infer_dir, f'prediction_results_{args.test_name}.jsonl')
    processor = AutoProcessor.from_pretrained(ori_processor_path) 
    model = model.to(torch.device(rank))
    model = model.eval()
    
    error_count = 0
    correct_count = 0
    pred_results = []
    image_dir = os.path.join(args.ss_path,"images")
    json_dir = os.path.join(args.ss_path,"annotations")
    if args.task_name == "all":
        json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    else:
        json_files = [f"{args.task_name}.json"]
    for json_file in json_files:
        # print(os.path.join(json_dir,json_file))
        data = json.load(open(os.path.join(json_dir,json_file), "r"))
        
        data = data[rank::world_size]
        print(f"Process {rank} handling {len(data)} samples", flush=True)

        for j, item in tqdm(enumerate(data), total=len(data)):
            image_path = os.path.join(image_dir, item["img_filename"])
            task_prompt = item["instruction"]

            question_template = (
                f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
                f"Please provide the action to perform (enumerate in 'click' and 'scroll') and the coordinate where the cursor is moved to(integer) if click is performed.\n"
                "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                "The output answer format should be as follows:\n"
                "<think> ... </think> <answer>[{'action': enum['click','scroll'], 'coordinate': [x, y]}]</answer>\n"
                "Please strictly follow the format."
            )
            # w/o thinking
            # question_template = (
            #     f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
            #     f"Please provide the action to performe (enumerate in 'click' and 'swipe') and the coordinate where the cursor is moved to(integer).\n"
            #     f"If no object belonging to the category '{task_prompt}' in the image, return 'No Objects'.\n"
            #     "Output the final answer in <answer> </answer> tags."
            #     "The output answer format should be as follows:\n"
            #     "<answer>[{'action': str, 'coordinate': [x, y]}, ...]</answer>\n"
            #     "Please strictly follow the format."
            # )
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
                resized_height = inputs['image_grid_thw'][0][1] * processor.image_processor.patch_size
                resized_width = inputs['image_grid_thw'][0][2] * processor.image_processor.patch_size
                origin_height = image_inputs[0].size[1]
                origin_width = image_inputs[0].size[0]
                scale_x = origin_width / resized_width
                scale_y = origin_height / resized_height
                inputs = inputs.to(model.device)
                
                generated_ids = model.generate(**inputs, max_new_tokens=1024)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                response = response[0]
                
                gt_bbox = item["bbox"]
                pred_coord, _ = extract_coord(response)
                pred_coord = [int(pred_coord[0] * scale_x), int(pred_coord[1] * scale_y)]

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
                    json_file.write('\n')  
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
            global_results.extend(result_lists[i][2])

        logger.info(f'Error number: {global_count_error}')  
        
        logger.info('Finished running')
    
    else:
        logger.info("Not enough GPUs")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--ori_processor_path", type=str, default=None)
    parser.add_argument("--ss_path", type=str, default=None)
    parser.add_argument("--task_name", type=str, default="all")
    parser.add_argument("--test_name", type=str, required=True)
    args = parser.parse_args()
    main(args)
