# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import PIL

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

# from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from open_r1.trainer import Qwen2VLGRPOTrainer,DASTQwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import json

# GRPO训练参数
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )    
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

# ============================ 自定义获取坐标/坐标框/动作类型 ===================================
# def extract_coord(response):
#     answer_tag_pattern = r'<answer>(.*?)</answer>'
#     bbox_pattern = r'\[(\d+),\s*(\d+)]'
#     content_answer_match = re.search(answer_tag_pattern, response, re.DOTALL)
#     if content_answer_match:
#         content_answer = content_answer_match.group(1).strip()
#         coord_match = re.search(bbox_pattern, content_answer)
#         if coord_match:
#             coord = [int(coord_match.group(1)), int(coord_match.group(2))]
#             return coord , True
#     return [0, 0], False
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
def extract_bbox(response):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)]'
    content_answer_match = re.search(answer_tag_pattern, response, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        coord_match = re.search(bbox_pattern, content_answer)
        if coord_match:
            coord = [int(coord_match.group(1)), int(coord_match.group(2)), int(coord_match.group(3)), int(coord_match.group(4))]
            return coord, True
    return [0, 0, 0, 0] , False


def extract_action(response):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'action':\s*'(\w+)'"
    action_pattern_1 = r"'action':\s*(\w+)"
    content_answer_match = re.search(answer_tag_pattern, response, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
        action_match = re.search(action_pattern_1, content_answer)
        if action_match:
            return action_match.group(1)
    return None


# todo: 这是啥？
def accuracy_reward_action(completions, solution, **kwargs):
    contents = [completion[0]["content"] for completion in completions]


def accuracy_reward_action(completions, solution, scales, **kwargs):
    """ 动作类型reward：判断动作类型是否一致
    Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    show_flage = False
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        # print("content: ", content)
        # print("sol: ", sol)
        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and student_answer_action == ground_truth_action:
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of Action: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if student_answer_action and ground_truth_action:
                    f.write(f"student_answer_action: {student_answer_action}\n")
                    f.write(f"ground_truth_action: {ground_truth_action}\n")
    return rewards


def accuracy_reward_coord(completions, solution,scales, **kwargs):
    """ 动作坐标reward：判断预测的动作坐标是否在真值的坐标框内
    Reward function that checks if the completion is correct using either symbolic verification or exact string matching.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")

    show_flage = False
    for content, sol, scale in zip(contents, solution,scales):
        reward = 0.0
        # Try symbolic verification first
        # print("content: ", content)
        # print("sol: ", sol)
        try:
            student_answer_action = extract_action(content)
            ground_truth_action = extract_action(sol)
            if student_answer_action and ground_truth_action and student_answer_action == ground_truth_action:
                if student_answer_action == "click":
                    student_answer_coord, flag1 = extract_coord(content)
                    student_answer_coord = [int(student_answer_coord[0] * scale[0]), int(student_answer_coord[1] * scale[1])]
                    ground_truth_bbox, flag2 = extract_bbox(sol)
                    show_flage = flag1 and flag2
                    if ground_truth_bbox[0] <= student_answer_coord[0] <= ground_truth_bbox[2] and ground_truth_bbox[1] <= student_answer_coord[1] <= ground_truth_bbox[3]:
                        reward = 1.0
                    else:
                        reward = 0.0
                else:
                    reward = 1.0
            else:
                reward = 0.0
        except Exception:
            pass  # Continue to next verification method if this fails
                
        rewards.append(reward)
        # import pdb; pdb.set_trace()
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward of Coord: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
                if show_flage:
                    f.write(f"student_answer_coord: {student_answer_coord}\n")
                    f.write(f"ground_truth_bbox: {ground_truth_bbox}\n")
    return rewards


def format_reward(completions, **kwargs):
    """ 输出格式reward
    Reward function that checks if the completion has a specific format.
    """
    pattern = r"<answer>.*?</answer>"
    # pattern = r"<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    # matches = [re.match(pattern, content) for content in completion_contents]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

# 三个reward的定义
# action_type对应的reward
# 坐标对应的reward
# 输出格式对应的reward
###  reward registry three parts
reward_funcs_registry = {
    "accuracy_action": accuracy_reward_action,
    "accuracy_coord": accuracy_reward_coord,
    "format": format_reward,
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False
    
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['accuracy_action','accuracy_coord','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset from huggingface
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    # Load the dataset from local disk
    from datasets import DatasetDict
    # dataset = DatasetDict.load_from_disk(script_args.dataset_name)
    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    
    # if script_args.reward_method is None:
    #     accu_reward_methods = ["default"] * len(data_files)
    # else:
    #     accu_reward_methods = script_args.reward_method.split(":")
    #     assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"

    
    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")
    all_data = []
    for data_file, image_folder in zip(data_files, image_folders):
        with open(data_file, 'r') as f:
            # for line in f:
            data = json.load(f)
            for item in data:
                if 'img_filename' in item:
                    # Store image path instead of loading the image
                    item['image_path'] = os.path.join(image_folder, item['img_filename'])
                    del item['img_filename'] # remove the image column so that it can be loaded later
                # Remove immediate image loading
                task_prompt = item['instruction']
                item['problem'] = (
                    f"In this UI screenshot, I want to perform the command '{task_prompt}'.\n"
                    "Please provide the action to perform (enumerate in ['click'])"
                    "and the coordinate where the cursor is moved to(integer) if click is performed.\n"
                    "Output the final answer in <answer> </answer> tags directly."
                    "The output answer format should be as follows:\n"
                    "<answer>[{'action': 'click', 'coordinate': [x, y]}]</answer>\n"
                    "Please strictly follow the format."
                )
                if 'bbox' in item:
                    item['solution'] = f"<answer>[{{'action': 'click' ,'coordinate': {item['bbox']} }}]</answer>"
                else:
                    item['solution'] = f"<answer>[{{'action': '{item['action']}' ,'coordinate': [0,0,0,0]}}]</answer>"
                # Handle solution that could be a float or string
                # if isinstance(solution_value, str):
                #     item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                # else:
                #     # If it's a float or other non-string type, keep it as is
                #     item['solution'] = str(solution_value)
                
                # del item['conversations']
                # item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method) # if accu_reward_method is in the data jsonl, use the value in the data jsonl, otherwise use the defined value
                all_data.append(item)

    dataset = Dataset.from_list(all_data)
    def make_conversation_from_json(example):
        if 'image_path' in example and example['image_path'] is not None:
            # Don't load image here, just store the path
            return {
                # 'image': PIL.Image.open(example['image_path']),
                'image_path': example['image_path'],  # Store path instead of loaded image
                # 'problem': example['problem'],
                'solution': example['solution'],
                # 'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'text': None},
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }
        else:
            return {
                'problem': example['problem'],
                'solution': example['solution'],
                # 'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': example['problem']}
                    ]
                }]
            }

    dataset = dataset.map(make_conversation_from_json, num_proc=8)
    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)


    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
