jsonl_file = "ac_test_xxx.jsonl"
from collections import Counter
import re
import math
error = 1080 * 0.14
click_correct = 0
import json
correct = {"open_app":0,"click":0,"scroll":0,"input_text":0,"navigate_back":0}
total = {"open_app":0,"click":0,"scroll":0,"input_text":0,"navigate_back":0}
with open(jsonl_file, "r") as f:
    for line in f:
        answer_action = None
        action_type = None
        match = re.search(r'"content":\s*"({.*})"', line)
        action_type_pattern = r'"action_type"\s*:\s*"(\w+)"'
        if match:
            content_str = match.group(1)
        for t in correct.keys():
            if t in content_str:
                action_type = t
        response_match = re.search(r'"response":\s*"({.*})"', line)
        response = response_match.group(1) if response_match else None
        for t in correct.keys():
            if response and t in response:
                answer_action = t
        # print(action_type,answer_action)
        if action_type == answer_action:
            correct[action_type] += 1
        if answer_action == 'click' and action_type == 'click':
            # (2400,1080)  => (1484,644, )
            numbers = re.findall(r'\d+', content_str)
            gt_x = int(numbers[0])
            gt_y = int(numbers[1])
            numbers = re.findall(r'\d+', response)
            if len(numbers) <= 1:
                continue
            pred_x = int(numbers[0])
            pred_y = int(numbers[1])
            scale_x = 1080 / 644
            scale_y = 2400 / 1484
            pred_x = int(scale_x * pred_x) if pred_x else None
            pred_y = int(scale_y * pred_y) if pred_y else None
            if pred_x and pred_y and math.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2) < error:
                click_correct += 1
        total[action_type] += 1
print(click_correct)
