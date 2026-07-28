jsonl_file = ".../infer/prediction_results_ScreenSpot-pro-all.jsonl"
ref_dir = ".../datasets/ScreenSpot-Pro/annotations"
import os
import json
def process_ref_dir(ref_dir):
    result_dict = {}
    application_dict = {}
    # 遍历ref_dir目录下的所有文件
    for filename in os.listdir(ref_dir):
        if filename.endswith('.json'):  # 只处理JSON文件
            file_path = os.path.join(ref_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)  # 加载JSON数据

                # 处理每个文件中的项目
                for item in data:
                    bbox = item.get('bbox')
                    ui_type = item.get('ui_type')
                    group = item.get('group')
                    application = item.get('application')

                    if bbox is not None and ui_type is not None and group is not None:
                        # 使用str(bbox)作为字典的键
                        bbox_str = str(bbox)
                        ui_type_group = f"{ui_type}_{group}"  # 假设以"_"连接ui_type和group
                        result_dict[bbox_str] = ui_type_group
                        application_dict[bbox_str] = application
    
    return result_dict,application_dict
import re
def extract_bbox(content):
    # 调整正则表达式以匹配负整数
    bbox_pattern = r'\{.*\[\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+)\s*]\s*.*\}'
    bbox_match = re.search(bbox_pattern, content)
    if bbox_match:
        # 将匹配的数值转换为整数
        bbox = [int(bbox_match.group(i)) for i in range(1, 5)]
        return bbox
    return [0, 0, 0, 0]
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
            return coord
        # else:
        #     coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
        #     coord_match = re.search(coord_pattern, content)
        #     if coord_match:
        #         coord = [int(coord_match.group(1)), int(coord_match.group(2))]
        # return coord, True
    return [0, 0]

group = ["Dev", "Creative", "CAD", "Scientific", "Office", "OS"]
type = ["text", "icon"]
result = {}
total = {}
total_application = {}
result_application = {}
for g in group:
    for t in type:
        result[t + "_" + g] = 0
        total[t + "_" + g] = 0

ref_dict, application_dict = process_ref_dir(ref_dir)
with open(jsonl_file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        coord = extract_coord(line)
        bbox = extract_bbox(line)
        total[ref_dict[str(bbox)]] += 1
        if application_dict[str(bbox)] not in total_application:
            total_application[application_dict[str(bbox)]] = 1
            result_application[application_dict[str(bbox)]] = 0
        else:
            total_application[application_dict[str(bbox)]] += 1
        if "true" in line:
            result[ref_dict[str(bbox)]] += 1
            result_application[application_dict[str(bbox)]] += 1
for k in result.keys():
    print(k, result[k] / total[k])
for k in result_application.keys():
    print(k, result_application[k] / total_application[k])    
