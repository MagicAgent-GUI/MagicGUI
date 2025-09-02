import pandas as pd
import numpy as np
import json
import re
import os
import os.path as osp
import shutil
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.image as mpimg
from PIL import Image


ENTITIES_TYPES = ['输入框', '图标', '文本按钮', '选择按钮', '按钮', '下拉选项框', '开关', '多重滚动选择器', '滚动选择器', '导航栏', '列表项', '弹窗', '卡片视图', '日历选择器', '页面指示器', '日期选择器', '选项区', '选项框', '滑块', '文本', '通知', '广告']

def extract_entity_type_from_prompt(prompt):
    for entity_type in ENTITIES_TYPES:
        if prompt.find(entity_type) != -1:
            return entity_type
    return prompt

def extract_infer_result_by_re(prompt, patterns):
    prompt = prompt.replace(' ', '').replace('\n', '')
    for pattern in patterns:
        results = re.findall(pattern, prompt)
        if results:
            res = results[0].strip('：').strip('。').strip('“').strip('”')
            return res
        else:
            continue
    return prompt

def load_jsonl(f):
    lines = open(f, encoding='utf-8').readlines()
    lines = [x.strip() for x in lines]
    if lines[-1] == '':
        lines = lines[:-1]
    raw_data = [json.loads(x) for x in lines]
    return raw_data

def dump_jsonl(data, result_file):
    result = []
    for i in range(len(data)):
        res = json.dumps(data[i], ensure_ascii=False)
        result.append(res)
    fd = os.open(result_file, os.O_WRONLY | os.O_CREAT, 0o600)  
    with os.fdopen(fd, 'w', encoding='utf-8') as fout:
        fout.write('\n'.join(result))

def point_isin_bbox(point, bbox):
    x, y = point
    x1, y1, x2, y2 = bbox
    if (x1 <= x <= x2) and (y1 <= y <= y2):
        return True
    else:
        return False

def point_match(point1, point2, max_distince):
    distance = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    if distance <= max_distince:
        return True
    else:
        return False

def points_distance(point1, point2, image_paths):
    try:
        image_path = eval(image_paths)
    except:
        image_path = image_paths
    if type(image_path) == list:
        image_path = image_path[0]
    with Image.open(image_path) as img:
        width, height = img.size
    
    for point in [point1, point2]:
        point[0] = point[0] * width / 1000
        point[1] = point[1] * height / 1000

    distance = math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    return distance
    

def bbox_iou(bbox1, bbox2):
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1]))

    area1 = (xmax1 - xmin1 ) * (ymax1 - ymin1) 
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    iou = inter_area / (area1 + area2 - inter_area)
    return iou


def parse_action_output(output_text):
    thought_match = re.search(r'Thought:(.*?)\nAction:', output_text, re.DOTALL)
    thought = thought_match.group(1).strip() if thought_match else ""

    action_match = re.search(r'Action:(.*?)(?:\n|$)', output_text, re.DOTALL)
    action_text = action_match.group(1).strip() if action_match else ""

    result = {
        "thought": thought,
        "action": "",
        "key": None,
        "content": None,
        "start_box": None,
        "end_box": None,
        "direction": None
    }

    if not action_text:
        return json.dumps(result, ensure_ascii=False)

    action_parts = action_text.split('(')
    action_type = action_parts[0]
    result["action"] = action_type

    if len(action_parts) > 1:
        params_text = action_parts[1].rstrip(')')
        params = {}

        for param in params_text.split(','):
            param = param.strip()
            if '=' in param:
                key, value = param.split('=', 1)
                key = key.strip()
                value = value.strip().strip('\'"')

                if 'box' in key:
                    numbers = re.findall(r'\d+', value)
                    if numbers:
                        coords = [int(num) for num in numbers]
                        if len(coords) == 4:
                            if key == 'start_box':
                                result["start_box"] = coords
                            elif key == 'end_box':
                                result["end_box"] = coords
                elif key == 'key':
                    result["key"] = value
                elif key == 'content':
                    value = value.replace('\\n', '\n').replace('\\"', '"').replace("\\'", "'")
                    result["content"] = value
                elif key == 'direction':
                    result["direction"] = value

    return json.dumps(result, ensure_ascii=False, indent=2)


def adjust_res_from_uitars_to_jarvis(rsp):
    parsed_output=json.loads(parse_action_output(rsp))
    action = parsed_output['action']
    if action=='scroll':
        bbox = parsed_output['start_box']
        direction = parsed_output['direction']
        tmp_action = f'scroll({bbox[0]}, {bbox[1]}, {direction})'
    elif action == 'click':
        bbox=parsed_output['start_box']
        tmp_action=f'tap({bbox[0]}, {bbox[1]})'
    elif action == 'long_press':
        bbox=parsed_output['start_box']
        tmp_action=f'long_press({bbox[0]}, {bbox[1]})' 
    elif action=='type':
        content = parsed_output['content']
        tmp_action = f'text(0,0,\"{content}\")'
    elif action=='finished':
        tmp_action = f"finish()"
    elif action =='wait':
        tmp_action = 'wait()'
    elif action =='drag':
        bbox1=parsed_output['start_box']
        bbox2=parsed_output['end_box']
        tmp_action = f'drag({bbox1[0]}, {bbox1[1]}, {bbox2[0]}, {bbox2[1]})'
    else:
        return rsp
    return tmp_action

def extract_action(action_str, patterns):
    action_str = str(action_str).replace('（','(').replace('）',')')
    if re.search(r'Thought:(.*?)\nAction:', action_str, re.DOTALL):
        try:
            action_str = adjust_res_from_uitars_to_jarvis(action_str)
        except:
            print(f"adjust res from uitars to javis failed, action is {action_str}")
    for action in patterns.keys():
        if action in action_str:
            pattern = patterns[action]['pattern']
            args = re.findall(pattern, action_str.replace('\n', '').replace(' ', ''))

            if args:
                if patterns[action]['agrs_nums'] == 1:
                    args = [tuple(args)]

                if action == 'click':
                    action = 'tap'
                if action == 'type':
                    action = 'text'
                if len(args[0]) == patterns[action]['agrs_nums']:
                    return (action, args)
            else:
                print(f"wrong normal match")
            
            if action == 'scroll':
                args = re.findall(r'\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*', action_str.replace('\n', '').replace(' ', ''))
                point_x = 0
                point_y = 0
                if len(args) > 0:
                    point_x = args[0][0]
                    point_y = args[0][1]

                dir = [d for d in ['up','down','left','right'] if d in action_str]
                if len(dir) == 1:
                    return (action, [(point_x,point_y,dir[0])])
                else:
                    print(f"scroll wrong match")
            if action == 'click':
                args = re.findall(r'\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*', action_str.replace('\n', '').replace(' ', ''))
                if len(args) == 1:
                    point_x = str(int((int(float(args[0][0])) +  int(float(args[0][2]))) / 2))
                    point_y = str(int((int(float(args[0][1])) +  int(float(args[0][3]))) / 2))
                    return ('tap', [(point_x,point_y)])
                else:
                    print(f"click wrong match")
            if action == 'drag':
                args = re.findall(r'\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*', action_str.replace('\n', '').replace(' ', ''))
                if len(args) == 2:
                    point_xs = str(int((int(float(args[0][0])) +  int(float(args[0][2]))) / 2))
                    point_ys = str(int((int(float(args[0][1])) +  int(float(args[0][3]))) / 2))

                    point_xe = str(int((int(float(args[0][0])) +  int(float(args[0][2]))) / 2))
                    point_ye = str(int((int(float(args[0][1])) +  int(float(args[0][3]))) / 2))
                    return ('drag', [(point_xs,point_ys,point_xe,point_ye)])
                else:
                    print(f"drag wrong match")
    
    if 'press_home' in action_str:
        return ('navigate_home', [('')])
    if 'press_back' in action_str:
        return ('navigate_back', [('')])
    if 'finish' in action_str:
        return ('finish', [('')])
    if 'hotkey' in action_str:
        args = re.findall(r'key=(.*)', action_str.replace('\n', '').replace(' ', ''))
        if '截图' in args[0]:
            return ('screen_shot', [('')])
    if 'open_app' in action_str:
        args = re.findall(r'app_name=(.*)\)', action_str.replace('\n', '').replace(' ', ''))
        if args:
            return ('call_api', [(f"{args[0]}", 'open')])
        else:
            print(f"open_app wrong match")
    try:
        json_obj = json.loads(action_str)
        if 'to' in json_obj:
            if 'POINT' in json_obj:
                if type(json_obj['to']) is str:
                    return ('scroll', [(json_obj['POINT'][0], json_obj['POINT'][1], json_obj['to'])])
                elif type(json_obj['to']) is list:
                    return ('drag', [(json_obj['POINT'][0], json_obj['POINT'][1], json_obj['to'][0], json_obj['to'][1])])
                else:
                    print(f"point to error: {json_obj}")
            else:
                print(f"point to error: {json_obj}")

        elif 'TYPE' in json_obj:
            if 'POINT' in json_obj:
                return ('text', [(json_obj['POINT'][0], json_obj['POINT'][1]),json_obj['TYPE']])
            else:
                return ('text', [0, 0,json_obj['TYPE']])
        elif 'STATUS' in json_obj:
            if json_obj['STATUS'] in ['impossible','interrupt']:
                return ('no_answer', [('')])
            elif json_obj['STATUS'] in ['finish','satisfied']:
                return ('action_completed', [('')])
            elif json_obj['STATUS'] in ['need_feedback']:
                return ('take_over', [('')])
            else:
                print(f"status error: {json_obj}")

        elif 'PRESS' in json_obj:
            if json_obj['PRESS'] == 'HOME':
                return ('navigate_home', [('')])
            elif json_obj['PRESS'] == 'BACK':
                return ('navigate_back', [('')])
            elif json_obj['PRESS'] == 'ENTER':
                return ('enter', [('')])
            elif json_obj['PRESS'] == 'APPSELECT':
                return (None, [('')])
            else:
                print(f"press error: {json_obj}")

        elif 'duration' in json_obj:
            return ('wait', [('')]) 

        elif 'POINT' in json_obj:
            return ('tap', [(json_obj['POINT'][0], json_obj['POINT'][1])])

    except:
        print(f'json load or parse action result failed , action result is {action_str}')
    print(f"no pattern match: {action_str}")
    return (None, [('')])
    
def extract_point(point_str):
    try:
        if '<|box_start|>' in point_str:
            pattern = r'<\|box_start\|>\((.*?),(.*?)\)<\|box_end\|>'
        elif '<point>' in point_str:
            pattern = r'<point>\((.*?),(.*?)\)</point>'
        else:
            pattern = r'.*\(\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*).*\).*'
        point = re.findall(pattern, point_str)
        if not point:
            point = []
        else:
            point = [int(value) for value in point[0]]
    except Exception as e:
        point = []

    return point

def extract_bboxs(bboxes_str, pattern1, pattern2=None):
    bboxes = []
    try:
        for pattern in [pattern1, pattern2]:
            if pattern:
                bboxes_tmp = re.findall(pattern, bboxes_str.replace('\n', '').replace(' ', ''))
                if bboxes_tmp:
                    for bbox in bboxes_tmp:  
                        bbox = [point_str.strip() for point_str in bbox] 
                        bboxes.append(list(map(float, bbox)))
                    break

    except:
        print(f'failed to extract bbox, bboxes is {bboxes_str}, pattern1 is {pattern1}, pattern2 is {pattern2}')
    return bboxes

def reshape_bbox(bbox, width_factor, height_factor):
    x1, y1, x2, y2 = bbox
    x1 *= width_factor
    x2 *= width_factor
    y1 *= height_factor
    y2 *= height_factor
    new_bbox = [x1, y1, x2, y2]
    return new_bbox

def reshape_data(true_box, pred_box, gt_point, pred_point, background_pic_path):
    background_pic = mpimg.imread(background_pic_path)
    height = background_pic.shape[0]
    width = background_pic.shape[1]
    width_factor = width/1000
    height_factor = height/1000

    if true_box:
        true_box = reshape_bbox(true_box, width_factor, height_factor)

    if pred_box:
        pred_box = reshape_bbox(pred_box, width_factor, height_factor)

    if gt_point:
        gt_point = [gt_point[0]*width_factor, gt_point[1]*height_factor]

    if pred_point:
        pred_point = [pred_point[0]*width_factor, pred_point[1]*height_factor]

    return background_pic, true_box, pred_box, gt_point, pred_point


def draw_plt(filename, pred_center_point, true_box, pred_box, gt_point, pred_point, background_pic=None, entity_type='unknown', figsize=(7, 15), dpi=80):
    file_path = osp.join('tmp', filename)
    if not osp.exists('tmp'):
        os.makedirs('tmp', exist_ok=True)
    if osp.exists(file_path):
        return file_path, figsize[1]*dpi, figsize[0]*dpi
    
    plt.clf()

    if background_pic is not None:
        height = background_pic.shape[0]
        width = background_pic.shape[1]
        plt.figure(figsize=(width/150, height/150), dpi=60)

    plt.rcParams['font.sans-serif']=['STXihei']
    if pred_center_point:
        plt.scatter([pred_center_point[0]], [pred_center_point[1]], s=3, linewidths=8, color = 'blue')

    for bbox, color in [(pred_box, 'blue'), (true_box, 'hotpink')]:
        if bbox:
            x1, y1, x2, y2 = bbox
            rect=mpatches.Rectangle((x1, y1,),x2-x1,y2-y1, fill=False, color=color, linewidth=4)
            plt.gca().add_patch(rect)

    for point, color in [(pred_point, 'blue'), (gt_point, 'hotpink')]:
        if point:
            plt.scatter([point[0]], [point[1]], s=5, linewidths=8, color=color)

    plt.axis('off')

    if background_pic is not None:
        plt.imshow(background_pic)
    else:
        plt.show()

    
    plt.savefig(file_path, bbox_inches = 'tight', pad_inches=0.0)
    return file_path, figsize[1]*dpi, figsize[0]*dpi

def generate_grounding_image(output_filename, image_path, true_box, pred_box, gt_point, pred_point):
    background_pic, true_box, pred_box, gt_point, pred_point = reshape_data(true_box, pred_box, gt_point, pred_point, image_path)
    if pred_box:
        pred_center_point = ((pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2)
    else:
        pred_center_point = None
    file_path, height, width = draw_plt(output_filename, pred_center_point, true_box, pred_box, gt_point, pred_point, background_pic=background_pic)
    return file_path, height, width

def write_excel(df, output_path, max_height=540, max_width=1500, grounding=False, pattern1=None, pattern2=None):
    writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

    if 'image' in df.columns:
        image_col_name = 'image'
    elif 'images' in df.columns:
        image_col_name = 'images'
    else:
        image_col_name = None

    image_col_name = None
    if not image_col_name:
        df.to_excel(writer, index=False)
        writer.close()
        return
    if osp.exists('tmp'):
        shutil.rmtree('tmp')
    image_col_index = df.columns.get_loc(image_col_name)
    df.insert(image_col_index+1, 'image_loaded', None)
    df.to_excel(writer, index=False)
    worksheet = writer.sheets['Sheet1']

    row_width = 32
    worksheet.set_column(image_col_index+1, image_col_index+1, row_width)
    for index in tqdm(range(len(df))):
        row = df.iloc[index]
        if pd.notna(row[image_col_name]):
            try:
                image_path = eval(row[image_col_name])
            except:
                image_path = row[image_col_name]
            if type(image_path) == list:
                image_path = image_path[0]
            if os.path.exists(image_path):
                gt_box = None
                pred_box = None
                gt_point = None
                pred_point = None
                for bbox_str in [row['response'], row['question']]:
                    gt_boxes = extract_bboxs(bbox_str, pattern1, pattern2)
                    if gt_boxes:
                        gt_box = gt_boxes[0]
                        break

                    gt_point = extract_point(bbox_str)
                    if gt_point:
                        break
                
                pred_boxes = extract_bboxs(row['prediction'], pattern1, pattern2)
                pred_point = extract_point(row['prediction'])
                if pred_boxes:
                    pred_box = pred_boxes[0]

                output_filename = str(index) + '_' + osp.basename(image_path)
                image_path, height, width = generate_grounding_image(output_filename, image_path, gt_box, pred_box, gt_point, pred_point)
                
                scale_w = min(1, max_width/width)
                scale_h = min(1, max_height/height)
                scale = min(scale_w, scale_h)

                new_height = int(height*scale)
                new_width = int(width*scale)

                row_height = new_height
                worksheet.set_row(index+1, 0.75*row_height)
                worksheet.insert_image(index+1, image_col_index+1, image_path, {'x_scale':scale, 'y_scale':scale})
            else:
                print(f"image path {image_path} not exists")
                raise Exception

    writer.close()




