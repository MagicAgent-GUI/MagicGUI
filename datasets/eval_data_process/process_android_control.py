from android_env.proto.a11y import android_accessibility_forest_pb2
import tensorflow as tf
import json
import os
from tqdm import tqdm
import shutil
import re

import logging
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

filenames = tf.io.gfile.glob(os.path.join(current_dir, 'tmp/android_control/android_control*'))

raw_dataset = tf.data.TFRecordDataset(filenames, compression_type='GZIP')
dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(raw_dataset)

total_samples = sum(1 for _ in tf.data.TFRecordDataset(filenames, compression_type='GZIP'))

output_dir = os.path.join(current_dir, 'tmp/android_control')
images_dir = os.path.join(output_dir, 'screenshots')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)

def get_interactive_nodes(forest):
    interactives = []
    for window in forest.windows:
        nodes = window.tree.nodes
        id_to_node = {getattr(node, 'unique_id', idx): node for idx, node in enumerate(nodes)}

        def is_interactive(node):
            return (
                getattr(node, 'is_clickable', False) or
                getattr(node, 'is_focusable', False) or
                any(action.id in [1, 16, 32] for action in node.actions)
            )

        def iterative_dfs(root):
            if not root:
                return
            stack = [root]
            visited = set()
            while stack:
                node = stack.pop()
                if id(node) in visited:
                    continue
                visited.add(id(node))
                if is_interactive(node):
                    bounds = {
                        'left': getattr(getattr(node, 'bounds_in_screen', None), 'left', 0),
                        'top': getattr(getattr(node, 'bounds_in_screen', None), 'top', 0),
                        'right': getattr(getattr(node, 'bounds_in_screen', None), 'right', 0),
                        'bottom': getattr(getattr(node, 'bounds_in_screen', None), 'bottom', 0)
                    }
                    interactives.append({
                        'unique_id': getattr(node, 'unique_id', None),
                        'class_name': node.class_name,
                        'content_description': node.content_description,
                        'text': node.text,
                        'resource_id': getattr(node, 'view_id_resource_name', 'N/A'),
                        'bounds': bounds
                    })
                for child_id in reversed(node.child_ids):
                    child_node = id_to_node.get(child_id)
                    if child_node:
                        stack.append(child_node)

        all_child_ids = {cid for node in nodes for cid in node.child_ids}
        root_nodes = [node for node in nodes if getattr(node, 'unique_id', None) not in all_child_ids]
        for root in root_nodes:
            iterative_dfs(root)
    return interactives


fd = os.open(os.path.join(output_dir, 'data.jsonl'), os.O_WRONLY | os.O_CREAT, 0o600) 
with os.fdopen(fd, 'w', encoding='utf-8') as jsonl_file:
    pbar = tqdm(total=total_samples, desc="processing data")
    index = 0
    while True:
        try:
            example = tf.train.Example.FromString(dataset_iterator.get_next().numpy())
            step_instructions = [d.decode('utf-8') for d in example.features.feature['step_instructions'].bytes_list.value]
            episode_id = [d for d in example.features.feature['episode_id'].int64_list.value]
            goal = [d.decode('utf-8') for d in example.features.feature['goal'].bytes_list.value]
            screenshot_widths = [d for d in example.features.feature['screenshot_widths'].int64_list.value]
            screenshot_heights = [d for d in example.features.feature['screenshot_heights'].int64_list.value]
            actions = [d.decode('utf-8') for d in example.features.feature['actions'].bytes_list.value]
            forests_list = []
            node_info_list = []
            for forest_bytes in example.features.feature['accessibility_trees'].bytes_list.value:
                forest = android_accessibility_forest_pb2.AndroidAccessibilityForest().FromString(forest_bytes)
                forests_list.append(str(forest))
                node_info_list.append(get_interactive_nodes(forest))
            screenshot_bytes = example.features.feature['screenshots'].bytes_list.value
            screenshot_paths = []
            for img_idx, screenshot_byte in enumerate(screenshot_bytes):
                episode_dir = os.path.join(images_dir, str(episode_id[0]))
                os.makedirs(episode_dir, exist_ok=True)
                screenshot_path = os.path.join(episode_dir, f'screenshot_{episode_id[0]}_{img_idx}.png')
                fd_screen = os.open(screenshot_path, os.O_WRONLY | os.O_CREAT, 0o600) 
                with os.fdopen(fd_screen, 'wb') as f:
                    f.write(screenshot_byte)
                screenshot_paths.append(screenshot_path)
            data = {
                'index': index,
                'screenshot_path': screenshot_paths,
                'accessibility_tree': forests_list,
                'step_instructions': step_instructions,
                'episode_id': episode_id,
                'goal': goal,
                'screenshot_widths': screenshot_widths,
                'screenshot_heights': screenshot_heights,
                'actions': actions,
                'node_info': node_info_list,
            }
            for key, feature in example.features.feature.items():
                if key in data:
                    continue
                kind = feature.WhichOneof('kind')
                if kind == 'int64_list':
                    data[key] = list(feature.int64_list.value)
                elif kind == 'bytes_list':
                    data[key] = [v.decode('utf-8', errors='ignore') for v in feature.bytes_list.value]
                elif kind == 'float_list':
                    data[key] = list(feature.float_list.value)
                else:
                    data[key] = f"<Unsupported type: {kind}>"
            jsonl_file.write(json.dumps(data, ensure_ascii=False) + '\n')
            index += 1
            pbar.update(1)
        except tf.errors.OutOfRangeError:
            logger.info('parse tf format data end')
            break
        except Exception:
            logger.error('parse tf format data error !!!')
            #continue
    pbar.close()


############################################################
# convert data to eval format
############################################################

def transform_action_data(action_data):
    action = json.loads(action_data['action'])
    action_type = action['action_type']
    
    result = {}
    result['result_action_type'] = 2 #unused
    result['result_action_text'] = ""
    result['result_lift_yx'] = [-1.0,-1.0]
    result['result_touch_yx'] = [-1.0,-1.0]
    result['duration'] = None

    if action_type in ['click', 'long_press', 'scroll']:
        try:
            start_x = (action['x'] / action_data['screenshot_width'])
            start_y = (action['y'] / action_data['screenshot_height'])
        except:
            start_x = 0.5
            start_y = 0.5
        if action_type == 'click' or action_type == 'long_press':
            end_x, end_y = start_x, start_y
        elif action_type == 'scroll':
            # Android control has reversed scroll directions
            map_direction = {'left': 'right', 'right': 'left', 'up': 'down', 'down': 'up'}
            direction = map_direction[action['direction']]
            if direction == 'up':
                end_x, end_y = start_x, max(0, start_y - 0.3)
            elif direction == 'down':
                end_x, end_y = start_x, min(1, start_y + 0.3)
            elif direction == 'left':
                end_x, end_y = max(0, start_x - 0.3), start_y
            else:  # right
                end_x, end_y = min(1, start_x + 0.3), start_y
                
        result['result_action_type'] = 4 # dual point
        result['result_touch_yx'] = [start_y, start_x]  # [y,x]
        result['result_lift_yx'] = [end_y, end_x]
        
    elif action_type == 'input_text':
        # Type action
        result['result_action_type'] = 3 #type
        result['result_action_text'] = action['text']
        
    elif action_type == 'navigate_home':
        # Button press actions
        result['result_action_type'] = 6 #home
        
    elif action_type == 'navigate_back':
        result['result_action_type'] = 5 #back
    elif action_type == 'wait':
        result['result_action_type'] = 1 #wait
    elif action_type == 'long_press':
        result['result_action_type'] = 0 #long_press
        start_x = (action['x'] / action_data['screenshot_width'])
        start_y = (action['y'] / action_data['screenshot_height'])
        end_x = start_x
        end_y = start_y
        result['result_touch_yx'] = [start_y, start_x]
        result['result_lift_yx'] = [end_y, end_x]
    elif action_type == 'finish':
        result['result_action_type'] = 10 #finish
    else:
        return None

    return result

def turn_ui_trees(action_data):
    accessibility_tree = action_data['ui_trees']
    bounds_pattern = r'bounds_in_screen \{([^}]+)\}'
    bounds_matches = list(re.finditer(bounds_pattern, accessibility_tree))
    ui_trees = []
    for i, match in enumerate(bounds_matches, 1):
        bounds_content = match.group(1)
        bounds_dict = {}
        for item in bounds_content.strip().split('\n'):
            item = item.strip()
            if item:
                key, value = item.split(': ')
                bounds_dict[key] = int(value)
        y_top_left = bounds_dict.get('top',0)
        x_top_left = bounds_dict.get('left',0)
        y_bottom_right = bounds_dict.get('bottom',0)
        x_bottom_right = bounds_dict.get('right',0)
        height = y_bottom_right - y_top_left
        width=x_bottom_right - x_top_left
        y_top_left_norm = y_top_left / action_data['screenshot_height']
        x_top_left_norm = x_top_left / action_data['screenshot_width']
        height_norm = height / action_data['screenshot_height']
        width_norm = width / action_data['screenshot_width']
        ui_trees.append([y_top_left_norm, x_top_left_norm, height_norm, width_norm])
    return ui_trees

def transform_action_data_and_build_test_data(action_data):
    turn_action = transform_action_data(action_data)
    if turn_action is None:
        return None
        
    test_data = {'episode_id': action_data['episode_id'], 'step_id': action_data['step'], 'episode_length': action_data['episode_length']}
    test_data['image_width'] = action_data['screenshot_width']
    test_data['image_height'] = action_data['screenshot_height']
    test_data['image_path'] = action_data['screenshot_path']
    test_data['instruction'] = action_data['goal']
    test_data['result_action_type'] = turn_action['result_action_type']
    test_data['result_touch_yx'] = str(turn_action['result_touch_yx'])
    test_data['result_lift_yx'] = str(turn_action['result_lift_yx'])
    test_data['duration'] = turn_action['duration']
    test_data['result_action_text'] = str(turn_action['result_action_text'])
    test_data['ui_positions'] = str(turn_ui_trees(action_data)) 
    test_data['low_instruction'] = action_data['low_instruction']
    test_data['subset'] = "android_control"

    return test_data

def read_train_list(train_json_path):
    try:
        with open(train_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return set(data.get('train', []))
    except Exception as e:
        print(f"Error at: {str(e)}")
        return set()

def read_episodes_from_jsonl(jsonl_file_path, train_episodes, output_test_dir):
    episodes = {'in_train': [], 'not_in_train': []}
    
    os.makedirs(output_test_dir, exist_ok=True)
    
    total_lines = sum(1 for _ in open(jsonl_file_path, 'r', encoding='utf-8'))
    
    with open(jsonl_file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=total_lines, desc="saving data"):
            data = json.loads(line.strip())
            episode_id = data['episode_id'][0]
            
            episode_data = []
            goal = data['goal'][0]
            actions = data['actions']
            screenshot_widths = data['screenshot_widths']
            screenshot_heights = data['screenshot_heights']
            screenshot_path = data['screenshot_path']
            
            for index, action in enumerate(actions):
                action_data = {
                    'action': action,
                    'screenshot_width': screenshot_widths[index],
                    'screenshot_height': screenshot_heights[index],
                    'screenshot_path': screenshot_path[index],
                    'low_instruction': data['step_instructions'][index],
                    'goal': goal,
                    'episode_id': episode_id,
                    'step': index,
                    'episode_length': len(screenshot_path),
                    'ui_trees': data['accessibility_tree'][index]
                }
                test_data = transform_action_data_and_build_test_data(action_data)
                if test_data is not None:
                    episode_data.append(test_data)
            
            # add last finish
            action_data={
                'action': "{\"action_type\":\"finish\"}",
                'screenshot_width': screenshot_widths[-1],
                'screenshot_height': screenshot_heights[-1],
                'screenshot_path': screenshot_path[-1],
                'goal': goal,
                'episode_id': episode_id,
                'step': len(actions),
                'episode_length': len(screenshot_path),
                'ui_trees': data['accessibility_tree'][-1],
                'low_instruction': "finish the task"
            }
            finish_data =transform_action_data_and_build_test_data(action_data)
            episode_data.append(finish_data)
                                    
            if episode_id in train_episodes:
                episodes['in_train'].append(episode_id)
                continue
            else:
                episodes['not_in_train'].append(episode_id)
            
            episode_dir = os.path.join(output_test_dir, str(episode_id))
            os.makedirs(episode_dir, exist_ok=True)
            output_file = os.path.join(episode_dir, f"{episode_id}.json")

            fd_output = os.open(output_file, os.O_WRONLY | os.O_CREAT, 0o600) 

            with os.fdopen(fd_output, 'w', encoding='utf-8') as f:
                json.dump(episode_data, f, ensure_ascii=False)
                
    return episodes

train_json_path = os.path.join(output_dir, 'splits.json')
jsonl_file_path = os.path.join(output_dir, 'data.jsonl')
# output_test_dir = os.path.join(os.path.join(current_dir, 'android_control_test_new/test/android_control'))
output_test_dir = os.path.join(os.path.join(current_dir, 'android_control_test/test/android_control'))

train_episodes = read_train_list(train_json_path)
episodes = read_episodes_from_jsonl(jsonl_file_path, train_episodes, output_test_dir)

print(f"Number of test episodes: {len(episodes['not_in_train'])}")
print(f"Save test set at: {output_test_dir}")

current_dir = os.path.dirname(os.path.abspath(__file__))

dataset_dir = os.path.join(current_dir, 'android_control_test', 'test', 'android_control')

low_level_data = []
high_level_data = []
sample_id = 0

for folder_name in tqdm(sorted(os.listdir(dataset_dir))):
    json_file_path = os.path.join(dataset_dir, folder_name, f"{folder_name}.json")
    with open(json_file_path, 'r') as f:
        sample_list = json.load(f)
    
    low_history_instruction = ''
    for sample in sample_list:

        image_path = sample['image_path']
        low_level_data.append({
            "id": sample_id,
            "query": f"\"{sample['low_instruction']}\"\nAction History: \"{low_history_instruction}\"",
            "images": [image_path]
        })
        


        high_level_data.append({
            "id": sample_id,
            "query": f"\"{sample['instruction']}\"\nAction History: \"{low_history_instruction}\"",
            "images": [image_path]
        })
        low_history_instruction += f"{sample['low_instruction']} \n"
        sample_id += 1

def process_samples(data_list, output_root_dir, dataset_name):

    target_dir = os.path.join(output_root_dir, dataset_name)
    images_dir = os.path.join(target_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    output_jsonl_path = os.path.join(target_dir, f"{dataset_name}.jsonl")

    fd_output = os.open(output_jsonl_path, os.O_WRONLY | os.O_CREAT, 0o600) 

    with os.fdopen(fd_output, 'w', encoding='utf-8') as f:
        for item in tqdm(data_list, desc=f"Processing：{dataset_name}"):
            image_filenames = [os.path.basename(p) for p in item['images']]
            new_item = {
                "id": item["id"],
                "query": item["query"],
                "images": image_filenames
            }
            f.write(json.dumps(new_item, ensure_ascii=False) + '\n')

            for original_image_path in item['images']:
                dst_path = os.path.join(images_dir, os.path.basename(original_image_path))
                shutil.copy2(original_image_path, dst_path)

    print(f"Saved at:{target_dir}")

output_base_dir = os.path.dirname(current_dir)
datasets = {
    "AC-Low": low_level_data,
    "AC-High": high_level_data
}

for dataset_name, dataset_content in datasets.items():
    process_samples(dataset_content, output_base_dir, dataset_name)