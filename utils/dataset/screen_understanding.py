import pandas as pd
import numpy as np
import json
import re
import os
import os.path as osp
import logging
from tqdm import tqdm
from functools import partial
from abc import abstractmethod
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from ..smp import *
from .utils.tool import *
from .utils.configs import *
from .utils.vqa_eval import anls_compute


LOG_FORMAT = "%(asctime)s -【%(levelname)s】%(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class ScreenUnderstandingDataset:
    MODALITY = 'IMAGE'

    # HF_ROOT = '/opt/nas/n/mm/ie_env/zhangshenghui/local_data_nas/hf_dataset'
    # LOCAL_ROOT = '/opt/nas/n/mm/ie_env/zhangshenghui/local_data_nas/test_data/images'

    DATASET_CONFIGS = {}

    EVALUATE_METHOD = 'RULE'

    BBOX_PATTERN = re.compile(r'<\|box_start\|>\((.*?),(.*?)\),\((.*?),(.*?)\)<\|box_end\|>')
    BBOX_PATTERN2 = re.compile(r'\((.*?),(.*?)\),\((.*?),(.*?)\)')

    REASON_PATTERN = None
    SCORE_PATTERN = None

    def __init__(self, dataset, dataset_path, model_name=None, model_type=None):
        self.dataset_name = dataset
        self.model_name = model_name
        self.model_type = model_type
        self.processed_path = os.path.join(dataset_path,f'{dataset}.jsonl')

        if not self.rawdata_processed():
            self.preprocess_rawdata()

        self.data = self.load_processed_data()
        # self.data = self.data[:10]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        return item
    
    def dump_image(self, line):
        if 'image' in line:
            tgt_path = [line['image']]
        elif 'images' in line:
            tgt_path = line['images']
        else:
            logger.error(f"image path extract error: {line}")
            raise Exception
        
        return tgt_path
    
    def build_prompt(self, line):
        prompt = self.build_prompt_text(line)

        tgt_path = self.dump_image(line)
        msgs = []
        if isinstance(tgt_path, list):
            msgs.extend([dict(type='image', value=p) for p in tgt_path])
        else:
            msgs = [dict(type='image', value=tgt_path)]
        msgs.append(dict(type='text', value=prompt))
        if msgs[-1]['type'] != 'text':
            raise ValueError(f"Error msg type {msgs[-1]['type']} not match 'text' ")
        return msgs
    
    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_CONFIGS)

    def rawdata_processed(self):
        if osp.exists(self.processed_path):
            return True
        return False
    
    def preprocess_line(self, line):
        if self.dataset_name == 'ScreenSpot':
            new_line, image = self.preprocess_line_screenspot(line)
             
        elif self.dataset_name == 'RICO-ScreenQA':
            new_line, image = self.preprocess_line_screenqa(line)

        return new_line, image

    def preprocess_line_screenqa(self, line):
        new_line = {}
        image = line['image']
        new_line['image'] = osp.join(self.images_path, line['file_name'].replace('/', '_'))
        new_line['query'] = 'Please answer the question based on the provided screenshot, if unable to answer based on the screenshot, output <no answer>. question: ' + line['question']

        new_line['response'] = line['ground_truth'][0]['full_answer']

        return new_line, image

    def preprocess_line_screenspot(self, line):
        new_line = {}
        image = line['image'].convert("RGB")
        new_line['image'] = osp.join(self.images_path, line['file_name'].replace('.png', '.jpg'))
        new_line['query'] = '根据英语短语的指令要求，找到屏幕截图中能够完成指令要求的控件位置，返回对应的bbox框，格式如<|box_start|>(669, 515),(902, 538)<|box_end|>。当前指令为：' + line['instruction']

        gt_box = [float(v*1000) for v in line['bbox']]
        new_line['response'] = f"<|box_start|>({gt_box[0]},{gt_box[1]}),({gt_box[2]},{gt_box[3]})<|box_end|>"

        new_line['data_type'] = line['data_type']
        new_line['data_source'] = line['data_source']

        if line['data_source'] not in ['ios', 'android']:
            new_line = None

        return new_line, image
    
    def build_prompt_text(self, line):
        query = line['query']
        try:
            query = query.replace('<image>', '').replace('<ref>', '')
        except:
            print(query)
            query = ''
        try:
            if 'prompt_template' in self.DATASET_CONFIGS[self.dataset_name] \
                and ('need_prompt_template' in self.DATASET_CONFIGS[self.dataset_name] \
                    or self.model_name in ['Qwen2-VL-7B-Instruct', 'InternVL2-8B', 'deepseek_vl2']):
                prompt_template = self.DATASET_CONFIGS[self.dataset_name]['prompt_template']
                query = prompt_template.format(query=query)
            return query
        except:
            prompt_template = self.DATASET_CONFIGS[self.dataset_name]['prompt_template']
            query = prompt_template.format(query=query)
            return query
    @abstractmethod
    def eval_line(self, line):
        pass

    def preprocess_rawdata(self):
        self.raw_path = osp.join(self.HF_ROOT, 'raw', self.dataset_name)
        self.images_path = osp.join(self.HF_ROOT, 'images', self.dataset_name)
        subset = None
        if 'subset' in self.DATASET_CONFIGS[self.dataset_name]:
            subset = self.DATASET_CONFIGS[self.dataset_name]['subset']
        if 'split' in self.DATASET_CONFIGS[self.dataset_name]:
            split = self.DATASET_CONFIGS[self.dataset_name]['split']
        else:
            split = 'test'
        raw_dataset = load_dataset(self.raw_path, subset, split=split)

        if not osp.exists(self.images_path):
            os.makedirs(self.images_path, exist_ok=True)

        logger.info('Start to process rawdata')

        processed_dataset = []
        index = 0
        for i, line in tqdm(enumerate(raw_dataset)):
            new_line, img = self.preprocess_line(line)
            if not new_line:
                continue
            new_line['id'] = str(index)
            index += 1
            image_file_path = osp.join(self.images_path, new_line['image'])
            img.save(image_file_path)
            processed_dataset.append(new_line)

        dump_jsonl(processed_dataset, self.processed_path)

    def load_processed_data(self):
        data_jsonl = load_jsonl(self.processed_path)
        base_dir = os.path.join(os.path.dirname(self.processed_path),'images')
        for i in range(len(data_jsonl)):
            line = data_jsonl[i]
            if 'image' in line:
                tgt_path = [os.path.join(base_dir, line['image'])]
            elif 'images' in line:
                tgt_path = [os.path.join(base_dir, img) for img in line['images']]

            else:
                logger.error(f"image path extract error: {line}")
                raise Exception
        
            for i in range(len(tgt_path)):
                img_path = tgt_path[i]

                if not img_path.startswith('/') and self.LOCAL_ROOT:
                    img_path = osp.join(self.LOCAL_ROOT, img_path)
                    tgt_path[i] = img_path

                if not osp.exists(img_path):
                    img_path = img_path.replace('.jpg', '_screen.jpg').replace('.png', '_screen.png')
                    if not osp.exists(img_path):
                        logger.error(f"file {img_path} not exist")
                        raise Exception
                    tgt_path[i] = img_path

            if 'image' in line:
                line['image'] = tgt_path[0]
            elif 'images' in line:
                line['images'] = tgt_path

        df = pd.DataFrame(data_jsonl)
        df['index'] = df.index

        if 'self_build' in self.processed_path or 'android-control' in self.processed_path or 'GUI_Odyssey' in self.processed_path:
            pass
        else:
            pass

        return df

    def evaluate_by_rule(self, data, eval_file):

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        res = []
        for line in lines:
            tmp = self.eval_line(line)
            tmp['index'] = line['index']
            res.append(tmp)

        res_df = pd.DataFrame(res)
        data_after_merge = pd.merge(data, res_df, on='index', how='left', validate='1:1')
        data_after_merge.drop(columns='index', inplace=True) 

        eval_output_file = eval_file.replace('.xlsx', '_rule_eval_results.xlsx')

        grounding_flag = True if self.TYPE == 'GROUNDING' else False
        write_excel(data_after_merge, eval_output_file, grounding=grounding_flag, pattern1=self.BBOX_PATTERN, pattern2=self.BBOX_PATTERN2)

        return res
    
    def build_eval_llm(self,eval_model_path):
        model_name = eval_model_path

        rank, world_size = get_rank_and_world_size()

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=f"cuda:{rank}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.REASON_PATTERN = re.compile(r'{"reason": "(.*)", "score":')
        self.SCORE_PATTERN = re.compile(r'"score": ([0-9])}')

    def llm_eval(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response
    
    def aggregate_results(self, eval_file, infer_dataset):
        _, world_size = get_rank_and_world_size()
        tmp_filename_template = 'tmpfile_rank_{rank}_'+osp.basename(eval_file).replace('.xlsx', '.jsonl')
        aggregated_results = [] 
        for rank in range(world_size):
            tmp_filename = tmp_filename_template.format(rank=rank)
            tmp_file_path = osp.join(osp.dirname(eval_file),tmp_filename)
            if not osp.exists(tmp_file_path):
                logger.error(f'tmp llm eval file not exists: {tmp_file_path}')
                raise Exception
            data_jsonl = load_jsonl(tmp_file_path)
            aggregated_results.extend(data_jsonl)
            os.remove(tmp_file_path)
            
        df = pd.DataFrame(aggregated_results)
        
        data_after_merge = pd.merge(infer_dataset, df, on='index', how='left', validate='1:1')
        data_after_merge.drop(columns='index', inplace=True) 

        eval_output_file = eval_file.replace('.xlsx', '_llm_eval_results.xlsx')

        grounding_flag = True if self.TYPE == 'GROUNDING' else False
        write_excel(data_after_merge, eval_output_file, grounding=grounding_flag, pattern1=self.BBOX_PATTERN, pattern2=self.BBOX_PATTERN2)

        return aggregated_results


    def evaluate_by_llm(self, infer_dataset, eval_file,eval_model_path=None):
        rank, world_size = get_rank_and_world_size()
        sheet_indices = list(range(rank, len(infer_dataset), world_size))
        lt = len(sheet_indices)
        data = infer_dataset.iloc[sheet_indices]
        
        self.build_eval_llm(eval_model_path)
        
        tmp_filename = f'tmpfile_rank_{rank}_'+osp.basename(eval_file).replace('.xlsx', '.jsonl')
        tmp_file_path = osp.join(osp.dirname(eval_file),tmp_filename)
        fd = os.open(tmp_file_path, os.O_WRONLY | os.O_CREAT, 0o600) 
        with os.fdopen(fd, 'a') as f:
            for i in tqdm(range(lt)):
                line = data.iloc[i]
                res = self.eval_line(line)
                res['index'] = int(line['index'])
                item_line = json.dumps(res, ensure_ascii=False) + '\n'
                f.write(item_line)
                f.flush()

        if world_size > 1:
            dist.barrier()

        if rank == 0:
            aggregated_result = self.aggregate_results(eval_file, infer_dataset)
            
            return aggregated_result
        else:
            return None

    def hit_calculate(self, res, dataset):
        return [x['match'] for x in res]
    
    def evaluate(self, eval_file, **judge_kwargs):
        infer_dataset = load(eval_file)

        rank, world_size = get_rank_and_world_size()
        if self.EVALUATE_METHOD == 'RULE':
            if rank == 0:
                res = self.evaluate_by_rule(infer_dataset, eval_file)

        elif self.EVALUATE_METHOD == 'LLM':
            eval_model_path = judge_kwargs['eval_model_path']
            res = self.evaluate_by_llm(infer_dataset, eval_file, eval_model_path)

        else:
            raise Exception
        
        self.calculate_results(eval_file, infer_dataset, res)
        return f"{eval_file[:-5]}_acc.csv"

    def calculate_results(self, eval_file, infer_dataset, res):
        lines = [infer_dataset.iloc[i] for i in range(len(infer_dataset))]
        hit = self.hit_calculate(res, self.dataset_name)
        ret = dict()
        if 'split' in infer_dataset:
            splits = set(infer_dataset['split'])
            for sp in splits:
                sub = [r for l, r in zip(lines, res) if l['split'] == sp]
                hit = self.hit_calculate(sub, self.dataset_name)
                ret[sp] = np.mean(hit) * 100
            sub = [r for l, r in zip(lines, res)]
            hit = self.hit_calculate(sub, self.dataset_name)
            ret['Overall'] = np.mean(hit) * 100
        else:
            ret['Overall'] = np.mean(hit) * 100
            if 'category' in infer_dataset:
                cates = list(set(infer_dataset['category']))
                cates.sort()
                for c in cates:
                    sub = [r for l, r in zip(lines, res) if l['category'] == c]
                    hit = self.hit_calculate(sub, self.dataset_name)
                    ret[c] = np.mean(hit) * 100
        ret = d2df(ret)
        ret.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(ret, result_file)
        return result_file
    
    def change_line_format(self, input, pattern1, pattern2, new_format):
        for pattern in [pattern1, pattern2]:
            bboxes = re.findall(pattern, input)
            if bboxes:
                if type(bboxes[0]) is str:
                    bbox_str = bboxes[0]
                else:
                    bbox_str = ','.join(bboxes[0])
                new_str = new_format.format(bbox_str=bbox_str)
                input = re.sub(pattern, new_str, input)
        return input

    def change_dataset_format(self, model_type):
        if model_type not in ['InternVLChat', 'DeepSeekVL2']:
            return
        
        new_format = None
        query_prefix = None
        if model_type == 'InternVLChat':
            new_format = '<box>[[{bbox_str}]]</box>'
        elif model_type == 'DeepSeekVL2':
            new_format = '<|det|>[[{bbox_str}]]<|/det|>'
            if self.TYPE == 'GROUNDING':
                query_prefix = '<|grounding|>'

        TMP_PATTERN = re.compile(r'<\|box_start\|>(.*?)<\|box_end\|>')
        for index in range(len(self.data)):
            line = self.data.iloc[index]
            line['query'] = self.change_line_format(line['query'], self.BBOX_PATTERN, TMP_PATTERN, new_format)
            if query_prefix:
                line['query'] = query_prefix + line['query']
            line['response'] = self.change_line_format(line['response'], self.BBOX_PATTERN, TMP_PATTERN, new_format)
            self.data.iloc[index] = line

        if model_type == 'InternVLChat':
            self.BBOX_PATTERN = re.compile(r'<box>\[\[(.*?),(.*?),(.*?),(.*?)\]\]</box>')
            self.BBOX_PATTERN2 = re.compile(r'\[\[(.*?),(.*?),(.*?),(.*?)\]\]')
        elif model_type == 'DeepSeekVL2':
            self.BBOX_PATTERN = re.compile(r'<\|det\|>\[\[(.*?),(.*?),(.*?),(.*?)\]\]<\|/det\|>')
            self.BBOX_PATTERN2 = re.compile(r'\[\[(.*?),(.*?),(.*?),(.*?)\]\]')
    

class ReferringDataset(ScreenUnderstandingDataset):
    TYPE = 'REFERRING'
    DATASET_CONFIGS = REFERRING_CONFIGS
    
    JSON_PATTERN = re.compile(r'```json(.*?)```')

    DELETE_PATTERN = r' |>|<|\*|\n|:|：|\.'

    def eval_item(self, infer_item, gt_item, entity_attribute):
        if entity_attribute in ['内部文本', '外部邻近文本', '名称', '注释', '当前值']:
            infer_item = re.sub(self.DELETE_PATTERN, '', infer_item).replace('(', '（').replace(')', '）').replace(',', '，')
            gt_item = re.sub(self.DELETE_PATTERN, '', gt_item).replace('(', '（').replace(')', '）').replace(',', '，').strip('-')
        return infer_item == gt_item

    def eval_entity(self, infer_result, ground_truth, entity_type):

        if '类型' in ground_truth:
            entity_type = ground_truth['类型']
        elif not entity_type:
            logger.error(f"empty entity_type error: {ground_truth}")
            raise Exception
        
        attribute_matrix = {'correct_num': 0, 'infer_num': 0, 'gt_num': 0}
        for entity_attribute in ground_truth.keys():
            if entity_attribute not in infer_result:
                infer_flag = 0
                score = 0
            else:
                infer_flag = 1
                try:
                    score = self.eval_item(infer_result[entity_attribute], ground_truth[entity_attribute], entity_attribute)
                except Exception:
                    logger.error(f'eval_item failed! 【{infer_result[entity_attribute]}】') 
                    score = 0

            attribute_matrix['correct_num'] += score
            attribute_matrix['infer_num'] += infer_flag
            attribute_matrix['gt_num'] += 1

        attribute_matrix['entity_type'] = entity_type

        return attribute_matrix
    
    def eval_line(self, line):
        gt_str = re.findall(self.JSON_PATTERN, line['response'].replace(' ', '').replace('\n', ''))[0]
        ground_truth = eval(gt_str)

        try:  
            pred_str = re.findall(self.JSON_PATTERN, line['prediction'].replace(' ', '').replace('\n', ''))[0]
            infer_result = eval(pred_str)
            
        except Exception:
            logger.error(f"load json error: {line['prediction']}") 
            pred_str = line['prediction']
            infer_result = {}
        
        if '类型' in ground_truth:
            entity_type = ground_truth['类型']
        else:
            entity_type = 'others'
        ret = self.eval_entity(infer_result, ground_truth, entity_type)

        ret['question'] = self.build_prompt_text(line)
        ret['pred'] = pred_str
        ret['gt'] = gt_str

        return ret

    def calculate_results(self, eval_file, infer_dataset, res):
        ret = {'Overall': {'correct_num': 0, 'infer_num': 0, 'gt_num': 0}}
        outputs = {}
        lines = [infer_dataset.iloc[i] for i in range(len(infer_dataset))]
        for line in res:
            if line['entity_type'] not in ret:
                ret[line['entity_type']] = {'correct_num': 0, 'infer_num': 0, 'gt_num': 0}
            ret[line['entity_type']]['correct_num'] += line['correct_num']
            ret[line['entity_type']]['infer_num'] += line['infer_num']
            ret[line['entity_type']]['gt_num'] += line['gt_num']

            ret['Overall']['correct_num'] += line['correct_num']
            ret['Overall']['infer_num'] += line['infer_num']
            ret['Overall']['gt_num'] += line['gt_num']

        for entity_type in ret.keys():
            correct_num_total = ret[entity_type]['correct_num']
            correct_infer_total = ret[entity_type]['infer_num']
            correct_gt_total = ret[entity_type]['gt_num']
            precision = correct_num_total / (correct_infer_total+(1e-5))
            recall = correct_num_total / correct_gt_total
            f1 = 2 * precision * recall / (precision + recall+(1e-5))
            ret[entity_type]['f1'] = f1
            outputs[entity_type] = f1

        outputs = d2df(outputs)
        outputs.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(outputs, result_file)
        return outputs

class GroundingDataset(ScreenUnderstandingDataset):
    TYPE = 'GROUNDING'
    DATASET_CONFIGS = GROUNDING_CONFIGS
    MAX_DISTANCE = 100
    DATA_FORMAT = 'BBOX'

    def __init__(self, dataset, dataset_path, model_name=None, model_type=None):
        super().__init__(dataset, dataset_path, model_name, model_type)

        if 'data_format' in self.DATASET_CONFIGS[self.dataset_name]:
            if self.DATASET_CONFIGS[self.dataset_name]['data_format'] == 'POINT':
                self.DATA_FORMAT = 'POINT'
            else:
                self.DATA_FORMAT = 'BBOX'

    def eval_line(self, line):
        if self.DATA_FORMAT == 'POINT':
            return self.eval_line_point(line)
        else:
            return self.eval_line_bbox(line)
        
    def eval_line_point(self, line):
        ret = {}

        gt_point = extract_point(line['response'])
        gt_boxes = extract_bboxs(line['response'], self.BBOX_PATTERN)
        if not gt_point and not gt_boxes:
            logger.error(f'extract_point and extract_bboxs from response failed! 【{line}】') 
            raise Exception
        
        pred_point = extract_point(line['prediction'])
        if pred_point:
            if gt_boxes:
                gt_box = gt_boxes[0]
                if point_isin_bbox(pred_point, gt_box):
                    ret['point_in_bbox'] = 1
                    ret['match'] = 1
                else:
                    ret['point_in_bbox'] = 0
                    ret['match'] = 0
            else:
                distance = points_distance(gt_point, pred_point, line['images'])
                ret['center_points_distance'] = distance
                if distance > self.MAX_DISTANCE:
                    ret['match'] = 0
                else:
                    ret['match'] = 1
        else:
            ret['match'] = 0

        ret['question'] = self.build_prompt_text(line)
        ret['gt'] = line['response']
        ret['pred'] = line['prediction']

        return ret

    def eval_line_bbox(self, line):
        ret = {}
        
        gt_boxes = extract_bboxs(line['response'], self.BBOX_PATTERN)
        pred_boxes = extract_bboxs(line['prediction'], self.BBOX_PATTERN, self.BBOX_PATTERN2)

        if pred_boxes and gt_boxes:
            pred_box = pred_boxes[0]
            gt_box = gt_boxes[0]
            pred_center_point = ((pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2)

            if point_isin_bbox(pred_center_point, gt_box):
                ret['center_point_in_bbox'] = 1
            else:
                ret['center_point_in_bbox'] = 0
            
            iou = bbox_iou(pred_box, gt_box)
            ret['bbox_iou'] = iou
        else:
            ret['center_point_in_bbox'] = None
            ret['bbox_iou'] = None
            if not gt_boxes:
                ret['excluded'] = True

        ret['question'] = self.build_prompt_text(line)
        ret['pred'] = pred_boxes
        ret['gt'] = gt_boxes

        return ret
    
    def calculate_results(self, eval_file, infer_dataset, res):
        if self.DATA_FORMAT == 'POINT':
            return self.calculate_results_point(eval_file, infer_dataset, res)
        else:
            return self.calculate_results_bbox(eval_file, infer_dataset, res) 

    def calculate_results_point(self, eval_file, infer_dataset, res):
        ret = {}
        res_df = pd.DataFrame(res)

        if 'type' in infer_dataset:
            res_df['type'] = infer_dataset['type']

        ret['Overall'] = res_df['match'].fillna(0).mean()

        if 'type' in res_df:
            data_types = set()
            for i in set(res_df['type'].unique()):
                for t in i.split('&'):
                    data_types.add(t)
            data_types = list(data_types)
            data_types.sort()
            for data_type in data_types:
                res_df_filtered = res_df[res_df['type'].str.contains(data_type)]
                ret[f"{data_type}({len(res_df_filtered)})"] = res_df_filtered['match'].mean()

        outputs = d2df(ret)
        outputs.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(outputs, result_file)
        return outputs
    
    def calculate_results_bbox(self, eval_file, infer_dataset, res):
        ret = {}
        res_df = pd.DataFrame(res)

        if 'type' in infer_dataset:
            res_df['type'] = infer_dataset['type']

        if 'excluded' in res_df:
            res_df = res_df[res_df['excluded'] != True]
        
        res_df['bbox_iou'] = res_df['bbox_iou'].fillna(0)
        res_df['center_point_in_bbox'] = res_df['center_point_in_bbox'].fillna(0)

        ret['center_point_in_bbox'] = res_df['center_point_in_bbox'].mean()
        ret['bbox_iou@0.5'] = (res_df['bbox_iou']>=0.5).mean()
        ret['bbox_iou@0.1'] = (res_df['bbox_iou']>=0.1).mean()


        if 'type' in res_df:
            data_types = set()
            for i in set(res_df['type'].unique()):
                for t in i.split('&'):
                    data_types.add(t)
            data_types = list(data_types)
            data_types.sort()
            for data_type in data_types:
                res_df_filtered = res_df[res_df['type'].str.contains(data_type)]
                ret[f"{data_type}({len(res_df_filtered)})"] = res_df_filtered['center_point_in_bbox'].mean()

        outputs = d2df(ret)
        outputs.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(outputs, result_file)
        return outputs

class NavigationDataset(ScreenUnderstandingDataset):
    TYPE = 'NAVIGATION'
    DATASET_CONFIGS = NAVIGATION_CONFIGS

    FUNC_CALL_CONFIGS = {
        'FORMAT_GT': {
            'tap': {'pattern': r'tap\(<\|box_start\|>\((\d+\.?\d*),(\d+\.?\d*)\),\((\d+\.?\d*),(\d+\.?\d*)\)<\|box_end\|>\)', 'agrs_nums': 4}, 
            'long_press': {'pattern': r'long_press\(<\|box_start\|>\((\d+\.?\d*),(\d+\.?\d*)\),\((\d+\.?\d*),(\d+\.?\d*)\)<\|box_end\|>\)', 'agrs_nums': 4}, 
            'text': {'pattern': r'text\(<\|box_start\|>\((\d+\.?\d*),(\d+\.?\d*)\),\((\d+\.?\d*),(\d+\.?\d*)\)<\|box_end\|>,(.*)\)', 'agrs_nums': 5}, 
            'scroll': {'pattern': r'scroll\(<\|box_start\|>\((\d+\.?\d*),(\d+\.?\d*)\),\((\d+\.?\d*),(\d+\.?\d*)\)<\|box_end\|>,(.*)\)', 'agrs_nums': 5}, 
            
            'drag': {'pattern': r'drag\(<\|box_start\|>\((\d+\.?\d*),(\d+\.?\d*)\),\((\d+\.?\d*),(\d+\.?\d*)\)<\|box_end\|>,<\|box_start\|>\((\d+\.?\d*),(\d+\.?\d*)\),\((\d+\.?\d*),(\d+\.?\d*)\)<\|box_end\|>\)', 'agrs_nums': 8}, 

            'call_api': {'pattern': r'call_api\((.*),(.*)\)', 'agrs_nums': 2},

            'take_over': {'pattern': r'take_over\((.*)\)', 'agrs_nums': 1},
            'no_answer': {'pattern': r'no_answer\((.*)\)', 'agrs_nums': 1},
            'wait': {'pattern': r'wait\((.*)\)', 'agrs_nums': 1},
            'navigate_back': {'pattern': r'navigate_back\((.*)\)', 'agrs_nums': 1},
            'navigate_home': {'pattern': r'navigate_home\((.*)\)', 'agrs_nums': 1},
            'enter': {'pattern': r'enter\((.*)\)', 'agrs_nums': 1},
            'screen_shot': {'pattern': r'screen_shot\((.*)\)', 'agrs_nums': 1},
            'long_screen_shot': {'pattern': r'long_screen_shot\((.*)\)', 'agrs_nums': 1},
            'action_completed': {'pattern': r'action_completed\((.*)\)', 'agrs_nums': 1},
        },
        'FORMAT_pre':{
            'tap': {'pattern': r'tap\(\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*\)', 'agrs_nums': 2}, 
            'long_press': {'pattern': r'long_press\(\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*\)', 'agrs_nums': 2}, 
            'text': {'pattern': r'text\(\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*,(.*)\)', 'agrs_nums': 3}, 
            'scroll': {'pattern': r'scroll\(\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*,(.*)\)', 'agrs_nums': 3}, 
            
            'drag': {'pattern': r'drag\(\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*\)', 'agrs_nums': 4}, 

            'call_api': {'pattern': r'call_api\((.*),(.*)\)', 'agrs_nums': 2},

            'take_over': {'pattern': r'take_over\((.*)\)', 'agrs_nums': 1},
            'no_answer': {'pattern': r'no_answer\((.*)\)', 'agrs_nums': 1},
            'wait': {'pattern': r'wait\((.*)\)', 'agrs_nums': 1},
            'navigate_back': {'pattern': r'navigate_back\((.*)\)', 'agrs_nums': 1},
            'navigate_home': {'pattern': r'navigate_home\((.*)\)', 'agrs_nums': 1},
            'enter': {'pattern': r'enter\((.*)\)', 'agrs_nums': 1},
            'screen_shot': {'pattern': r'screen_shot\((.*)\)', 'agrs_nums': 1},
            'long_screen_shot': {'pattern': r'long_screen_shot\((.*)\)', 'agrs_nums': 1},
            'action_completed': {'pattern': r'action_completed\((.*)\)', 'agrs_nums': 1},

            'click': {'pattern': r'click\(\D*(\d+\.?\d*)\D*,\D*(\d+\.?\d*)\D*\)', 'agrs_nums': 2}, 
            'type': {'pattern': r'type\(content=\'(.*)\'\)', 'agrs_nums': 1}, 

        },
    }

    MAX_DISTANCE = 140
    MAX_ANLS = 0.2

    def __init__(self, dataset, dataset_path, model_name=None, model_type=None):
        super().__init__(dataset, dataset_path, model_name, model_type)
        self.DATA_FORMAT = 'BBOX'
        if 'data_format' in self.DATASET_CONFIGS[self.dataset_name]:
            if self.DATASET_CONFIGS[self.dataset_name]['data_format'] == 'POINT':
                self.DATA_FORMAT = 'POINT'
                
    def eval_line(self, line):
        if self.DATA_FORMAT == 'POINT':
            return self.eval_line_point(line)
        else:
            return self.eval_line_bbox(line)

    def eval_line_point(self, line):
        ret = {}
        ret['question'] = self.build_prompt_text(line)


        '''
        tap(x: float,y: float)                          click   校验动作    校验坐标
        scroll(x: float,y: float,direction: str)                校验动作    校验坐标    校验字符串
        text(x: float,y: float,text_input: str)                 校验动作    校验坐标    校验字符串
        long_press(x: float,y: float)                           校验动作    校验坐标
        drag(x1: float,y1: float,x2: float,y2: float)           校验动作    校验坐标
        call_api(api_name: str,params: str)                     校验动作                校验字符串

        take_over(text_input: str)                              校验动作
        navigate_back()                                         校验动作
        navigate_home()                                         校验动作
        wait()                                                  校验动作
        enter()                                                 校验动作
        screen_shot()                                           校验动作
        long_screen_shot()                                      校验动作

        no_answer(text_input: str)                              校验动作
        action_completed()                                      校验动作
        '''

        action_gt,args_gt = extract_action(line['response'], self.FUNC_CALL_CONFIGS['FORMAT_pre'])
        action_pred, args_pred = extract_action(line['prediction'], self.FUNC_CALL_CONFIGS['FORMAT_pre'])

        ret['pred'] = f"{action_pred}, {args_pred}"
        ret['gt'] = f"{action_gt}, {args_gt}"

        ret['match'] = 1
        # print(args_gt,args_pred)
        if action_gt == action_pred:
            ret['action_match'] = 1
        else:
            ret['action_match'] = 0
            ret['match'] = 0


        if len(args_gt) == 1 and len(args_pred) == 1:
            args_gt = args_gt[0]
            args_pred = args_pred[0]

            if action_gt in ['tap','long_press','text','scroll'] and action_pred in ['tap','long_press','text','scroll']:
                gt_point = (int(float(args_gt[0])), int(float(args_gt[1])))
                if gt_point != (0,0) and len(args_pred) >= 2:
                    point = (int(float(args_pred[0]))), int(float(args_pred[1]))
                    if point_match(gt_point, point, self.MAX_DISTANCE):
                        ret['center_point_in_bbox'] = 1
                    else:
                        ret['center_point_in_bbox'] = 0
                        ret['match'] = 0

            if action_gt == 'drag' and action_pred == 'drag':
                point_1 = (int(float(args_pred[0]))), int(float(args_pred[1]))
                point_2 = (int(float(args_pred[2]))), int(float(args_pred[3]))
                gt_point_1 = (int(float(args_gt[0])), int(float(args_gt[1])))
                gt_point_2 = (int(float(args_gt[2])), int(float(args_gt[3])))

                if point_match(point_1, gt_point_1, self.MAX_DISTANCE) and point_match(point_2, gt_point_2, self.MAX_DISTANCE):
                    ret['center_point_in_bbox'] = 1
                else:
                    ret['center_point_in_bbox'] = 0
                    ret['match'] = 0

            if action_gt in ['text','scroll'] and action_pred in ['text','scroll']:
                anls_distance = anls_compute(args_pred[-1], args_gt[-1].replace('"',''))
                # print(anls_distance)
                if anls_distance > self.MAX_ANLS:
                    ret['text_match'] = 0
                    ret['match'] = 0
                else:
                    ret['text_match'] = 1
            
            if action_gt == 'call_api' and action_pred == 'call_api':
                app_gt, operation_gt = args_gt
                app_pre, operation_pre = args_pred

                app_anls_distance = anls_compute(app_pre.replace('"','').replace('\'',''), app_gt.replace('"',''))
                operation_anls_distance = anls_compute(operation_pre.replace('"','').replace('\'',''), operation_gt.replace('"',''))

                # print(anls_distance)
                if app_anls_distance > self.MAX_ANLS or operation_anls_distance:
                    ret['text_match'] = 0
                    ret['match'] = 0
                else:
                    ret['text_match'] = 1
        else:
            ret['match'] = 0

        return ret

    def eval_line_bbox(self, line):
        ret = {}
        ret['question'] = self.build_prompt_text(line)


        '''
        tap(x: float,y: float)                          click   校验动作    校验坐标
        scroll(x: float,y: float,direction: str)                校验动作    校验坐标    校验字符串
        text(x: float,y: float,text_input: str)                 校验动作    校验坐标    校验字符串
        long_press(x: float,y: float)                           校验动作    校验坐标
        drag(x1: float,y1: float,x2: float,y2: float)           校验动作    校验坐标
        call_api(api_name: str,params: str)                     校验动作                校验字符串

        take_over(text_input: str)                              校验动作
        navigate_back()                                         校验动作
        navigate_home()                                         校验动作
        wait()                                                  校验动作
        enter()                                                 校验动作
        screen_shot()                                           校验动作
        long_screen_shot()                                      校验动作

        no_answer(text_input: str)                              校验动作
        action_completed()                                      校验动作
        '''

        action_gt,args_gt = extract_action(line['response'], self.FUNC_CALL_CONFIGS['FORMAT_GT'])
        action_pred, args_pred = extract_action(line['prediction'], self.FUNC_CALL_CONFIGS['FORMAT_pre'])

        ret['pred'] = f"{action_pred}, {args_pred}"
        ret['gt'] = f"{action_gt}, {args_gt}"

        ret['match'] = 1
        if action_gt == action_pred:
            ret['action_match'] = 1
        else:
            ret['action_match'] = 0
            ret['match'] = 0
        args_pred = list(set(args_pred))
        if len(args_gt) == 1 and len(args_pred) == 1:
            args_gt = args_gt[0]
            args_pred = args_pred[0]

            if action_gt in ['tap','long_press','text','scroll'] and action_pred in ['tap','long_press','text','scroll']:
                bbox = (int(float(args_gt[0])), int(float(args_gt[1])), int(float(args_gt[2])), int(float(args_gt[3])))
                if bbox != (0,0,0,0) and len(args_pred) >= 2:
                    point = (int(float(args_pred[0]))), int(float(args_pred[1]))
                    if point_isin_bbox(point, bbox):
                        ret['center_point_in_bbox'] = 1
                    else:
                        ret['center_point_in_bbox'] = 0
                        ret['match'] = 0

            if action_gt == 'drag' and action_pred == 'drag':
                point_1 = (int(float(args_pred[0]))), int(float(args_pred[1]))
                point_2 = (int(float(args_pred[2]))), int(float(args_pred[3]))
                bbox_1 = (int(float(args_gt[0])), int(float(args_gt[1])), int(float(args_gt[2])), int(float(args_gt[3])))
                bbox_2 = (int(float(args_gt[4])), int(float(args_gt[5])), int(float(args_gt[6])), int(float(args_gt[7])))
                if point_isin_bbox(point_1, bbox_1) and point_isin_bbox(point_2, bbox_2):
                    ret['center_point_in_bbox'] = 1
                else:
                    ret['center_point_in_bbox'] = 0
                    ret['match'] = 0

            if action_gt in ['text','scroll'] and action_pred in ['text','scroll']:
                anls_distance = anls_compute(args_pred[-1], args_gt[-1].replace('"',''))
                if anls_distance > self.MAX_ANLS:
                    ret['text_match'] = 0
                    ret['match'] = 0
                else:
                    ret['text_match'] = 1
            
            if action_gt == 'call_api' and action_pred == 'call_api':
                app_gt, operation_gt = args_gt
                app_pre, operation_pre = args_pred

                app_anls_distance = anls_compute(app_pre.replace('"','').replace('\'',''), app_gt.replace('"',''))
                operation_anls_distance = anls_compute(operation_pre.replace('"','').replace('\'',''), operation_gt.replace('"',''))

                if app_anls_distance > self.MAX_ANLS or operation_anls_distance:
                    ret['text_match'] = 0
                    ret['match'] = 0
                else:
                    ret['text_match'] = 1

        elif action_gt == 'tap' and action_pred == 'tap':
            bbox_match = []
            for b in args_gt:
                bbox = (int(float(b[0])), int(float(b[1])), int(float(b[2])), int(float(b[3])))
                for p in args_pred:

                    point = (int(float(p[0]))), int(float(p[1]))
                    if point_isin_bbox(point, bbox):
                        bbox_match.append(f"{bbox} match")
                        break
            
            acc = len(bbox_match)/len(args_pred)
            recall = len(bbox_match)/len(args_gt)
            if acc != 0 or recall != 0:
                f1 = 2 * acc * recall / (acc + recall)
            else:
                f1 = 0
            ret['center_point_in_bbox_acc'] = acc
            ret['center_point_in_bbox_recall'] = recall
            ret['center_point_in_bbox_f1'] = f1
            if f1 == 1:
                ret['center_point_in_bbox'] = 1
                ret['match'] = 1
            else:
                ret['center_point_in_bbox'] = 0
                ret['match'] = 0

        
        else:
            ret['match'] = 0

        return ret
    
    def calculate_results(self, eval_file, infer_dataset, res):
        ret = {}
        res_df = pd.DataFrame(res)

        if 'type' in infer_dataset:
            res_df['type'] = infer_dataset['type']
        if 'task_id' in infer_dataset:
            res_df['task_id'] = infer_dataset['task_id']
            task_num = 0
            task_correct_num = 0
            for task_id in res_df['task_id'].unique():
                task_num += 1
                if res_df[res_df['task_id']==task_id]['match'].fillna(0).mean() == 1:
                    task_correct_num += 1
            ret['Task_Overall'] = task_correct_num/task_num
            ret['Task'] = f"{task_correct_num}/{task_num}"



        ret['Overall'] = res_df['match'].fillna(0).mean()
        ret['action_match'] = res_df['action_match'].fillna(0).mean()
        if 'center_point_in_bbox' in res_df.columns:
            ret['grounding_match'] = res_df['center_point_in_bbox'].mean()
        if 'center_point_in_bbox_f1' in res_df.columns:
            ret['Overall_f1'] = res_df['center_point_in_bbox_f1'].fillna(res_df['center_point_in_bbox']).mean()
        if 'text_match' in res_df.columns:
            ret['text_match'] = res_df['text_match'].mean()

        if 'type' in res_df and res_df['type'].notnull().all():
            data_types = set()
            q_types = set()
            for i in set(res_df['type'].unique()):
                for t in i.split('&'):
                    if 'q_' in t or 'q1_' in t or 'q2_' in t:
                        q_types.add(t)
                    else:
                        data_types.add(t)
            data_types = list(data_types)
            data_types.sort()
            q_types = list(q_types)
            q_types.sort()
            for q_type in q_types:
                res_df_filtered = res_df[res_df['type'].str.contains(q_type)]
                ret['_'.join(f"{q_type}({len(res_df_filtered)})".split('_')[1:])] = res_df_filtered['match'].mean()
            
            res_df = res_df[res_df['type'].str.contains('直接指令')]
            for data_type in data_types:
                res_df_filtered = res_df[res_df['type'].str.contains(data_type)]
                ret['_'.join(f"{data_type}({len(res_df_filtered)})".split('_')[1:])] = res_df_filtered['match'].mean()

        outputs = d2df(ret)
        outputs.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(outputs, result_file)
        return result_file
    


class Old_NavigationDataset(ScreenUnderstandingDataset):
    TYPE = 'NAVIGATION'
    DATASET_CONFIGS = NAVIGATION_CONFIGS

    FUNC_CALL_CONFIGS = {
        'FORMAT1': {
            'tap': {'pattern': r'\((.*?),(.*?)\)', 'agrs_nums': 2},  
            'text': {'pattern': r'\((.*?),(.*?),(.*?)\)', 'agrs_nums': 3},  
            'scroll': {'pattern': r'\((.*?),(.*?),(.*?)\)', 'agrs_nums': 3},  
            'long_press': {'pattern': r'\((.*?),(.*?)\)', 'agrs_nums': 2},  
            'navigate_back': None, 'kill_app': None,
        },
        'FORMAT2': {
            'long_touch': {'pattern': r'<\|box_start\|>\[(.*?),(.*?)\],\[(.*?),(.*?)\]<\|box_end\|>', 'agrs_nums': 4}, 
            'touch': {'pattern': r'<\|box_start\|>\[(.*?),(.*?)\],\[(.*?),(.*?)\]<\|box_end\|>', 'agrs_nums': 4}, 
            'scroll': {'pattern': r'<\|box_start\|>\[(.*?),(.*?)\],\[(.*?),(.*?)\]<\|box_end\|>(.*)', 'agrs_nums': 5}, 
            'set_text': {'pattern': r'<\|box_start\|>\[(.*?),(.*?)\],\[(.*?),(.*?)\]<\|box_end\|>(.*)', 'agrs_nums': 5},
            'back': None, 'kill_app': None, 
        },
    }

    MAX_DISTANCE = 100
    MAX_ANLS = 0.2


    def infer_func_call_format(self, query):
        if 'tap(x: float, y: float)' in query:
            func_call_format = 'FORMAT1'
            use_bbox_extract = False
        elif 'long_touch' in query:
            func_call_format = 'FORMAT2'
            use_bbox_extract = True
        else:
            logger.error(f'fail to infer_func_call_format from {query}')
            raise Exception
        
        return func_call_format, use_bbox_extract
    
    def extract_func_call(self, action_str, func_call_format, use_bbox_extract):
        action = {}

        if '(' in action_str and ')' not in action_str:
            action_str += ')'

        func_call_configs = self.FUNC_CALL_CONFIGS[func_call_format]
        for func_call_name in func_call_configs.keys():
            if action_str.startswith(func_call_name):
                action['func_call'] = func_call_name
                func_call_config = func_call_configs[func_call_name]
                if func_call_config:
                    try:
                        action_args = extract_action(action_str, func_call_config['pattern'])
                    except:
                        logger.error(f"extract_func_call - extract_action Error: {action_str}")
                        break
                    
                    try:
                        if use_bbox_extract:
                            if func_call_config['agrs_nums'] == 4:
                                x1, y1, x2, y2 = action_args
                            elif func_call_config['agrs_nums'] == 5:
                                x1, y1, x2, y2, text = action_args
                                action['text'] = text
                            bbox = [float(x1), float(y1), float(x2), float(y2)]
                            action['bbox'] = bbox

                        else:
                            if func_call_config['agrs_nums'] == 2:
                                x, y = action_args
                            elif func_call_config['agrs_nums'] == 3:
                                x, y, text = action_args
                                action['text'] = text
                            center_point = [float(x), float(y)]
                            action['center_point'] = center_point
                    
                    except:
                        logger.error(f"extract_func_call - postprocess Error: {action_args}")

                return action
            
        if not action:
            logger.error(f"extract_func_call - no match func_call Error: {action_args}")
        
        return action
    
    def eval_line(self, line):
        ret = {}

        func_call_format, use_bbox_extract = self.infer_func_call_format(line['query'])
        action_gt = self.extract_func_call(line['response'], func_call_format, use_bbox_extract)
        action_pred = self.extract_func_call(line['prediction'], func_call_format, use_bbox_extract)

        ret['match'] = 1
        for action_args in ['func_call', 'bbox', 'center_point', 'text']:
            if action_args not in action_gt:
                continue

            if action_args == 'func_call':
                if 'func_call' not in action_pred or action_gt['func_call'] != action_pred['func_call']:
                    ret['match'] = 0
                    ret['func_call_match'] = 0
                    break
                else:
                    ret['func_call_match'] = 1

            elif use_bbox_extract and action_args == 'bbox':
                if 'bbox' not in action_pred: 
                    ret['match'] = 0
                    ret['bbox_exist'] = 0

                else:
                    ret['bbox_exist'] = 1

                    gt_box = action_gt['bbox']
                    pred_box = action_pred['bbox']
                    iou = bbox_iou(pred_box, gt_box)
                    ret['bbox_iou'] = iou

                    pred_center_point = ((pred_box[0] + pred_box[2]) / 2, (pred_box[1] + pred_box[3]) / 2)
                    gt_center_point = ((gt_box[0] + gt_box[2]) / 2, (gt_box[1] + gt_box[3]) / 2)
                    if point_isin_bbox(pred_center_point, gt_box):
                        ret['center_point_in_bbox'] = 1
                    else:
                        ret['center_point_in_bbox'] = 0
                        ret['match'] = 0
                    
                    distance = points_distance(gt_center_point, pred_center_point, line['images'])
                    ret['center_points_distance'] = distance

            elif not use_bbox_extract and action_args == 'center_point':
                if 'center_point' not in action_pred: 
                    ret['match'] = 0
                    ret['center_point_exist'] = 0

                else:
                    ret['center_point_exist'] = 1

                    gt_center_point = action_gt['center_point']
                    pred_center_point = action_pred['center_point']
                    distance = points_distance(gt_center_point, pred_center_point, line['images'])
                    ret['center_points_distance'] = distance
                    if distance > self.MAX_DISTANCE:
                        ret['match'] = 0

            elif action_args == 'text' and action_gt['func_call'] == 'scroll':
                if 'text' not in action_pred:
                    ret['match'] = 0
                    ret['scroll_direction_exist'] = 0
                else:
                    ret['scroll_direction_exist'] = 1
                    if action_gt['text'] != action_pred['text']:
                        ret['match'] = 0
                        ret['scroll_direction_match'] = 0
                    else:
                        ret['scroll_direction_match'] = 1
            
            elif action_args == 'text':
                if 'text' not in action_pred:
                    ret['match'] = 0
                    ret['text_exist'] = 0
                else:
                    ret['text_exist'] = 1
                    anls_distance = anls_compute(action_gt['text'], action_pred['text'])
                    ret['anls_distance'] = anls_distance
                    if anls_distance > self.MAX_ANLS:
                        ret['match'] = 0

        ret['question'] = self.build_prompt_text(line)
        ret['pred'] = line['prediction']
        ret['gt'] = line['response']

        return ret
    
    def calculate_results(self, eval_file, infer_dataset, res):
        ret = {}
        res_df = pd.DataFrame(res)

        ret['Overall'] = res_df['match'].fillna(0).mean()
        ret['func_call_match'] = res_df['func_call_match'].fillna(0).mean()
        ret['bbox_exist'] = res_df['bbox_exist'].fillna(0).mean()
        ret['center_point_exist'] = res_df['center_point_exist'].fillna(0).mean()

        ret['bbox_iou'] = res_df['bbox_iou'].fillna(0).mean()
        ret['center_point_in_bbox'] = res_df['center_point_in_bbox'].fillna(0).mean()
        ret['center_points_distance'] = res_df['center_points_distance'].fillna(0).mean()

        ret['scroll_direction_exist'] = res_df['scroll_direction_exist'].fillna(0).mean()
        ret['scroll_direction_match'] = res_df['scroll_direction_match'].fillna(0).mean()
        ret['text_exist'] = res_df['text_exist'].fillna(0).mean()
        ret['anls_distance'] = res_df['anls_distance'].fillna(0).mean()

        outputs = d2df(ret)
        outputs.round(2)

        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        dump(outputs, result_file)
        return outputs
    
