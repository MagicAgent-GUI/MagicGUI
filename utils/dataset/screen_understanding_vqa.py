import json
import re
from .utils.configs import *
from .screen_understanding import ScreenUnderstandingDataset, logger


ENTITIES_TYPES = ['输入框', '图标', '文本按钮', '选择按钮', '按钮', '下拉选项框', '开关', '多重滚动选择器', '滚动选择器', '导航栏', '列表项', '弹窗', '卡片视图', '日历选择器', '页面指示器', '日期选择器', '选项区', '选项框', '滑块', '文本', '通知', '广告']



def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def anls_compute(groundtruth, prediction):
    gt_answer = ' '.join(groundtruth.strip().lower().split())
    det_answer = ' '.join(prediction.strip().lower().split())
    dist = levenshtein_distance(gt_answer, det_answer)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    values = 0.0 if length == 0 else float(dist) / float(length)
    return values



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
            return results[0]
        else:
            continue
    return prompt


class VQADataset(ScreenUnderstandingDataset):
    TYPE = 'SCREEN_VQA'
    DATASET_CONFIGS = VQA_CONFIGS

    def __init__(self, dataset, dataset_path, model_name=None, model_type=None):
        super().__init__(dataset, dataset_path, model_name, model_type)

        if 'eval_method' in self.DATASET_CONFIGS[self.dataset_name]:
            if self.DATASET_CONFIGS[self.dataset_name]['eval_method'] == 'LLM':
                self.EVALUATE_METHOD = 'LLM'
            else:
                self.EVALUATE_METHOD = 'RULE'

    def eval_line(self, line):
        if self.EVALUATE_METHOD == 'LLM':
            return self.eval_line_by_llm(line)
        else:
            return self.eval_line_by_rule(line)

    def eval_line_by_rule(self, line):
        ret = {}

        gt_str = line['response']
        pred_str = line['prediction']
        if 'extract_infer_result' in self.DATASET_CONFIGS[self.dataset_name]:
            if 'extract_infer_args' in self.DATASET_CONFIGS[self.dataset_name]:
                args = self.DATASET_CONFIGS[self.dataset_name]['extract_infer_args']
                pred_str = self.DATASET_CONFIGS[self.dataset_name]['extract_infer_result'](pred_str, args)
            else:
                pred_str = self.DATASET_CONFIGS[self.dataset_name]['extract_infer_result'](pred_str)

        if self.DATASET_CONFIGS[self.dataset_name]['eval_method'] == 'ANLS':
            ret['match'] = 1-anls_compute(gt_str, pred_str)

        elif self.DATASET_CONFIGS[self.dataset_name]['eval_method'] == 'ACC':
            if (pred_str.strip().lower() == gt_str.strip().lower()):
                ret['match'] = 1
            elif (gt_str.strip().lower() in pred_str.strip().lower()):
                ret['match'] = 1
                ret['fuzzy_match'] = True
            else:
                ret['match'] = 0

        else:
            raise Exception

        ret['question'] = self.build_prompt_text(line)
        ret['pred'] = pred_str
        ret['gt'] = gt_str

        return ret


    def eval_line_by_llm(self, line):
        ret = {}

        question = self.build_prompt_text(line)
        gts = [line['response']]
        if 'other_responses' in line.keys():
            gts += eval(line['other_responses'])
        predict_result = line['prediction']
        max_score = -1
        reason = ''
        match_gt = ''
        for ground_truth in set(gts):
            init_instruction = '你是一个评委老师，需要你根据给出问题和标准答案，对被试者的回答进行评分，评分范围在[0,5]，5代表完全正确，0代表完全错误。你应该只关心回答内容是否正确，而不用关注句式和句子结构。评分结果需要按照json格式返回。'
            examples = self.DATASET_CONFIGS[self.dataset_name]['eval_examples']
            prompt = f'{{"question": {question}, "ground_truth": {ground_truth}, "predict_result": {predict_result}}}'

            messages = [
                {"role": "system", "content": init_instruction},
            ]
            for example in examples:
                messages.append({"role": "user", "content": example['user']})
                messages.append({"role": "assistant", "content": example['assistant']})

            messages.append({"role": "user", "content": prompt})

            response = self.llm_eval(messages)

            try:
                formated_response = json.loads(response)
            except:
                try:
                    reason_str = re.findall(self.REASON_PATTERN, response)[0]
                    reason_str_new = re.sub('"', r'\"', reason_str)
                    res = re.sub(reason_str, reason_str_new, response)
                    formated_response = json.loads(res)
                except:
                    try:
                        res = re.findall(self.SCORE_PATTERN, response)[0]
                        formated_response = {'reason': response, 'score': int(res)}
                    except:
                        logger.error(f"llm evaluator response error: {response}")
                        formated_response = {'reason': 'Extract error: ' + response, 'score': 0}

            try:
                score = formated_response['score'] / 5
                if score > max_score:
                    max_score = score
                    reason = response
                    match_gt = ground_truth
            except:
                logger.error(f"llm evaluator score error: {response}")

        ret['match'] = max_score
        ret['match_reason'] = reason

        ret['question'] = question
        ret['pred'] = predict_result
        ret['gt'] = match_gt

        return ret
    