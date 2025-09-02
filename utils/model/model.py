from __future__ import annotations

import os
import sys
import warnings
import math
import logging
import re

import torch

from . import BaseModel, Qwen2VLPromptMixin
from ..smp import get_rank_and_world_size, get_gpu_memory, auto_split_flag


def ensure_image_url(image: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:image;']
    if any(image.startswith(prefix) for prefix in prefixes):
        return image
    if os.path.exists(image):
        return 'file://' + image
    raise ValueError(f'Invalid image: {image}')


def ensure_video_url(video: str) -> str:
    prefixes = ['http://', 'https://', 'file://', 'data:video;']
    if any(video.startswith(prefix) for prefix in prefixes):
        return video
    if os.path.exists(video):
        return 'file://' + video
    raise ValueError(f'Invalid video: {video}')


def split_model():
    device_map = {}

    total_gpus = torch.cuda.device_count()
    rank, world_size = get_rank_and_world_size()
    num_gpus = total_gpus // world_size
    # + 8 is virtual layers for the memory of visual
    num_layers = 80 + 8
    num_layers_per_gpu = math.ceil(num_layers / num_gpus)
    num_layers_per_gpu = [num_layers_per_gpu] * num_gpus
    num_layers_per_gpu[0] -= 6
    num_layers_per_gpu[-1] -= 2
    layer_cnt = 0

    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'model.layers.{layer_cnt}'] = rank + i * world_size
            layer_cnt += 1

    last_gpu = rank + (num_gpus - 1) * world_size
    device_map['visual'] = rank
    device_map['model.embed_tokens'] = rank
    device_map['model.norm'] = last_gpu
    device_map['model.rotary_emb'] = last_gpu
    device_map['lm_head'] = last_gpu
    return device_map


class Qwen2VLChat(Qwen2VLPromptMixin, BaseModel):
    INSTALL_REQ = False
    INTERLEAVE = True
    VIDEO_LLM = True

    def __init__(
        self,
        model_path: str,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        max_new_tokens=2048,
        top_p=0.001,
        top_k=1,
        temperature=0.01,
        repetition_penalty=1.0,
        use_custom_prompt: bool = True,
        system_prompt: str | None = None,
        verbose: bool = False,
        abs_coord: bool = False,
    ):
        super().__init__(use_custom_prompt=use_custom_prompt)
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.generate_kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )
        self.system_prompt = system_prompt
        self.verbose = verbose
        self.fps = 2.0
        self.nframe = 64
        self.abs_coord = abs_coord

        rank, world_size = get_rank_and_world_size()

        if model_path is  None:
            raise ValueError('Error model path is None')
        self.model_path = model_path

        MODEL_CLS = None 

        from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
        MODEL_CLS = Qwen2VLForConditionalGeneration 
        default_path = model_path
        self.processor = Qwen2VLProcessor.from_pretrained(default_path)

        gpu_mems = get_gpu_memory()
        max_gpu_mem = max(gpu_mems) if gpu_mems != [] else -1
        if max_gpu_mem <= 0:
            raise ValueError('Error GPU mem , less than 0')
        

        # If only one process and GPU memory is less than 40GB
        if auto_split_flag():
            if world_size != 1:
                raise ValueError('Only support world_size == 1 when AUTO_SPLIT is set for non-72B Qwen2-VL') 
            # Will Use All GPUs to run one model
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='auto', attn_implementation='flash_attention_2'
            )
        elif '72b' not in self.model_path.lower():
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map='cpu', attn_implementation='flash_attention_2'
            )
            self.model.cuda().eval()
        else:
            self.model = MODEL_CLS.from_pretrained(
                model_path, torch_dtype='auto', device_map=split_model(), attn_implementation='flash_attention_2'
            )
            self.model.eval()

        torch.cuda.empty_cache()

    def _prepare_content(self, inputs: list[dict[str, str]], dataset: str | None = None) -> list[dict[str, str]]:
        """
        inputs list[dict[str, str]], each dict has keys: ['type', 'value']
        """
        content = []
        for s in inputs:
            if s['type'] == 'image':
                item = {'type': 'image', 'image': ensure_image_url(s['value'])}
                if dataset == 'OCRBench':
                    item['min_pixels'] = 10 * 10 * 28 * 28
                    warnings.warn(f"OCRBench dataset uses custom min_pixels={item['min_pixels']}")
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
                else:
                    if self.min_pixels is not None:
                        item['min_pixels'] = self.min_pixels
                    if self.max_pixels is not None:
                        item['max_pixels'] = self.max_pixels
            elif s['type'] == 'video':
                item = {'type': 'video', 'video': ensure_video_url(s['value'])}
                if self.fps is not None:
                    item['fps'] = self.fps
                elif self.nframe is not None:
                    item['nframes'] = self.nframe
            elif s['type'] == 'text':
                item = {'type': 'text', 'text': s['value']}
            else:
                raise ValueError(f"Invalid message type: {s['type']}, {s}")
            content.append(item)
        return content

    def extract_bboxs(self, bboxes_str, pattern1, pattern2=None):
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
            print(f"Extract Error: {bboxes_str}")
        return bboxes
    
    def extract_point(self, point_str, pattern):
        try:
            if '<point>' in point_str:
                point = re.findall(pattern, point_str)
                point = [int(value) for value in point[0]]
            else:
                point = []
        except Exception as e:
            print(f"extract_point Error: {point_str}, traceback: {e}")
            point = []

        return point
    
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
    
    def transfer_coord(self, input_str, input_height, input_width, reverse=False):
        BBOX_PATTERN = re.compile(r'<\|box_start\|>\((.*?),(.*?)\),\((.*?),(.*?)\)<\|box_end\|>')
        POINT_PATTERN = re.compile(r'<point>\((.*?),(.*?)\)')
        ACTION_PATTERN = re.compile(r'\(\D*(\d+).*,\D*(\d+).*\)')
        # REPL_PATTERN = re.compile(r'<\|box_start\|>(.*?)<\|box_end\|>')
        bboxes = self.extract_bboxs(input_str, BBOX_PATTERN)
        point = self.extract_point(input_str, POINT_PATTERN)
        action = re.findall(ACTION_PATTERN, input_str)

        if bboxes:
            bbox = bboxes[0]

            if reverse:
                abs_y1 = int(bbox[1]/input_height * 1000)
                abs_x1 = int(bbox[0]/input_width * 1000)
                abs_y2 = int(bbox[3]/input_height * 1000)
                abs_x2 = int(bbox[2]/input_width * 1000)
                bbox_str = f'({abs_x1},{abs_y1}),({abs_x2},{abs_y2})'
                if max(abs_y1, abs_x1, abs_y2, abs_x2) > 1000:
                    print('transfer_coord_error')
                    print(f"{bbox} to {bbox_str}")
            else:
                abs_y1 = int(bbox[1]/1000 * input_height)
                abs_x1 = int(bbox[0]/1000 * input_width)
                abs_y2 = int(bbox[3]/1000 * input_height)
                abs_x2 = int(bbox[2]/1000 * input_width)
                bbox_str = f'<|box_start|>({abs_x1},{abs_y1}),({abs_x2},{abs_y2})<|box_end|>'
            input_str = re.sub(BBOX_PATTERN, bbox_str, input_str)
        
        elif point:

            if reverse:
                abs_y1 = int(point[1]/input_height * 1000)
                abs_x1 = int(point[0]/input_width * 1000)
                point_str = f'<point>({abs_x1},{abs_y1})'
                if max(abs_y1, abs_x1) > 1000:
                    print('transfer_coord_error')
                    print(f"{point} to {point_str}")
            else:
                abs_y1 = int(point[1]/1000 * input_height)
                abs_x1 = int(point[0]/1000 * input_width)
                point_str = f'<point>({abs_x1},{abs_y1})'
            input_str = re.sub(POINT_PATTERN, point_str, input_str)
        
        elif action:

            for p in action:
                point = [float(i) for i in p]
                # print(point)
                if reverse:
                    abs_y1 = int(point[1]/input_height * 1000)
                    abs_x1 = int(point[0]/input_width * 1000)
                    if max(abs_y1, abs_x1) > 1000:
                        print('transfer_coord_error')
                        print(f"{point} to {(abs_x1,abs_y1)}")
                else:
                    abs_y1 = int(point[1]/1000 * input_height)
                    abs_x1 = int(point[0]/1000 * input_width)
                input_str = input_str.replace(p[0], str(abs_x1)).replace(p[1], str(abs_y1))
        else:
            input_str = input_str
        return input_str
    
    
    def generate_inner(self, message, dataset=None):
        try:
            from .vision_process import process_vision_info
        except Exception as err:
            logging.critical("qwen_vl_utils not found, please install it via 'pip install qwen-vl-utils'")
            raise err

        messages = []
        if self.system_prompt is not None:
            messages.append({'role': 'system', 'content': self.system_prompt})
        messages.append({'role': 'user', 'content': self._prepare_content(message, dataset=dataset)})
        if self.verbose:
            print(f'\033[31m{messages}\033[0m')

        if self.abs_coord:
            text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
            images, videos = process_vision_info([messages])
            inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
            
            input_height = inputs['image_grid_thw'][0][1]*14
            input_width = inputs['image_grid_thw'][0][2]*14
            for message in messages:
                message['content'][1]['text'] = self.transfer_coord(message['content'][1]['text'], input_height, input_width)

        text = self.processor.apply_chat_template([messages], tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info([messages])
        
        inputs = self.processor(text=text, images=images, videos=videos, padding=True, return_tensors='pt')
        inputs = inputs.to('cuda')

        generated_ids = self.model.generate(
            **inputs,
            **self.generate_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        out = self.processor.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        response = out[0].removesuffix('<|im_end|>')
        if self.verbose:
            print(f'\033[32m{response}\033[0m')

        if self.abs_coord:
            input_height = inputs['image_grid_thw'][0][1]*14
            input_width = inputs['image_grid_thw'][0][2]*14
            response = self.transfer_coord(response, input_height, input_width, reverse=True)

        return response