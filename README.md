<div align="center">
  <img src="./assets/MagicGUI_logo.png" width="600em"></img>
</div>

<p align="center">
    【English | <a href="README_zh.md">中文</a>】
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#framework">Framework</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#Download the MagicGUI">Model</a> •
  <a href="#action-space">Action Space</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#performance-evaluation">Performance</a> •
  <a href="https://arxiv.org/abs/2508.03700">Technical Report</a>
</p>

## News

* [2025-07-20] 📄📄📄 We have released the **technical report** of MagicGUI! Check it out [here](https://arxiv.org/abs/2508.03700).
* [2025-09-01] 🚀🚀🚀 We have released the **MagicGUI**, an on-device GUI agent capable of operating Chinese & English apps and equipped with RFT-enhanced reasoning abilities.

## Overview

MagicGUI is an open-source GUI agent model developed by Honor, built on Qwen2-VL with 7 billion parameters. It demonstrates outstanding capabilities in visual grounding, screen question answering, and action sequence planning and execution. MagicGUI enables multimodal perception, understanding, and automated execution of user tasks on mobile devices.

**Data Collection Framework**: Propose a scalable and modular framework for GUI data collection that efficiently gathers high-quality data on mobile devices.

**Powerful Perception and Grounding Capabilities**: Enhance the perception and grounding abilities on mobile device screens by integrating large-scale knowledge through tasks such as element referring, element grounding, and screen captioning.

**Unified Action Space**: Develop a comprehensive and unified action space for various mobile platforms, encompassing fundamental operations like Tap, Text Input, and Scroll, while also supporting more complex actions such as Wait, Drag, and Takeover.

**Planning-Oriented Reasoning**: Implement a planning-oriented reasoning mechanism to improve the stability of task execution and enhance the accuracy of action decisions in dynamic environments.

**Two-Stage Training Paradigm**: Strengthen core perception, localization, and navigation capabilities through Continued Pre-training (CPT), while enhancing model robustness and generalization via Reinforcement Fine-tuning (RFT).

## Framework
The overall training framework of our MagicGUI contains two stages:

**Stage I**: Continue Pre-training (CPT), which involves training a
foundational model on a large and diverse dataset followed by an annealing phase using a balanced and high-quality
dataset.

**Stage II**: Reinforcement Fine-tuning (RFT), aimed at further enhancing the
model’s robustness and generalization capabilities.

<div align="center">
  <img src="./assets/framework.png" width="800em"></img>
</div>

## Quick Start

### Install dependencies

```bash
git clone https://github.com/MagicAgent-GUI
cd MagicGUI
conda create -n gui_agent python=3.11
conda activate gui_agent
pip install -r requirements.txt
```

### Download the MagicGUI 

Download [MagicGUI_CPT](https://huggingface.co/GUIAgent/MagicGUI_CPT) .
Download [MagicGUI_RFT](https://huggingface.co/GUIAgent/MagicGUI_RFT) .
#### Huggingface Inference

```python
import torch
from utils.model import Qwen2VLChat

# 1. Load the model and tokenizer
model_path = "./models/RFT"  # model path
model = Qwen2VLChat.from_pretrained(model_path, min_pixels=4*28*28, max_pixels=768*28*28)
model = model.to("cuda:0") 

# 2. Build the input
instruction = """你是一个训练有素的手机智能体，能够帮助用户进行单步导航任务。已知当前智能手机的截图<image>，和用户指令"查看会员信息"请输出正确的函数调用以实现用户指令。除了函数调用之外，你不能输出任何其他内容。你可以调用以下函数来控制智能手机：- UI基础操作：1. tap(x: float,y: float) 该函数用于在智能手机屏幕上点击特定点。坐标 x 和 y 表示待点击控件的中心位置。2. scroll(x: float,y: float,direction: str) 该函数用于从起始坐标 (x,y) 开始在智能手机屏幕上滑动操作，方向为手指滑动的方向。坐标 x 和 y 表示屏幕上待滑动控件的中心位置。方向可以是 "up"、"down"、"left" 或 "right"。3. text(x: float,y: float,text_input: str) 该函数用于在智能手机屏幕上输入指定的text。坐标 x 和 y 表示待点击控件的中心位置。- 手机按键操作：4. navigate_back() 该函数用于返回智能手机的上一个屏幕。5. navigate_home() 该函数用于返回手机的home screen或关闭当前应用。- 其他操作：6. long_press(x: float,y: float) 该函数用于在智能手机屏幕上的特定点执行长按操作。坐标 x 和 y 表示待点击控件的中心位置。7. wait() 该函数表示在当前页面等候。8. enter() 该函数表示按下enter键。9. take_over(text_input: str) 该函数用于提示用户接管智能手机，其中 text_input 是提示用户接管手机的原因。如果原因不确定，请填写“请您接管当前界面”。10. drag(x1: float,y1: float,x2: float,y2: float) 该函数执行一个对起始和终点敏感的拖动操作，表示手指从点1拖到点2。常见的场景包括滑块拖动、滚动选择器拖动和图片裁剪。11. screen_shot() 该函数用于截图。12. long_screen_shot() 该函数执行长截图。13. call_api(api_name: str,params: str) 调用指定的API并传入给定的参数。api_name是API的名称。params包含API所需的输入参数。例如，call_api(Amazon, open)意味着打开亚马逊APP。如果你发现当前指令无法在当前页面上执行，你需要输出no_answer。如果你发现当前指令已完成，你需要输出action_completed。"""

image_path = "./assets/test_action.png"

# 3. Build the message format
messages = [{"type": "image", "value":f"{image_path}",
            {"type": "text", "value":f"{instruction}"]

# 4. Inference
response = model.generate(
    message = messages,
)

print(response)
```

Expected output:

```JSON
{"tap(700,964)"}
```

### Action Space

At each step, the agent outputs is a single JSON object that contains:
- One (and only one) primitive action, chosen from the list below;
- Optional modifiers (`duration`, `thought`) and/or a task-level flag (`STATUS`).

Note that all keywords are **case-sensitive**, and we use **compact JSON** (i.e., no extra whitespace), which affects the tokenizer’s behavior.

<table>
  <thead>
    <tr>
      <th>Action</th>
      <th>Description</th>
      <th>Conditions for R<sub>acc</sub> = +2</th>
      <th>Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><b>Tap</b></td>
      <td>Click at coordinate (x, y)</td>
      <td>dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14%</td>
      <td><code>tap(x,y)</code></td>
    </tr>
    <tr>
      <td><b>Scroll</b></td>
      <td>Scroll at coordinate (x, y) with<br>direction up / down / left / right</td>
      <td>dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14%<br>and direction = gt[direction]</td>
      <td><code>scroll(x,y,direction)</code></td>
    </tr>
    <tr>
      <td><b>Text Input</b></td>
      <td>Type <i>text</i> at coordinate (x, y)</td>
      <td>dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14%<br>and F1(text, gt[text]) > 0.5</td>
      <td><code>text(x,y,text_input)</code></td>
    </tr>
    <tr>
      <td><b>Navigation Back</b></td>
      <td>Adb command to go back to the previous page</td>
      <td>–</td>
      <td><code>navigate_back()</code></td>
    </tr>
    <tr>
      <td><b>Navigation Home</b></td>
      <td>Adb command to go to the home screen of the mobile</td>
      <td>–</td>
      <td><code>navigate_home()</code></td>
    </tr>
    <tr>
      <td><b>Long Press</b></td>
      <td>Long press at coordinate (x, y)</td>
      <td>dist([x, y], [x<sub>c</sub>, y<sub>c</sub>]) ≤ 14%</td>
      <td><code>long_press(x,y)</code></td>
    </tr>
    <tr>
      <td><b>Finish</b></td>
      <td>Indicate that navigation task has been completed</td>
      <td>–</td>
      <td><code>finish()</code></td>
    </tr>
    <tr>w
      <td><b>Wait</b></td>
      <td>Wait for several seconds</td>
      <td>–</td>
      <td><code>wait()</code></td>
    </tr>
    <tr>
      <td><b>Enter</b></td>
      <td>Adb command to press enter</td>
      <td>–</td>
      <td><code>enter()</code></td>
    </tr>
    <tr>
      <td><b>Takeover</b></td>
      <td>Request user takeover</td>
      <td>–</td>
      <td><code>take_over(message)</code></td>
    </tr>
    <tr>
      <td><b>Drag</b></td>
      <td>Drag from coordinate (x₁, y₁) to (x₂, y₂)</td>
      <td>
        dist([x₁, y₁], [x<sub>1c</sub>, y<sub>1c</sub>]) ≤ 7.5%<br>
        and dist([x₂, y₂], [x<sub>2c</sub>, y<sub>2c</sub>]) ≤ 7.5%
      </td>
      <td><code>drag(x1,y1,x2,y2)</code></td>
    </tr>
    <tr>
      <td><b>Call API</b></td>
      <td>Adb command to <i>open</i> or <i>kill</i> app</td>
      <td>app = gt[app]<br>and open/kill = gt[operation]</td>
      <td><code>call_api(api_name,operation)</code></td>
    </tr>
    <tr>
      <td><b>Screenshot</b></td>
      <td>Adb command to take a screenshot</td>
      <td>–</td>
      <td><code>screen_shot()</code></td>
    </tr>
    <tr>
      <td><b>Long Screenshot</b></td>
      <td>Adb command to take a long screenshot</td>
      <td>–</td>
      <td><code>long_screen_shot()</code></td>
    </tr>
  </tbody>
</table>


## Evaluation
### 1.Data preparation
Please download the dataset from the subset from the [Magic-RICH dataset](https://huggingface.co/datasets/GUIAgent/Magic-RICH) and palce the folders into the .datasets/ directory.

- `assets/` 
- `datasets/` 
  - `Routine`
  - `Instruction`
  - `Complex`
  - `Handing_Exception`
- `utils/`

For the preparation of other open-source datasets, please refer to [Other datasets preparation](datasets/eval_data_process/readme.md).

### 2. Param
We use run_eval.py for evaluation.

- `--data`: Name of a eval dataset
- `--model`: Path to the model  
- `--work-dir (str, default to '.')`: Directory to save evaluation results  
- `--mode (str, default: 'all', choices: ['all', 'infer'])`: If set to "all", the script performs both inference and evaluation; if set to "infer", it performs inference only.
- `--eval_model_path (str, default: 'None')`:'Path to eval model (required if mode is 'all' and data is 'ScreenQA-short')'

### 3. Run
```python
# Referring Benchmark
python run_eval.py --data ScreenQA-short --model MagicGUI_Path  --mode all --eval_model_path Eval_Model_Path
python run_eval.py --data ScreenSpot_v2_mobile --model MagicGUI_Path  --mode all
python run_eval.py --data Os-Atlas-mobile --model MagicGUI_Path  --mode all
# Magic-RICH dataset
python run_eval.py --data Routine --model MagicGUI_Path  --mode all
python run_eval.py --data Complex --model MagicGUI_Path  --mode all
python run_eval.py --data Instruction --model MagicGUI_Path  --mode all
python run_eval.py --data Handling_Exception --model MagicGUI_Path  --mode all
# Open-source AndroidControl and GUI-Odyssey
python run_eval.py --data AC-Low --model MagicGUI_Path  --mode all
python run_eval.py --data AC-High --model MagicGUI_Path  --mode all
python run_eval.py --data GUI-Odyssey --model MagicGUI_Path  --mode all
```

## Performance Evaluation

### Performance comparison on the Referring Benchmark
<table>
  <thead>
    <tr>
      <th rowspan="1">Agent Models</th>
      <th colspan="1">ScreenQA-short</th>
      <th colspan="1">ScreenSpot v2 mobile</th>
      <th colspan="1">Os-Atlas-mobile</th>
    </tr>
  </thead>
  <tbody>
    <!-- Closed-source Models -->
    <tr><td colspan="4"><em>Closed-source Models</em></td></tr>
    <tr>
      <td>GPT-4o (Hurst et al., 2024)</td>
      <td>90.3</td><td>10.6</td><td>4.6</td>
    </tr>
    <tr>
      <td>Gemini 2.0 (Pichai et al., 2024)</td>
      <td>90.4</td><td>10.6</td><td>5.8</td>
    </tr>
    <!-- Open-source Models -->
    <tr><td colspan="4"><em>Open-source Models</em></td></tr>
    <tr>
      <td>InternVL-2-8B (Chen et al., 2024)</td>
      <td>88.4</td><td>4.2</td><td>2.4</td>
    </tr>
    <tr>
      <td>Qwen2-VL-7B (Wang et al., 2024)</td>
      <td>92.6</td><td>70.7</td><td>27.2</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-7B (Bai et al., 2025)</td>
      <td>92.1</td><td>56.1</td><td>26.6</td>
    </tr>
    <tr>
      <td>UI-TARS-7B (Qin et al., 2025)</td>
      <td><b>95.4</b></td><td>88.6</td><td>82.5</td>
    </tr>
    <tr>
      <td>UI-TARS-1.5-7B (Seed, 2025)</td>
      <td>93.0</td><td>85.8</td><td>79.3</td>
    </tr>
    <!-- MagicGUI -->
    <tr style="background-color:#e8eafc;">
      <td>MagicGUI-CPT</td>
      <td>94.6</td><td><b>90.2</b></td><td><b>95.2</b></td>
    </tr>
  </tbody>
</table>


### Performance comparison on the Magic-RICH dataset

<table>
  <thead>
    <tr>
      <th rowspan="2">Agent Models</th>
      <th colspan="3">Routine</th>
      <th colspan="3">Instruction</th>
      <th colspan="3">Complex</th>
      <th rowspan="2">Handing Exception</th>
    </tr>
    <tr>
      <th>Type</th><th>Grd</th><th>SR</th>
      <th>Type</th><th>Grd</th><th>SR</th>
      <th>Type</th><th>Grd</th><th>SR</th>
    </tr>
  </thead>
  <tbody>
    <!-- Closed-source Models -->
    <tr><td colspan="11"><em>Closed-source Models</em></td></tr>
    <tr>
      <td>GPT-4o (Hurst et al., 2024)</td>
      <td>49.3</td><td>16.7</td><td>4.6</td>
      <td>56.6</td><td>13.5</td><td>19.8</td>
      <td>49.0</td><td>14.6</td><td>7.4</td>
      <td>85.1</td>
    </tr>
    <tr>
      <td>Gemini 2.0 (Pichai et al., 2024)</td>
      <td>89.2</td><td>49.4</td><td>34.7</td>
      <td>84.1</td><td>54.2</td><td>51.4</td>
      <td>83.3</td><td>50.3</td><td>42.0</td>
      <td>73.7</td>
    </tr>
    <!-- Open-source Models -->
    <tr><td colspan="11"><em>Open-source Models</em></td></tr>
    <tr>
      <td>InternVL-2-8B (Chen et al., 2024)</td>
      <td>30.1</td><td>2.8</td><td>1.3</td>
      <td>37.1</td><td>4.0</td><td>15.8</td>
      <td>17.1</td><td>6.0</td><td>1.3</td>
      <td>70.8</td>
    </tr>
    <tr>
      <td>Qwen2-VL-7B (Wang et al., 2024)</td>
      <td>71.7</td><td>41.0</td><td>28.1</td>
      <td>73.6</td><td>43.9</td><td>41.5</td>
      <td>65.6</td><td>28.7</td><td>21.2</td>
      <td>68.3</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-7B (Bai et al., 2025)</td>
      <td>94.3</td><td>92.6</td><td>76.3</td>
      <td>89.3</td><td><u>95.7</u></td><td>83.6</td>
      <td>86.6</td><td>69.6</td><td>60.0</td>
      <td>67.0</td>
    </tr>
    <tr>
      <td>UI-TARS-7B (Qin et al., 2025)</td>
      <td>83.5</td><td>84.9</td><td>73.3</td>
      <td>76.6</td><td>85.6</td><td>69.8</td>
      <td>91.4</td><td>69.1</td><td>67.0</td>
      <td>3.6</td>
    </tr>
    <tr>
      <td>UI-TARS-1.5-7B (Seed, 2025)</td>
      <td>85.6</td><td>96.2</td><td>81.5</td>
      <td>78.6</td><td>92.1</td><td>72.2</td>
      <td><b>94.7</b></td><td>74.3</td><td>71.1</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>MiMo-VL-7B-SFT (Xiaomi, 2025)</td>
      <td>93.0</td><td>77.9</td><td>65.3</td>
      <td>89.7</td><td>85.7</td><td>75.4</td>
      <td>89.1</td><td>80.1</td><td>71.0</td>
      <td>57.0</td>
    </tr>
    <tr>
      <td>AgentCPM-GUI (Zhang et al., 2025)</td>
      <td>84.3</td><td>92.2</td><td>75.1</td>
      <td>70.4</td><td>80.7</td><td>56.0</td>
      <td>72.3</td><td>54.6</td><td>39.4</td>
      <td>2.4</td>
    </tr>
    <!-- MagicGUI -->
    <tr style="background-color:#e8eafc;">
      <td>MagicGUI-CPT</td>
      <td><b>98.5</b></td><td><b>98.5</b></td><td><b>97.2</b></td>
      <td><b>95.5</b></td><td><b>96.3</b></td><td><b>92.9</b></td>
      <td>88.5</td><td><b>82.3</b></td><td><b>72.9</b></td>
      <td><b>93.2</b></td>
    </tr>
    <tr style="background-color:#e8eafc;">
      <td>MagicGUI-RFT</td>
      <td><b>99.7</b></td><td>97.5</td><td><b>97.5</b></td>
      <td><b>97.2</b></td><td>95.6</td><td><b>94.0</b></td>
      <td>92.1</td><td>80.4</td><td><b>74.1</b></td>
      <td>92.1</td>
    </tr>
  </tbody>
</table>







### Performance comparison on open-source AndroidControl and GUI-Odyssey datasets. 

<table>
  <thead>
    <tr>
      <th rowspan="2">Agent Models</th>
      <th colspan="2">AC-Low</th>
      <th colspan="2">AC-High</th>
      <th colspan="2">GUI-Odyssey</th>
    </tr>
    <tr>
      <th>Type</th><th>SR</th>
      <th>Type</th><th>SR</th>
      <th>Type</th><th>SR</th>
    </tr>
  </thead>
  <tbody>
    <!-- Closed-source Models -->
    <tr><td colspan="7"><em>Closed-source Models</em></td></tr>
    <tr>
      <td>GPT-4o (Hurst et al., 2024)</td>
      <td>-</td><td>19.5</td>
      <td>-</td><td>20.8</td>
      <td>-</td><td>20.4</td>
    </tr>
    <tr>
      <td>Gemini 2.0 (Pichai et al., 2024)</td>
      <td>-</td><td>28.5</td>
      <td>-</td><td>60.2</td>
      <td>-</td><td>3.3</td>
    </tr>
    <tr>
      <td>Claude 2.0 (Anthropic, 2024)</td>
      <td>-</td><td>28.5</td>
      <td>-</td><td>12.5</td>
      <td>60.9</td><td>-</td>
    </tr>
    <!-- Open-source Models -->
    <tr><td colspan="7"><em>Open-source Models</em></td></tr>
    <tr>
      <td>Qwen2-VL-7B (Wang et al., 2024)</td>
      <td>55.7</td><td>36.2</td>
      <td>45.8</td><td>21.2</td>
      <td>58.6</td><td>13.3</td>
    </tr>
    <tr>
      <td>Qwen2.5-VL-7B (Bai et al., 2025)</td>
      <td>94.1</td><td>85.0</td>
      <td>75.1</td><td>62.9</td>
      <td>59.5</td><td>46.3</td>
    </tr>
    <tr>
      <td>Aguvis-7B (Xu et al., 2024)</td>
      <td>93.9</td><td>89.4</td>
      <td>65.6</td><td>54.2</td>
      <td>26.7</td><td>13.5</td>
    </tr>
    <tr>
      <td>OS-Atlas-7B (Wu et al., 2024)</td>
      <td>73.0</td><td>67.3</td>
      <td>70.4</td><td>56.5</td>
      <td>91.8*</td><td>76.8*</td>
    </tr>
    <tr>
      <td>UI-TARS-7B (Qin et al., 2025)</td>
      <td>95.2</td><td>91.8</td>
      <td>81.6</td><td>74.4</td>
      <td>86.1</td><td>67.9</td>
    </tr>
    <tr>
      <td>AgentCPM-GUI (Zhang et al., 2025)</td>
      <td>94.4</td><td>90.2</td>
      <td>77.7</td><td>69.2</td>
      <td><b>90.9</b></td><td><b>75.0</b></td>
    </tr>
    <!-- MagicGUI -->
    <tr style="background-color:#e8eafc;">
      <td>MagicGUI-CPT</td>
      <td>94.5</td><td>86.7</td>
      <td>84.6</td><td>73.1</td>
      <td><b>90.4</b></td><td>73.5</td>
    </tr>
    <tr style="background-color:#e8eafc;">
      <td>MagicGUI-RFT</td>
      <td><b>97.2</b></td><td><b>93.5</b></td>
      <td><b>84.7</b></td><td><b>76.3</b></td>
      <td>89.7</td><td><b>74.3</b></td>
    </tr>
  </tbody>
</table>

## License

* This project is licensed under the [Apache-2.0](./LICENSE) license. The model weights are fully open for academic research, and commercial use licenses can be applied for by contacting magicgui@honor.com. This project uses the pre-trained Qwen2VL-7B-Instruct for initialization, which is also licensed under the Apache- 2.0 License.

## Citation

If **MagicGUI** is useful for your research, please cite:

```bibtex
@article{tang2025magicguifoundationalmobilegui,
      title={MagicGUI: A Foundational Mobile GUI Agent with Scalable Data Pipeline and Reinforcement Fine-tuning}, 
      author={Liujian Tang and Shaokang Dong and Yijia Huang and Minqi Xiang and Hongtao Ruan and Bin Wang and Shuo Li and Zhiheng Xi and Zhihui Cao and Hailiang Pang and Heng Kong and He Yang and Mingxu Chai and Zhilin Gao and Xingyu Liu and Yingnan Fu and Jiaming Liu and Xuanjing Huang and Yu-Gang Jiang and Tao Gui and Qi Zhang and Kang Wang and Yunke Zhang and Yuran Wang},
      year={2025},
      eprint={2508.03700},
      archivePrefix={arXiv},
      primaryClass={cs.HC},
      url={https://arxiv.org/abs/2508.03700}, 
}
```
