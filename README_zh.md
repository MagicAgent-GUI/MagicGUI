<div align="center">
  <img src="./assets/MagicGUI_logo.png" width="600em"></img>
</div>

<p align="center">
    【<a href="README.md">English</a> | 中文】
</p>

<p align="center">
  <a href="#概览">概览</a> •
  <a href="#框架">框架</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#下载模型">模型</a> •
  <a href="#动作空间">动作空间</a> •
  <a href="#评测">评测</a> •
  <a href="#性能评测">性能对比</a> •
  <a href="https://arxiv.org/abs/2508.03700">技术报告</a>
</p>

## 新闻

* [2025-09-01] 🚀🚀🚀 开源了 **MagicGUI**，一个通过RFT增强推理能力，能够在移动设备上操作中文和英文app的GUI agent。
* [2025-08-20] 📊📊📊 发布了 **Magic-RICH**测试集！点击查看 [here](https://huggingface.co/datasets/GUIAgent/Magic-RICH)。
* [2025-07-20] 📄📄📄 发布了 **MagicGUI**技术报告！点击查看 [here](https://arxiv.org/abs/2508.03700)。

## 概览

MagicGUI 是由荣耀研发的开源 GUI 智能体模型，基于 Qwen2-VL（70 亿参数）。它在视觉定位、屏幕问答、动作序列规划与执行方面展现出卓越能力。MagicGUI 能够在移动设备上实现多模态感知、理解与自动化任务执行。

**数据采集框架**：提出可扩展、模块化的 GUI 数据采集框架，高效获取移动设备上的高质量数据。  

**强大的感知与定位能力**：通过元素指代、元素定位、屏幕描述等任务，结合大规模知识提升感知与定位能力。  

**统一动作空间**：为不同移动平台设计统一的动作空间，涵盖点击、输入、滑动等基本操作，同时支持等待、拖拽、接管等复杂动作。  

**面向规划的推理机制**：引入规划导向的推理机制，提升任务执行的稳定性与动态环境下动作决策的准确性。  

**双阶段训练范式**：通过持续预训练（CPT）增强核心感知、定位与导航能力，并通过强化微调（RFT）提升模型鲁棒性与泛化能力。  

## 框架

MagicGUI 的整体训练框架分为两个阶段：

**阶段 I**：持续预训练（CPT），先在大规模多样化数据集上训练基础模型，再通过平衡的高质量数据集进行退火训练。  

**阶段 II**：强化微调（RFT），进一步增强模型的鲁棒性与泛化能力。  

<div align="center">
  <img src="./assets/framework.png" width="800em"></img>
</div>

## 快速开始

### 安装依赖（需要修改）

```bash
git clone https://github.com/MagicAgent-GUI
cd MagicGUI
conda create -n gui_agent python=3.11
conda activate gui_agent
pip install -r requirements.txt
```

### 下载模型

下载 [MagicGUI_CPT](https://huggingface.co/GUIAgent/MagicGUI_CPT) .

下载 [MagicGUI_RFT](https://huggingface.co/GUIAgent/MagicGUI_RFT) .

#### Huggingface 推理示例

```python
import torch
from utils.model import Qwen2VLChat

# 1. 加载模型和 tokenizer
model_path = "./models/RFT"  # 模型路径
model = Qwen2VLChat.from_pretrained(model_path, min_pixels=4*28*28, max_pixels=768*28*28)
model = model.to("cuda:0") 

# 2. 构建输入
instruction = """你是一个训练有素的手机智能体，能够帮助用户进行单步导航任务。已知当前智能手机的截图<image>，和用户指令"查看会员信息"请输出正确的函数调用以实现用户指令。除了函数调用之外，你不能输出任何其他内容。你可以调用以下函数来控制智能手机：..."""

image_path = "./assets/test_action.png"

# 3. 构建消息格式
messages = [{"type": "image", "value":f"{image_path}",
            {"type": "text", "value":f"{instruction}"]

# 4. 推理
response = model.generate(
    message = messages,
)

print(response)
```

预期输出：

```JSON
{"tap(700,964)"}
```

## 动作空间

智能体在每一步的输出是一个 JSON 对象，包含：
- **一个且仅一个**原子动作；
- 可选修饰符（`duration`, `thought`）和任务级标志（`STATUS`）。

注意：所有关键词 **区分大小写**，并使用 **紧凑 JSON**（无多余空格）。

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
    <tr>
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

## 评测

### 1. 数据准备

请从 [Magic-RICH 数据集](https://huggingface.co/datasets/GUIAgent/Magic-RICH) 下载子集，并将其放入 `.datasets/` 目录中。

- `assets/`  
- `datasets/`  
  - `Routine`
  - `Instruction`
  - `Complex`
  - `Handing_Exception`
- `utils/`

其他开源数据集的准备方式请参考 [datasets/eval_data_process/readme.md](datasets/eval_data_process/readme.md)。

### 2. 参数

我们使用 `run_eval.py` 进行评测。

- `--data`: 数据集名称  
- `--model`: 模型路径  
- `--work-dir`: 保存评测结果的目录（默认 `.`）  
- `--mode`: 执行模式（默认 `all`，可选 `all` 或 `infer`）  
- `--eval_model_path`: 评测模型路径（当 `mode=all` 且 `data=ScreenQA-short` 时必填）  

### 3. 运行示例

```python
python run_eval.py --data ScreenQA-short --model MagicGUI_Path  --mode all --eval_model_path Eval_Model_Path
python run_eval.py --data ScreenSpot_v2_mobile --model MagicGUI_Path  --mode all
python run_eval.py --data Os-Atlas-mobile --model MagicGUI_Path  --mode all
python run_eval.py --data Routine --model MagicGUI_Path  --mode all
python run_eval.py --data Complex --model MagicGUI_Path  --mode all
python run_eval.py --data Instruction --model MagicGUI_Path  --mode all
python run_eval.py --data Handling_Exception --model MagicGUI_Path  --mode all
python run_eval.py --data AC-Low --model MagicGUI_Path  --mode all
python run_eval.py --data AC-High --model MagicGUI_Path  --mode all
python run_eval.py --data GUI-Odyssey --model MagicGUI_Path  --mode all
```

## 性能评测

### 基准数据集对比
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


### Magic-RICH 数据集对比

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







### 在开源的AndroidControl and GUI-Odyssey datasets上的对比. 

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

## 许可协议

* 本项目基于 [Apache-2.0](./LICENSE) 协议开源。模型权重完全开放供学术研究使用，商业使用需联系 magicgui@honor.com 获取授权。本项目使用了 Qwen2VL-7B-Instruct 作为初始化模型，该模型同样遵循 Apache-2.0 协议。

## 引用

如果 **MagicGUI** 对您的研究有帮助，请引用：

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

