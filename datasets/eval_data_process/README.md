# Data Processing Scripts

```
# Setup environment

cd MagicGUI/datasets/eval_data_process
conda create -n process_data python=3.11
conda activate process_data
pip install -r requirements.txt

mkdir tmp && cd tmp
git clone https://github.com/deepmind/android_env/
cd android_env; pip install .
```

## Android Control

Download [Android Control](https://github.com/google-research/google-research/tree/master/android_control) and save at ``MagicGUI/datasets/eval_data_process/tmp/android_control``

```
cd MagicGUI/datasets/eval_data_process
python process_android_control.py
```

## ScreenQA-Short

Download [test-00000-of-00002.parquet](https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA-Short/blob/main/data/test-00000-of-00002.parquet) and [test-00001-of-00002.parquet](https://huggingface.co/datasets/rootsautomation/RICO-ScreenQA-Short/blob/main/data/test-00001-of-00002.parquet) of ScreenQA-Short and save at ``MagicGUI/datasets/eval_data_process/tmp/ScreenQA-short``
```
cd MagicGUI/datasets/eval_data_process
python process_screenqa.py
```

## ScreenSpot-v2-mobile

Download [ScreenSpot-v2-mobile](https://huggingface.co/datasets/HongxinLi/ScreenSpot_v2) and save at ``MagicGUI/datasets/eval_data_process/tmp/ScreenSpot-v2-mobile``
```
cd MagicGUI/datasets/eval_data_process
python process_screenspotv2.py
```

## Os-Atlas-mobile

Download and unzip[mobile_images.zip](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data/blob/main/mobile_domain/mobile_images.zip), and [amex_raw.json](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data/blob/main/mobile_domain/amex_raw.json) of Os-Atlas-mobile at ``MagicGUI/datasets/eval_data_process/tmp/Os-Atlas-mobile``
```
cd MagicGUI/datasets/eval_data_process
python process_os_atlas.py.py
```

## Gui-odyssey

Download [GUI-Odyssey](https://github.com/OpenGVLab/GUI-Odyssey?tab=readme-ov-file) and save at ``MagicGUI/datasets/eval_data_process/tmp/GUI-Odyssey``

```
cd MagicGUI/datasets/eval_data_process
python process_odyssey.py
```
