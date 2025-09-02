import torch
from utils.model import Qwen2VLChat

# 1. Load the model and tokenizer
model_path = "/opt/nas/p/mm/ie_env/cmx/HONOR_TEST5/models/CPT"  # model path
# model = Qwen2VLChat.from_pretrained(model_path, min_pixels=4*28*28, max_pixels=768*28*28)
model = Qwen2VLChat(model_path, min_pixels=4*28*28, max_pixels=768*28*28)

# 2. Build the input
instruction = """你是一个训练有素的手机智能体，能够帮助用户进行单步导航任务。已知当前智能手机的截图<image>，和用户指令"查看会员信息"请输出正确的函数调用以实现用户指令。除了函数调用之外，你不能输出任何其他内容。你可以调用以下函数来控制智能手机：- UI基础操作：1. tap(x: float,y: float) 该函数用于在智能手机屏幕上点击特定点。坐标 x 和 y 表示待点击控件的中心位置。2. scroll(x: float,y: float,direction: str) 该函数用于从起始坐标 (x,y) 开始在智能手机屏幕上滑动操作，方向为手指滑动的方向。坐标 x 和 y 表示屏幕上待滑动控件的中心位置。方向可以是 "up"、"down"、"left" 或 "right"。3. text(x: float,y: float,text_input: str) 该函数用于在智能手机屏幕上输入指定的text。坐标 x 和 y 表示待点击控件的中心位置。- 手机按键操作：4. navigate_back() 该函数用于返回智能手机的上一个屏幕。5. navigate_home() 该函数用于返回手机的home screen或关闭当前应用。- 其他操作：6. long_press(x: float,y: float) 该函数用于在智能手机屏幕上的特定点执行长按操作。坐标 x 和 y 表示待点击控件的中心位置。7. wait() 该函数表示在当前页面等候。8. enter() 该函数表示按下enter键。9. take_over(text_input: str) 该函数用于提示用户接管智能手机，其中 text_input 是提示用户接管手机的原因。如果原因不确定，请填写“请您接管当前界面”。10. drag(x1: float,y1: float,x2: float,y2: float) 该函数执行一个对起始和终点敏感的拖动操作，表示手指从点1拖到点2。常见的场景包括滑块拖动、滚动选择器拖动和图片裁剪。11. screen_shot() 该函数用于截图。12. long_screen_shot() 该函数执行长截图。13. call_api(api_name: str,params: str) 调用指定的API并传入给定的参数。api_name是API的名称。params包含API所需的输入参数。例如，call_api(Amazon, open)意味着打开亚马逊APP。如果你发现当前指令无法在当前页面上执行，你需要输出no_answer。如果你发现当前指令已完成，你需要输出action_completed。"""

image_path = "./assets/test_action.png"

# 3. Build the message format
messages = [{"type": "image", "value":f"{image_path}"},
            {"type": "text", "value":f"{instruction}"}]

# 4. Inference
response = model.generate(
    message = messages,
)

print(response)