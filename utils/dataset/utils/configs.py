REFERRING_CONFIGS = { }

GROUNDING_CONFIGS = {
        'ScreenSpot_v2_mobile': {
            'path': '',
            'type': 'GROUNDING',
            'prompt_template': '根据英语短语的指令要求，找到屏幕截图中能够完成指令要求的控件位置，返回对应的bbox框，格式如<|box_start|>(669, 515),(902, 538)<|box_end|>。当前指令为：{query}',
            'need_prompt_template': 'always'
        },
        'Os-Atlas-mobile': { 
            'path': '',
            'type': 'GROUNDING',
            'prompt_template': '<image>\nAccording to the following instruction or description, find the object in the figure that can complete the instructions or meet the description, and return the corresponding bbox, the format is such as <|box_start|>(56, 114),(250, 123)<|box_end|>. Instructions or description: {query}',
            'need_prompt_template': 'always'
        },
    }

NAVIGATION_CONFIGS = {
    'Complex':{
        'path': '',
        'type': 'NAVIGATION',
        'prompt_template': '已知用户在界面<image>，提出了要求\n\"{query}"\n你认为合理的单步操作是什么？除了函数调用之外，你不能输出任何其他内容。你可以调用以下函数来控制智能手机：\nUI基础操作：\n1.tap(x,y) 该函数用于在智能手机屏幕上点击特定点，坐标 x 和 y 表示待点击控件中心位置。\n2.scroll(x,y,direction) 该函数用于从起始坐标 (x,y) 开始在智能手机屏幕上滑动操作，direction为手指滑动的方向，可以是 "up"、"down"、"left" 或 "right"。\n3.text(x,y,text_input) 该函数用于在智能手机屏幕上输入指定的文本text_input。坐标 x 和 y 表示待点击控件的中心位置。\n手机按键操作：\n4. navigate_back() 该函数用于返回智能手机的上一个屏幕。\n5. navigate_home() 该函数用于返回手机的home screen。\n其他操作：\n6. long_press(x,y) 该函数用于在智能手机屏幕上的特定点执行长按操作。坐标 x 和 y 表示待点击控件的中心位置。\n7. wait() 该函数表示在当前页面等候。\n8. enter() 该函数表示按下enter键。\n9. take_over(message) 该函数用于提示用户接管智能手机，其中 message 是提示用户接管手机的原因。如果原因不确定，请填写“请您接管当前界面”。\n10. drag(x1,y1,x2,y2) 该函数执行一个对起始和终点敏感的拖动操作，表示手指从点(x1,y1)拖到点(x2,y2)。常见的场景包括滑块拖动、滚动选择器拖动和图片裁剪。\n11. screen_shot() 该函数用于截图。\n12. long_screen_shot() 该函数用于长截图。\n13. call_api(api_name,operation) 对指定的APP进行操作。api_name是API的名称。operation可以选择open或者kill。例如，call_api(Amazon, open)意味着打开亚马逊APP。\n如果你发现当前指令无法在当前页面上执行，你需要输出no_answer()。如果你发现当前指令已完成，你需要输出action_completed()。',
        'need_prompt_template': 'always'
    },

    'Routine':{
        'path': '',
        'type': 'NAVIGATION',
        'prompt_template': '已知用户在界面<image>，提出了要求\n\"{query}"\n你认为合理的单步操作是什么？除了函数调用之外，你不能输出任何其他内容。你可以调用以下函数来控制智能手机：\nUI基础操作：\n1.tap(x,y) 该函数用于在智能手机屏幕上点击特定点，坐标 x 和 y 表示待点击控件中心位置。\n2.scroll(x,y,direction) 该函数用于从起始坐标 (x,y) 开始在智能手机屏幕上滑动操作，direction为手指滑动的方向，可以是 "up"、"down"、"left" 或 "right"。\n3.text(x,y,text_input) 该函数用于在智能手机屏幕上输入指定的文本text_input。坐标 x 和 y 表示待点击控件的中心位置。\n手机按键操作：\n4. navigate_back() 该函数用于返回智能手机的上一个屏幕。\n5. navigate_home() 该函数用于返回手机的home screen。\n其他操作：\n6. long_press(x,y) 该函数用于在智能手机屏幕上的特定点执行长按操作。坐标 x 和 y 表示待点击控件的中心位置。\n7. wait() 该函数表示在当前页面等候。\n8. enter() 该函数表示按下enter键。\n9. take_over(message) 该函数用于提示用户接管智能手机，其中 message 是提示用户接管手机的原因。如果原因不确定，请填写“请您接管当前界面”。\n10. drag(x1,y1,x2,y2) 该函数执行一个对起始和终点敏感的拖动操作，表示手指从点(x1,y1)拖到点(x2,y2)。常见的场景包括滑块拖动、滚动选择器拖动和图片裁剪。\n11. screen_shot() 该函数用于截图。\n12. long_screen_shot() 该函数用于长截图。\n13. call_api(api_name,operation) 对指定的APP进行操作。api_name是API的名称。operation可以选择open或者kill。例如，call_api(Amazon, open)意味着打开亚马逊APP。\n如果你发现当前指令无法在当前页面上执行，你需要输出no_answer()。如果你发现当前指令已完成，你需要输出action_completed()。',
        'need_prompt_template': 'always'
    },

    'Instruction':{
        'path': '',
        'type': 'NAVIGATION',
        'prompt_template': '已知用户在界面<image>，提出了要求\n\"{query}"\n你认为合理的单步操作是什么？除了函数调用之外，你不能输出任何其他内容。你可以调用以下函数来控制智能手机：\nUI基础操作：\n1.tap(x,y) 该函数用于在智能手机屏幕上点击特定点，坐标 x 和 y 表示待点击控件中心位置。\n2.scroll(x,y,direction) 该函数用于从起始坐标 (x,y) 开始在智能手机屏幕上滑动操作，direction为手指滑动的方向，可以是 "up"、"down"、"left" 或 "right"。\n3.text(x,y,text_input) 该函数用于在智能手机屏幕上输入指定的文本text_input。坐标 x 和 y 表示待点击控件的中心位置。\n手机按键操作：\n4. navigate_back() 该函数用于返回智能手机的上一个屏幕。\n5. navigate_home() 该函数用于返回手机的home screen。\n其他操作：\n6. long_press(x,y) 该函数用于在智能手机屏幕上的特定点执行长按操作。坐标 x 和 y 表示待点击控件的中心位置。\n7. wait() 该函数表示在当前页面等候。\n8. enter() 该函数表示按下enter键。\n9. take_over(message) 该函数用于提示用户接管智能手机，其中 message 是提示用户接管手机的原因。如果原因不确定，请填写“请您接管当前界面”。\n10. drag(x1,y1,x2,y2) 该函数执行一个对起始和终点敏感的拖动操作，表示手指从点(x1,y1)拖到点(x2,y2)。常见的场景包括滑块拖动、滚动选择器拖动和图片裁剪。\n11. screen_shot() 该函数用于截图。\n12. long_screen_shot() 该函数用于长截图。\n13. call_api(api_name,operation) 对指定的APP进行操作。api_name是API的名称。operation可以选择open或者kill。例如，call_api(Amazon, open)意味着打开亚马逊APP。\n如果你发现当前指令无法在当前页面上执行，你需要输出no_answer()。如果你发现当前指令已完成，你需要输出action_completed()。',
        'need_prompt_template': 'always'
    },

    'Handling_Exception':{
        'path': '',
        'type': 'NAVIGATION',
        'prompt_template': '已知用户在界面<image>，提出了要求\n\"{query}"\n你认为合理的单步操作是什么？除了函数调用之外，你不能输出任何其他内容。你可以调用以下函数来控制智能手机：\nUI基础操作：\n1.tap(x,y) 该函数用于在智能手机屏幕上点击特定点，坐标 x 和 y 表示待点击控件中心位置。\n2.scroll(x,y,direction) 该函数用于从起始坐标 (x,y) 开始在智能手机屏幕上滑动操作，direction为手指滑动的方向，可以是 "up"、"down"、"left" 或 "right"。\n3.text(x,y,text_input) 该函数用于在智能手机屏幕上输入指定的文本text_input。坐标 x 和 y 表示待点击控件的中心位置。\n手机按键操作：\n4. navigate_back() 该函数用于返回智能手机的上一个屏幕。\n5. navigate_home() 该函数用于返回手机的home screen。\n其他操作：\n6. long_press(x,y) 该函数用于在智能手机屏幕上的特定点执行长按操作。坐标 x 和 y 表示待点击控件的中心位置。\n7. wait() 该函数表示在当前页面等候。\n8. enter() 该函数表示按下enter键。\n9. take_over(message) 该函数用于提示用户接管智能手机，其中 message 是提示用户接管手机的原因。如果原因不确定，请填写“请您接管当前界面”。\n10. drag(x1,y1,x2,y2) 该函数执行一个对起始和终点敏感的拖动操作，表示手指从点(x1,y1)拖到点(x2,y2)。常见的场景包括滑块拖动、滚动选择器拖动和图片裁剪。\n11. screen_shot() 该函数用于截图。\n12. long_screen_shot() 该函数用于长截图。\n13. call_api(api_name,operation) 对指定的APP进行操作。api_name是API的名称。operation可以选择open或者kill。例如，call_api(Amazon, open)意味着打开亚马逊APP。\n如果你发现当前指令无法在当前页面上执行，你需要输出no_answer()。如果你发现当前指令已完成，你需要输出action_completed()。',
        'need_prompt_template': 'always'
    },

    'AC-Low':{
        'path': '',
        'type': 'NAVIGATION',
        'prompt_template': "You are a well-trained mobile intelligent agent capable of assisting users with step-by-step navigation tasks. Given the current smartphone screenshot <image> and the user instruction:\nInstruction: {query}\n\nPlease output the correct function call to accomplish the user instruction. Besides the function call, you should not output any other content.\nYou can call the following functions to control the smartphone.\n- UI Basic Operations:\n    1. tap(x: float,y: float) This function is used to click on a specific point on the smartphone screen. The coordinates x and y indicate the click position.  \n    2. scroll(x: float, y: float,direction: str) This function is used to swipe from the starting coordinates (x, y) in the specified direction. The coordinates x and y represent the center position of the control to be swiped. The direction can be \"up\", \"down\", \"left\", or \"right\".\n    3. text(x: float,y: float,text_input: str) This function is used to input the specified text at the given coordinates. The coordinates x and y represent the center position of the control to be clicked.\n- Phone Key Operations:\n    4. navigate_back() This function is used to return to the previous screen on the smartphone.\n    5. navigate_home() This function is used to return to the home screen of the phone.\n- Other Operations:\n    6. long_press(x: float,y: float) This function is used to perform a long press action at a specific point on the smartphone screen. The coordinates x and y indicate the long press position.\n    7. wait() This function is to wait at current page.\n    8. finish() The user task is finished.\n    ",
        'need_prompt_template': 'always'
    },

    'AC-High':{
        'path': '',
        'type': 'NAVIGATION',
        'prompt_template': "You are a well-trained mobile intelligent agent capable of assisting users with step-by-step navigation tasks. Given the current smartphone screenshot <image> and the user instruction:\nInstruction: {query}\n\nPlease output the correct function call to accomplish the user instruction. Besides the function call, you should not output any other content.\nYou can call the following functions to control the smartphone.\n- UI Basic Operations:\n    1. tap(x: float,y: float) This function is used to click on a specific point on the smartphone screen. The coordinates x and y indicate the click position.  \n    2. scroll(x: float, y: float,direction: str) This function is used to swipe from the starting coordinates (x, y) in the specified direction. The coordinates x and y represent the center position of the control to be swiped. The direction can be \"up\", \"down\", \"left\", or \"right\".\n    3. text(x: float,y: float,text_input: str) This function is used to input the specified text at the given coordinates. The coordinates x and y represent the center position of the control to be clicked.\n- Phone Key Operations:\n    4. navigate_back() This function is used to return to the previous screen on the smartphone.\n    5. navigate_home() This function is used to return to the home screen of the phone.\n- Other Operations:\n    6. long_press(x: float,y: float) This function is used to perform a long press action at a specific point on the smartphone screen. The coordinates x and y indicate the long press position.\n    7. wait() This function is to wait at current page.\n    8. finish() The user task is finished.\n    ",
        'need_prompt_template': 'always'
    },

    'GUI-Odyssey':{
        'path': '',
        'type': 'NAVIGATION',
        'prompt_template': "You are a well-trained mobile intelligent agent capable of assisting users with step-by-step navigation tasks. Given the current smartphone screenshot <image> and the user instruction:\nInstruction: {query}\n\nPlease output the correct function call to accomplish the user instruction. Besides the function call, you should not output any other content.\nYou can call the following functions to control the smartphone.\n- UI Basic Operations:\n    1. tap(x: float,y: float) This function is used to click on a specific point on the smartphone screen. The coordinates x and y indicate the click position.  \n    2. scroll(x: float, y: float,direction: str) This function is used to swipe from the starting coordinates (x, y) in the specified direction. The coordinates x and y represent the center position of the control to be swiped. The direction can be \"up\", \"down\", \"left\", or \"right\".\n    3. text(x: float,y: float,text_input: str) This function is used to input the specified text at the given coordinates. The coordinates x and y represent the center position of the control to be clicked.\n- Phone Key Operations:\n    4. navigate_back() This function is used to return to the previous screen on the smartphone.\n    5. navigate_home() This function is used to return to the home screen of the phone.\n- Other Operations:\n    6. long_press(x: float,y: float) This function is used to perform a long press action at a specific point on the smartphone screen. The coordinates x and y indicate the long press position.\n    7. wait() This function is to wait at current page.\n    8. finish() The user task is finished.\n    ",
        'need_prompt_template': 'always'
    },
}

VQA_CONFIGS = {
        'ScreenQA-short': {
            'path': '',
            'eval_method': 'LLM',
            'type': 'VQA',
            'prompt_template': 'Please answer the question based on the provided screenshot, if unable to answer based on the screenshot, output <no answer>. question: {query}',
            'need_prompt_template': 'always',
            'eval_examples': [
                {
                    'user': '{"question": "Please answer the question based on the provided screenshot, if unable to answer based on the screenshot, output <no answer>. question: What is the time on the screen?", "ground_truth": "The time on the screen is 12:45", "predict_result": "12:45"}',
                    'assistant': '{"reason": "the time from the predict_result is the same as the ground_truth", "score": 5}',
                },
                {
                    'user': '{"question": "Please answer the question based on the provided screenshot, if unable to answer based on the screenshot, output <no answer>. question: What is the total number of given images?", "ground_truth": "The total number of given images is 5", "predict_result": "4"}',
                    'assistant': '{"reason": "the predict_result is different from the ground_truth", "score": 0}',
                },
                {
                    'user': '{"question": "Please answer the question based on the provided screenshot, if unable to answer based on the screenshot, output <no answer>. question: What is the name of the restaurant?", "ground_truth": "The name of the restaurant is LA PETITE CHAISE", "predict_result": "LA PETITE CHEESE"}',
                    'assistant': '{"reason": "the restaurant name from the predict_result has some spelling mistakes", "score": 1}',
                },
                {
                    'user': '{"question": "Please answer the question based on the provided screenshot, if unable to answer based on the screenshot, output <no answer>. question: How many exercises in total are there to do?", "ground_truth": "There are total 12 exrecises to do.", "predict_result": "12"}',
                    'assistant': '{"reason": "the number from the predict_result is the same as the ground_truth", "score": 5}',
                },
                {
                    'user': '{"question": "Please answer the question based on the provided screenshot, if unable to answer based on the screenshot, output <no answer>. question: What is the name of the given next exercise?", "ground_truth": "The name of the next exercise is "WALL SIT".", "predict_result": "<no answer>"}',
                    'assistant': '{"reason": "The predict_result is <no answer>, which means the system couldn\'t extract the information from the screenshot. However, based on the ground truth, there is a clear name of the next exercise.", "score": 0}',
                },
            ]
        }
    }