import os
import json
import pandas as pd

def convert_xlsx_to_jsonl(xlsx_file):
    df = pd.read_excel(xlsx_file)

    if not {"images", "prediction"}.issubset(df.columns):
        raise ValueError("Excel must include 'images' 和 'prediction' 两列")

    jsonl_file = os.path.splitext(xlsx_file)[0] + ".jsonl"
    fd = os.open(jsonl_file, os.O_WRONLY | os.O_CREAT, 0o600)  
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            image_path = row["images"]
            image_name = os.path.basename(image_path)

            item = {
                "images": [{"path": image_name}],
                "response": row["prediction"]
            }

            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + "\n")

    print(f"Saved to {jsonl_file}")
    return jsonl_file


def convert_to_template(template_path, input_file, outout_file):
    f = open(template_path).readlines()

    input_list = []
    for item in f:
        item = json.loads(item)
        input_list.append(item)

    compare_list = {}

    f = open(input_file).readlines()
    for item in f:
        item = json.loads(item)
        compare_list[item["images"][0]["path"][:-2]] = item["response"]

    for item in input_list:
        img = item["image_path"]
        response = compare_list[img]

        if "before_trans" in item:
            before_trans = item["before_trans"]
            item["before_trans"] = response
            item["generate_content"] = item["generate_content"].replace(before_trans, response)

        result = {"STATUS": "continue"}
        
        # auxiliary function to extract coordinates
        def extract_coords(s):
            # directly find and extract the coordinates in the parentheses
            start = s.find("(")
            end = s.find(")")
            if start != -1 and end != -1:
                coords_str = s[start+1:end].strip()  # extract the content in (x,y)
                x, y = coords_str.split(",")
                return [int(x), int(y)]
            raise ValueError(f"Cannot find coordinates in the string: {s}")
        
        if "tap(" in response:
            result["POINT"] = extract_coords(response)
            
        elif "long_press(" in response:
            result["POINT"] = extract_coords(response)
            result["duration"] = 1000
                
        elif "text(" in response:
            content = response.split(",")[-1][:-1]
            result["TYPE"] = content
            
        elif "scroll(" in response:
            direction = response.split(",")[-1][:-1]
            result["POINT"] = [0, 0]  # screen center point
            result["to"] = direction
        elif "navigate_back" in response:
            result["PRESS"] = "BACK"

        elif "navigate_home" in response:
             result["PRESS"] = "HOME"

        elif "wait()" in response:
            result["duration"] = 500
    
        elif "finish()" in response or "Finish()" in response:
            result["STATUS"] = "finish"

        else:
            # print(response)
            pass

        item["pred"] = result

    outout_path = "/".join(outout_file.split("/")[:-1])
    if not os.path.exists(outout_path):
        os.makedirs(outout_path)

    fd = os.open(outout_file, os.O_WRONLY | os.O_CREAT, 0o600)  
    with os.fdopen(fd, 'w', encoding='utf-8') as f:    
        for item in input_list:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')