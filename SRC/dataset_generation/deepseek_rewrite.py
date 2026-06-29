import requests
import json
import re
from argparse import ArgumentParser
from pathlib import Path


GEN_PROMPT = """
You task is to rewrite image and video descriptions. Also you should write image and video objects, that are presented in the caption and no-exist image and video objects, that are not presented, but are close to presented one.
1. Your image description should be rewritten for using Stable Diffusion image generation model on the given caption. Remove ANY specific names/terms if they exist.
2. Your video description should be rewritten for using video generation model on the given caption. Remove ANY specific names/terms if they exist.
3. Your image and video objects must include ONLY the concrete objects that are explicitly mentioned and actually present in your caption rewrites (e.g., if the caption mentions 'no XXX', do not include 'XXX' in the object's list).
4. The no-exist image and video objects list must include concrete objects that are not present in the caption but could commonly occur in similar scenes (e.g., train => railroad).
5. Your image objects, video objects, no-exist image objects and no-exist video objects must include at least one object.
5. Your image objects, video objects, no-exist image objects and no-exist video objects must be separated with ';'.
6. Your image objects, video objects, no-exist image objects and no-exist video objects must include ONLY concrete objects (e.g., leaves, windowsill) and AVOID abstract/invisible concepts (e.g., season, color, action).
7. The output should be in English only.

Here is the image caption: {img_desc}, and here is the video caption: {vid_desc}.
Please strictly follow the instructions.

Provide your answer.
"New image caption": ,
"Image objects": ,
"No-exist image objects": ,
"New video caption": ,
"Video objects": ,
"No-exist video objects": ,
"""

def parse_args() -> tuple[Path, Path]:
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=Path, required=True, help="Path to .json with video descriptions")
    parser.add_argument("--save_path", type=Path, required=True, help="Path to save the result .json file")
    parser.add_argument("--url", type=str, required=True, help="URL to /api/chat/completions")
    parser.add_argument("--key", type=str, required=True, help="secret key for chat")

    args = parser.parse_args()
    return args.data_path, args.save_path

def post_deepseek(text: str, url: str, key: str) -> str:
    url = url
    headers = {
        'Authorization': f'Bearer {key}',
        'Content-Type': 'application/json'
    }
    data = {
        "model":"deepseek-v3-0324",
        "messages": [{"role": "user", "content": text}],
        "stream": False, # Example query
        "parameters": { # Optional, depending on API
            "filter": "relevant_filter_value",
            "count": 5
    }}
    response = requests.post(url, headers=headers, json=data)
    try:
        text = json.loads(response.text)['choices'][0]['message']['content']
    except:
        print(response)
        return ""
    return text


def ask_deepseek(img: dict, url: str, key: str) -> str:
    query = GEN_PROMPT.format(img_desc=img["img_desc"], vid_desc=img["vid_desc"])
    text = post_deepseek(query, url, key)
    return text

def parse_model_output_v2(text: str) -> dict:
    result = {
        'new_image_caption': '',
        'image_objects': [],
        'no_exist_image_objects': [],
        'new_video_caption': '',
        'video_objects': [],
        'no_exist_video_objects': []
    }
    
    # Remove asterisks and normalize
    clean_text = re.sub(r'\*+', '', text)
    
    # Split by known field names and extract content
    fields = [
        ('New image caption:', 'new_image_caption', False),
        ('Image objects:', 'image_objects', True),
        ('No-exist image objects:', 'no_exist_image_objects', True),
        ('New video caption:', 'new_video_caption', False),
        ('Video objects:', 'video_objects', True),
        ('No-exist video objects:', 'no_exist_video_objects', True)
    ]
    
    for i, (field_marker, dict_key, is_list) in enumerate(fields):
        # Find the field marker
        pattern = re.escape(field_marker) + r'\s*(.*?)(?='
        
        # Add lookaheads for next markers
        next_markers = [re.escape(marker) for marker, _, _ in fields[i+1:]]
        if next_markers:
            pattern += '|'.join(next_markers) + '|$)'
        else:
            pattern += '$)'
        
        match = re.search(pattern, clean_text, re.DOTALL | re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if is_list:
                if value:
                    result[dict_key] = [item.strip() for item in value.split(';') if item.strip()]
                else:
                    result[dict_key] = []
            else:
                result[dict_key] = value
    
    return result

def infer():
    args = parse_args()
    data_path = args.data_path
    save_path = args.save_path
    with data_path.open(encoding="utf-8") as file:
        data = json.load(file)
    deepseek_response = []
    if save_path.exists():
        with save_path.open(encoding="utf-8") as file:
            deepseek_response = json.load(file)

    for i in range(len(deepseek_response)):
        row = deepseek_response[i]
        if not row["new_image_caption"]:
            # 5 retries per sample
            for _ in range(5):
                text = ask_deepseek(data[i], args.url, args.key)
                if text:
                    break
            result = parse_model_output_v2(text)
            result["category"] = row["category"]
            result["orig_path"] = row["orig_path"]
            if not result["new_image_caption"]:
                result["ans"] = text
            deepseek_response[i] = result

        # partial save
        if i % 10 == 0:
            with save_path.open("w", encoding="utf-8") as file:
                json.dump(deepseek_response, file, indent=4)
    with save_path.open("w", encoding="utf-8") as file:
        json.dump(deepseek_response, file, indent=4)

if __name__ == '__main__':
    infer()
