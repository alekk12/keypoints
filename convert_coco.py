import collections
import json
import pandas as pd

def detect_age(captions:list, 
               keyword_list:list=['child ','little boy ','little girl ','small boy ','small girl ','infant ','toddler '], 
               adult_keywords:list = ['adult ','person ','woman ','man ']) -> str:
    """Detect age based on captions. There are usually multiple captions per image. 
    Images with groups are skipped (multiple detections are excluded in this version
    since we do not know who is being described in caption - no caption per detection.

    Args:
        captions (list): list of captions
        keyword_list (list, optional): Child's keyword list. Defaults to ['child ','little boy ','little girl ','small boy ','small girl ','infant ','toddler '].
        adult_keywords (list, optional): Adult's keyword list. Defaults to ['adult ','person ','woman ','man '].

    Returns:
        str: return age label
    """
    for caption in captions:
        if sum(bool(n in caption) for n in keyword_list)==1:
            return 'child'
        elif sum(bool(n in caption) for n in adult_keywords)==1:
            return 'adult'
    return 'ambigous' 

def create_label_csv(type_input:str = 'train', jsonfile:str = '', capfile:str = '',
                     out_path:str = '', ages:list = ["adult","child"]):
    """Create a csv file with labels

    Args:
        type_input (str, optional): train|val. Defaults to 'train'.
        jsonfile (str, optional): annotations. Defaults to ''.
        capfile (str, optional): captions. Defaults to ''.
        out_path (str, optional): csv output path. Defaults to ''.
        ages (list, optional): list of acceptable ages. Defaults to ["adult","child"].
    """
    
    jsonfile = jsonfile if len(jsonfile) > 0 else f"./annotations_trainval2017/annotations/person_keypoints_{type_input}2017.json"
    capfile = capfile if len(jsonfile) > 0 else f"./annotations_trainval2017/annotations/captions_{type_input}2017.json"
    out_path = out_path if len(jsonfile) > 0 else f"./label_coco_{type_input}.csv"

    results = []

    with open(jsonfile, "r") as f:
            ann = json.load(f)
    with open(capfile, "r") as f:
            cap = json.load(f)

    data = collections.defaultdict(list)

    for item in ann['annotations']:
        key = item['image_id']
        if key in data.keys():
            del data[key]
        else:
            data[key] = item
            data[key]['captions'] = []

    for item in cap['annotations']:
        key = item['image_id']
        if key in data.keys() and item['caption']:
            data[key]['captions'].append(item['caption'])

    for _,p in data.items():    
        age = detect_age(p['captions'])
        if age not in ages:
            continue
        row = {
            "age": age,
            "image_id": p["image_id"],
            "id": p["id"]
        }
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(out_path,index=False)
    return