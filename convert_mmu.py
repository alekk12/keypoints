import json
import math
import os
import numpy as np
import pandas as pd

coco_map = { "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
             "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
               "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
               "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
               }

def euclidean(p1:tuple, p2:tuple)->float:
    """find euclidean distance between two points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def midpoint(p1:tuple, p2:tuple)->tuple: 
    """Compute midpoint between two points"""
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

def z_score(data, th:float=0.3, win_size:int=3):
    """Smooth the keypoints if confidence of detection is less than th.
    Smooth over win_size windows (except the one with poor detection).
    Each detection has 17 keypoints or is filled with 0 to have it
    

    Args:
        data (_type_): input keypoints data
        th (float, optional): threshold. Defaults to 0.3.
        win_size (int, optional): window length. Defaults to 3.

    Returns:
        _type_: smoothed data
    """
    if win_size < 3:
        print("Window not long enough for smoothing")
        return data

    n = len(data)
    half = win_size//2
    for i in range(0, n):
        slices = list(range(max(0,i-half),min(n,i+half+1)))
        if len(slices) < 2:
            continue
        slices.remove(i)
        curr = data[i]['keypoints']
        for idx in range(0,len(curr),3):
            if curr[idx+2] < th:
                curr[idx:idx+3] = np.mean(
                    np.array([data[j]['keypoints'][idx:idx+3]
                         for j in slices]), axis=0)                            
        data[i]['keypoints'] = curr
    return data

def extract_kp(kpts, name:str, th:int = 0 )->tuple|None:
    """Return kptd """
    idx = coco_map[name]
    x, y, conf = kpts[idx*3: idx*3+3]
    return (x, y) if conf > th else None

def get_closer(kpt1:tuple, kpt2:tuple, orig:tuple = (0,0))->tuple:
    """Get keypoint closer to origin(0,0) or custom

    Args:
        kpt1 (tuple): _description_
        kpt2 (tuple): _description_
        orig (tuple, optional): fpr comaprison. Defaults to (0,0).

    Returns:
        tuple: closer keypint
    """
    return kpt1 if euclidean(kpt1,orig) < euclidean(kpt2,orig) else kpt2

def calculate_ratios(kpts)->dict:
    """Calculate different body ratios
    others:
    * head_to_shoulder
    * torso_to_shoulder
    * hip_to_shoulder
    * head_to_torso
    * head_to_upper_body
    * nose_to_shoulder_center
    * torso_diagonal_ratio = left_diagonal/right_diagonal
    * shoulder_hip_ratio = shoulder_width/hip_width
    
    from mmu gag paper:
    * head (head length) DE (head, neck) -> nose_to_shoulder_len
    * hand (Upper limb length) DE (shoulder, elbow) + DE (elbow, wrist)
    * bodyu (Upper body length) DE (neck, hip)
    * leg (Lower limb length) DE (hip, knee) + DE (knee, ankle)
    * stature (Total body length) head + bodyu + leg
    * handg (Hand-to-ground) bodyu + leg - hand
    * DE (neck, hip) + DE (hip, knee) + DE (knee, ankle) - DE (shoulder, elbow) - DE (elbow, wrist)

    Args:
        kpts (_type_): _description_

    Returns:
        dict: dictionary with ratios
    """
    nose = extract_kp(kpts, "nose")
    left_shoulder = extract_kp(kpts, "left_shoulder")
    right_shoulder = extract_kp(kpts, "right_shoulder")
    left_elbow = extract_kp(kpts, "left_elbow")
    right_elbow = extract_kp(kpts, "right_elbow")
    left_wrist = extract_kp(kpts, "left_wrist")
    right_wrist = extract_kp(kpts, "right_wrist")
    left_knee = extract_kp(kpts, "left_knee")
    right_knee = extract_kp(kpts, "right_knee")
    left_ankle = extract_kp(kpts, "left_ankle")
    right_ankle = extract_kp(kpts, "right_ankle")
    left_hip = extract_kp(kpts, "left_hip")
    right_hip = extract_kp(kpts, "right_hip")

    required = [nose, left_shoulder, right_shoulder, left_hip, right_hip, left_elbow, right_elbow,
                left_wrist, right_wrist, left_knee, right_knee, left_ankle, right_ankle]
    if any(p is None for p in required):
        return None

    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    nose_to_shoulder_len = euclidean(nose, shoulder_center)
    torso_height = euclidean(hip_center, shoulder_center)
    shoulder_to_hip_len = euclidean(shoulder_center, hip_center)
    nose_to_hip_len = nose_to_shoulder_len+shoulder_to_hip_len
    left_diagonal = euclidean(left_shoulder, right_hip)
    right_diagonal = euclidean(right_shoulder, left_hip)
    torso_diagonal_ratio = left_diagonal/right_diagonal
    torso_diagonal = (left_diagonal+right_diagonal)/2
    shoulder_width = euclidean(left_shoulder, right_shoulder)
    hip_width = euclidean(left_hip, right_hip)
    shoulder_hip_ratio = shoulder_width / hip_width if hip_width != 0 else 0
    left_arm =  euclidean(left_shoulder,left_elbow)
    right_arm = euclidean(right_shoulder,right_elbow)
    ave_arm_len = (left_arm+right_arm)/2
    arms_to_torso_ratio = ave_arm_len/torso_diagonal
    n2sw = nose_to_shoulder_len / shoulder_width if shoulder_width != 0 else 0
    shoulder_width_to_torso_ratio = shoulder_width/(torso_diagonal)
    nose_to_torso_ratio = nose_to_shoulder_len/(torso_diagonal)
    shoulder_width_to_torso_height_ratio = shoulder_width/(torso_height)
    hip_width_to_torso_height_ratio = hip_width/(torso_height)
    nose_to_upper_ratio = nose_to_shoulder_len/nose_to_hip_len
    nose_to_torso_height_ratio = nose_to_shoulder_len/torso_height
    shoulder = get_closer(left_shoulder,right_shoulder)
    elbow = get_closer(left_elbow,right_elbow)
    wrist = get_closer(left_wrist,right_wrist)
    ankle = get_closer(left_ankle,right_ankle)
    knee = get_closer(left_knee,right_knee)
    hip = get_closer(left_hip,right_hip)
    arm = euclidean(shoulder,elbow)+euclidean(elbow, wrist)
    femur = euclidean(hip, knee)
    leg = femur+euclidean(knee,ankle)
    torso_len = euclidean(shoulder,hip)
    head_to_torso = nose_to_shoulder_len/torso_len
    femur_to_torso = femur/torso_len
    leg_to_torso = leg/torso_len
    hand_to_leg = arm/leg

    return {
        "nose_to_shoulder_len": nose_to_shoulder_len,
        "torso_height": torso_height,
        "shoulder_width_to_torso_height_ratio":shoulder_width_to_torso_height_ratio,
        "hip_width_to_torso_height_ratio":hip_width_to_torso_height_ratio,
        "nose_to_upper_ratio":nose_to_upper_ratio,
        "nose_to_hip_len": nose_to_hip_len,
        "arms_to_torso_ratio":arms_to_torso_ratio,
        "nose_to_torso_ratio":nose_to_torso_ratio,
        "shoulder_width_to_torso_ratio":shoulder_width_to_torso_ratio,
        "torso_diagonal": torso_diagonal,
        "shoulder_hip_ratio": shoulder_hip_ratio,
        "torso_diagonal_ratio":torso_diagonal_ratio,
        "nose_to_torso_height_ratio":nose_to_torso_height_ratio,
        #"arm":arm,"bodyu":bodyu,"leg":leg,"stature":stature,"handg":handg,
        "leg_to_torso":leg_to_torso,"n2sw":n2sw,
        "femur_to_torso":femur_to_torso, "hand_to_leg":hand_to_leg,"head_to_torso":head_to_torso
    }

def class_ranges(sw2t:float, s2h:float, h2th:float,f2t:float)->str:
    """Determine class based on ratio

    Args:
        sw2t (float): shoulder width to torso diagonal ratio
        s2h (float):shoulder to hip ratio
        h2th (float):hip to torso height ratio
        f2t (float):femur to torso diagonal ratio

    Returns:
        str: class adult|child|ambigous
    """
    ch = (
        0.0 <= sw2t <= 0.970 and
        1.023 <= s2h <= 2.066 and
        0.001 <= h2th <= 0.683 and
        0.452 <= f2t <= 0.966 
    )
    ad = (
        0.0 <= sw2t <= 1.147 and
        0.658 <= s2h <= 2.202 and
        0.0 <= h2th <= 0.798 and
        0.517 <= f2t <= 0.874
    )

    if ch and not ad:
        return 'child'
    elif ad and not ch:
        return 'adult'
    else:
        return 'ambigous'

def classify_body(ratios:dict)->str:    
    """ Child vs Adult, ambigous if other
    
    Args:
        ratios (dict): dictionary with all ratios

    Returns:
        str: 'adult'|'child'|''
    """
    s2h = ratios["shoulder_hip_ratio"]
    sw2t = ratios["shoulder_width_to_torso_ratio"]
    h2th = ratios["hip_width_to_torso_height_ratio"]
    f2t = ratios["femur_to_torso"]
    return class_ranges(sw2t, s2h, h2th,f2t)

def describe_iqr(df:pd.DataFrame, gr_col:list)->pd.DataFrame:
    """Add iqr, lower/upper limit good for outliers

    Args:
        df (pd.DataFrame): _description_
        gr_col (list): _description_

    Returns:
        pd.DataFrame: _description_
    """
    ndf = df.select_dtypes(include='number').columns
    def _describe(group):
        desc = group[ndf].describe()
        iqr = desc.loc['75%']-desc.loc['25%']
        desc.loc['low_b'] = desc.loc['25%']-1.5*iqr
        desc.loc['up_b'] = desc.loc['75%']+1.5*iqr 
        return desc
    return df.groupby(gr_col).apply(_describe)

def detect_age_captions_mmu(f:str)->str:
    """Detect age bases on captions in mmu

    Args:
        f (str): filename

    Returns:
        str: 'adult'|'child'|''
    """
    real = 0
    age = 'ambigous'
    if f.startswith('subject'):
        real = int(f.split('_')[3])
    elif f.startswith('alphapose_'):
        real = int(f.split('_')[1])
    cond1 = 'adult' in f or (real > 18 and real < 60)
    cond2 = 'toddler' in f or ('child' in f and real >= 2 and real < 6)
    if cond1:
        age = 'adult'
    elif cond2:
        age = 'child'
    return age

if __name__ == "main":
    input_dir = "./data/MMU GAG Dataset/Gait in the Wild Dataset/"
    input_dir = "./data/MMU GAG Dataset/Self-Collected Dataset/"
    json_files = [os.path.join(input_dir,f) for f in os.listdir(input_dir)]

    ages = ["adult","child"]

    results = []
    th = 0.3
    win_len = 3
    n = len(json_files)
    for p in json_files:
        with open(p, "r") as f:
            data = json.load(f)
        data = z_score(data, th,win_len)
        filename = p.lower().split("/")[-1].replace('.json','').replace('.mp4','')
        age = detect_age_captions_mmu(filename)
        if age not in ages:
            continue
        for ann in data:
            kpts = ann["keypoints"]
            row = calculate_ratios(kpts)
            if row is not None: #exclude missing
                label = classify_body(row)
                row.update({
                    "age": age,
                    "filename_id": filename,
                    "image_id": ann["image_id"],
                    "label": label,
                    "label_matches_age" : age == label
                })
                results.append(row)

    df = pd.DataFrame(results)
    print(df['label_matches_age'].value_counts())
    df.to_csv(f"./data/label_scd_{th}_{win_len}.csv",index=False)
    df = describe_iqr(df,['age']).reset_index(level=1).round(3)
    df.to_csv(f"./data/ratios_scd_{th}_{win_len}.csv",index=False)