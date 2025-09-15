cn_en_map = {
    "是否正常": "Whether Normal",
    "畸形类别": "Type of Deformity",
    "不合理原因分析-元素属性不合理": "L2: Irrational Element Attributes",
    "不合理原因分析-元素交互关系不合理": "L2: Irrational Element Interaction",
    "不合理原因分析-人体解剖结构异常": "L2: Abnormal Human Anatomy",
    "不合理原因分析-动物解剖结构异常": "L2: Abnormal Animal Anatomy",
    "不合理原因分析-物体形态异常": "L2: Abnormal Object Morphology",
    "不合理原因分析-其他不合理": "L2: Other Irrationalities",

    "材质质感异常": "L3: Abnormal Material Texture",
    "细节描绘异常": "L3: Abnormal Detail Drawing",
    "元素比例异常": "L3: Abnormal Element Proportion",
    "颜色搭配异常": "L3: Abnormal Color Combination",

    "光影效果异常": "L3: Abnormal Light and Shadow Effect",
    "元素重叠异常": "L3: Abnormal Element Overlap",
    "空间位置异常": "L3: Abnormal Spatial Position",

    "四肢结构畸形": "L3: Limb Structure Deformity",
    "躯干结构畸形": "L3: Trunk Structure Deformity",
    "手部结构畸形": "L3: Hand Structure Deformity",
    "足部结构畸形": "L3: Foot Structure Deformity",
    "面部结构畸形": "L3: Facial Structure Deformity",
    "人体解剖结构异常": "L3: Abnormal Human Anatomy",
    "姿态异常不协调": "L3: Abnormal and Uncoordinated Posture",

    # "四肢结构异常": "L3: Abnormal Limb Structure", 
    "四肢构造异常": "L3: Abnormal Limb Structure",
    "姿态表现异常": "L3: Abnormal Posture Presentation",
    # "头部结构异常": "L3: Abnormal Head Structure",
    "头部构造异常": "L3: Abnormal Head Structure",
}


type_definition_en = {
    "L2: Irrational Element Attributes": {
        "Description": "The visual attributes of elements in the image do not conform to physical laws",
        "Sub-tags": {
            "L3: Abnormal Material Texture": {
                "Description": "The material texture does not match the actual properties of the object, such as metallic texture displaying a wooden pattern"
            },
            "L3: Abnormal Detail Drawing": {
                "Description": "Abnormal background elements in the image"
            },
            "L3: Abnormal Element Proportion": {
                "Description": "The relative sizes of elements in the image do not conform to real proportions or expected scales, such as a mosquito larger than a hand"
            },
            "L3: Abnormal Color Combination": {
                "Description": "Color combination violates visual color theory, leading to a visual appearance that does not conform to the real world"
            }
        }
    },
    "L2: Irrational Element Interaction": {
        "Description": "The spatial and logical interactions between elements in the image are unreasonable",
        "Sub-tags": {
            "L3: Abnormal Light and Shadow Effect": {
                "Description": "The position of the light source and shadow direction are inconsistent, causing unnatural light and shadow projection. The light and shadow effect does not match the light source position, intensity, and objective factors"
            },
            "L3: Abnormal Element Overlap": {
                "Description": "Overlap relationships between different elements do not conform to physical laws, such as a solid object partially penetrating another object"
            },
            "L3: Abnormal Spatial Position": {
                "Description": "The distribution and logical arrangement of elements in space are inconsistent, causing chaotic overall layout, such as floating, mismatch between inside and outside state in a mirror"
            }
        }
    },
    "L2: Abnormal Human Anatomy": {
        "Description": "The structure of the human body in the image does not conform to normal physiological and anatomical standards",
        "Sub-tags": {
            "L3: Limb Structure Deformity": {
                "Description": "Limb structure does not conform to conventional human form"
            },
            "L3: Trunk Structure Deformity": {
                "Description": "The spine shows unnatural curvature or twisting"
            },
            "L3: Hand Structure Deformity": {
                "Description": "Abnormal number of fingers or unreasonable joint angles"
            },
            "L3: Foot Structure Deformity": {
                "Description": "Disorganized toe arrangement or abnormal arch shape"
            },
            "L3: Facial Structure Deformity": {
                "Description": "Imbalance in facial features or lack of facial symmetry"
            },
            "L3: Abnormal Human Anatomy": {
                "Description": "Multiple human abnormalities"
            },
            "L3: Abnormal and Uncoordinated Posture": {
                "Description": "Whole body posture does not conform to gravitational direction or movements are inconsistent with ergonomics"
            }
        }
    },
    "L2: Abnormal Animal Anatomy": {
        "Description": "The structure of animals in the image does not conform to normal physiological and anatomical standards",
        "Sub-tags": {
            "L3: Abnormal Limb Structure": {
                "Description": "Imbalance in animal limb proportions or shape does not conform to common sense"
            },
            "L3: Abnormal Posture Presentation": {
                "Description": "Animal movement posture does not match its biological characteristics"
            },
            "L3: Abnormal Head Structure": {
                "Description": "Abnormal position or imbalance in the proportion of eyes or ears"
            }
        }
    },
    "L2: Abnormal Object Morphology": {
        "Description": "Geometric shape is abnormal, the object outline or geometric proportions do not match actual characteristics; or the construction is unreasonable, the connection method of object parts does not conform to logic or actual structure"
    },
    "L2: Other Irrationalities": {
        "Description": "Other irrationalities"
    }
}

label_rule_en = [
    "1. When judging, if an image corresponds to multiple issues, only the two most obvious issues need to be marked. However, there are two exceptions: if the number of people in the image is ≥3 and the number of abnormal issues is ≥3, you can simply label it as \"L3: Abnormal Human Anatomy\"; if a single person has  more than 3 abnormal issues, you can directly label it as \"L3: Abnormal Human Anatomy\".",
    "2. During the annotation process, only more obvious abnormalities need to be noted. If it cannot be identified within 3 seconds (it may be a blurred background entity or an entity with small proportion), if a reasonable explanation can be found, or it belongs to an imaginable special case, it is not considered abnormal, you can label it as \"Whether Normal: True\".",
    "3. Instantly fake or heavy AI feeling belongs to image style issues, not generated abnormal issues, so you can label it as \"Whether Normal: True\".",
    "4. If there is a text error in the image, do not judge it as unreasonable based on the text.",
    "5. When some images initially appear not to conform to the objective laws of the physical world, it is necessary to refer to the user's prompt style requirements to assist in judgment. Compare whether the image is consistent with the prompt style requirements. If it conforms to the special circumstances specified by the prompt, you can label it as \"Whether Normal: True\"."
]


answer_example_en = [
    {"Whether Normal": True},
    {"Whether Normal": False, "Type of Abnormality": {"L2: Abnormal Object Morphology": True}},
    {"Whether Normal": False, "Type of Abnormality": {"L2: Abnormal Human Anatomy": ["L3: Abnormal Human Anatomy"]}},
    {"Whether Normal": False, "Type of Abnormality": {"L2: Abnormal Object Morphology": True, "L2: Abnormal Human Anatomy": ["L3: Abnormal Human Anatomy"]}}
]


think_example = '''
Okay, let's start by analyzing the image based on the given prompt and the deformity categories. The prompt describes a young Asian woman in a ballet pose in a dance studio with natural light from large windows, mirrors, and colorful attire hanging on the walls.

First, I'll check for any abnormalities in human anatomy. The dancer's posture looks challenging but possible for a trained ballet dancer. However, upon closer inspection, her leg position might be unrealistic. In the image, her extended leg is lifted high, but the angle and position of her knee and foot seem unnatural. The foot appears to be twisted in a way that doesn't align with typical ballet poses, which could indicate a deformity in limb structure or abnormal posture. This would fall under L2: Abnormal Human Anatomy, specifically L3: Limb Structure Deformity or L3: Abnormal and Uncoordinated Posture.

Next, looking at the lighting and shadows. The prompt mentions natural light streaming through large windows, casting soft shadows. In the image, the light source seems to come from the windows behind the dancer, but the shadows cast on the floor don't align correctly with the direction of the light. For example, the shadow of her raised arm should be projected towards the opposite side of the light source, but it might not match the actual light direction. This inconsistency in light and shadow effects would be categorized under L2: Irrational Element Interaction, specifically L3: Abnormal Light and Shadow Effect.

Additionally, checking other elements: the mirrors reflect her image, which seems okay. The colorful attire hanging on the walls is present, matching the prompt. The wooden floor and overall setting appear realistic. There are no obvious issues with material textures, element proportions, or color combinations. The main problems are the anatomical issue with her leg and the lighting/shadow mismatch.

Therefore, the image has two types of deformities: Abnormal Human Anatomy (due to the leg position) and Abnormal Light and Shadow Effect. The answer should reflect both L2 categories with their respective L3 sub-tags.
'''
def get_prompt(prompt_ori):
    prompt = ""
    # print(prompt_ori)
    # print(subjects)
    if type(prompt_ori) != str:
        prompt_ori = str(prompt_ori)
    prompt += "\n<image>This is an image generated by a text-to-image model, with the corresponding text prompt as{" + prompt_ori + "}. "
    prompt += "You need to determine whether this image is reasonable (or whether there is any deformity), and if it is not reasonable, provide the corresponding type of deformity. If the provided type of deformity has sub-tags, additionally provide the corresponding sub-tag categories. \n"

    prompt += "All types of deformities and their sub-tags are: " + str(type_definition_en) + ". Note that the primary label is Whether Normal, L2 represents second-level tags, and L3 represents third-level tags. \n"

    prompt += "There are a few example answer formats: \n"
    prompt += "1. If a normal iamge, the answer is " + str(answer_example_en[0]) + ". "
    prompt += "2. If with abnormality, but not sub-tags, format is similar to: " + str(answer_example_en[1]) + ". \n"
    prompt += "3. If with abnormality, and with sub-tags, format is similar to: " + str(answer_example_en[2]) + ". \n"
    prompt += "4. If with two kind of abnormalities, format is similar to: " + str(answer_example_en[3]) + ". \n"

    prompt += "You need to first understand all the given labels and rules, then think about possible issues according to the text prompt and the subject of the prompt, and then observe the image to analyze every detail in the image to determine whether there is any deformity. "
    prompt += "Give a continuous thinking process using natural language. The response should flow seamlessly as a narrative or story, examining the image as a whole rather than in separate points. Please describe the reasoning process without using bullet points or distinct sections. \n"
    
    prompt += "Ensure that the answer matches the format of the given example. "
    prompt += "The output format should be <think>...</think>...\\boxed{answer}."
    # print(len(prompt))
    return prompt


def get_response(data_json):
    if data_json["图像类型"] == "正常图像":
        response = {"是否正常": True}
    elif data_json["图像类型"] == "不合理图像":
        response = {"是否正常": False, 
                    "畸形类别": {}}
        for key, value in data_json.items():
            if "不合理原因分析" in key:
                # print(value)
                if value != {}: # 存在该畸形
                    if key == "不合理原因分析-物体形态异常" or key == "不合理原因分析-其他不合理":
                        response["畸形类别"][key] = True
                    else:
                        sub_keys = [key1 for key1, value1 in value.items()]
                        response["畸形类别"][key] = sub_keys
    else:
        raise ERROR
    return response

def get_response_en(data_json):
    if data_json["图像类型"] == "正常图像":
        response = {cn_en_map["是否正常"]: True}
    elif data_json["图像类型"] == "不合理图像":
        response = {cn_en_map["是否正常"]: False, 
                    cn_en_map["畸形类别"]: {}}
        for key, value in data_json.items():
            if "不合理原因分析" in key:
                # print(value)
                if value != {}: # 存在该畸形
                    if key == "不合理原因分析-物体形态异常" or key == "不合理原因分析-其他不合理":
                        response[cn_en_map["畸形类别"]][cn_en_map[key]] = True
                    else:
                        sub_keys = [cn_en_map[key1] for key1, value1 in value.items()]
                        response[cn_en_map["畸形类别"]][cn_en_map[key]] = sub_keys
    else:
        raise ERROR
    return response




import re
import json

def extract_think_content(text):
    pattern = r'<[^<>]*>(.*?)<[^<>]*>'
    m = re.search(pattern, text, re.DOTALL)
    if m:
        return m.group(1)
    else:
        return None

def extract_and_check_boxed_json(text):
    """
    检查text中是否存在<think>、<\think>和\boxed{}结构，并提取boxed内容，判断是否能json解析。
    返回：(是否全部满足, boxed内容或None, 解析后的json或None)
    """
    # 1. 检查<think>和<\think>
    if "<think>" not in text or "</think>" not in text:
        return False, None, None

    # 2. 提取\boxed{...}
    m = re.search(r"\\boxed\{(.+?)\}", text, re.DOTALL)
    if not m:
        return False, None, None

    boxed_content = m.group(1).strip()
    # 3. 尝试json解析
    try:
        parsed_json = json.loads(boxed_content)
    except Exception:
        return False, boxed_content, None

    return True, boxed_content, parsed_json



union_abnormal_labels = {
    "L2: Irrational Element Attributes" : ["L3: Abnormal Material Texture", "L3: Abnormal Detail Drawing", "L3: Abnormal Element Proportion", "L3: Abnormal Color Combination"],
    "L2: Irrational Element Interaction": ["L3: Abnormal Light and Shadow Effect", "L3: Abnormal Element Overlap", "L3: Abnormal Spatial Position"],
    "L2: Abnormal Human Anatomy": ["L3: Limb Structure Deformity", "L3: Trunk Structure Deformity", "L3: Hand Structure Deformity", "L3: Foot Structure Deformity", "L3: Facial Structure Deformity", "L3: Abnormal Human Anatomy", "L3: Abnormal and Uncoordinated Posture"],
    "L2: Abnormal Animal Anatomy": ["L3: Abnormal Limb Structure", "L3: Abnormal Posture Presentation", "L3: Abnormal Head Structure"],
    "L2: Abnormal Object Morphology": True,
    "L2: Other Irrationalities": True,
}
def calculate_reward(gt, output, union_abnormal_labels=union_abnormal_labels):
    """
    计算GRPO强化学习的奖励值，包括四个维度：格式奖励、二分类奖励、L2标签奖励和L3标签奖励
    所有奖励值均规范化到0-1之间
    
    参数:
        gt (str/dict): 标准答案，JSON字符串或字典
        output (str/dict): 模型输出，JSON字符串或字典
        union_abnormal_labels (dict): 标签层级关系定义
    
    返回:
        dict: 包含各维度奖励值和总奖励的字典
    """
    # 确保输入是字典格式
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except:
            # print(output)
            return {"format": 0, "binary": 0, "l2": 0, "l3": 0, "total": 0}
    
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except:
            raise ValueError("标准答案格式错误")
    
    # 1. 格式奖励
    format_reward = _check_format(output, union_abnormal_labels=union_abnormal_labels)
    
    # 如果格式错误，后续所有奖励均为0
    if format_reward == 0:
        return {"format": 0, "binary": 0, "l2": 0, "l3": 0, "total": 0}
    
    # 2. 二分类奖励
    # print(gt)
    # a = gt["Whether Normal"]
    # print(a)
    binary_reward = 1.0 if output["Whether Normal"] == gt["Whether Normal"] else 0.0
    # if gt["Whether Normal"] == True:
    #     # binary_reward = binary_reward * 0.5
    #     binary_reward = binary_reward * 0.1
    
    # 如果二分类错误，L2和L3奖励均为0
    weights = [8, 4, 2, 1]  # 可调整权重
    if binary_reward == 0:
        return {"format": format_reward, "binary": 0, "l2": 0, "l3": 0, 
                "total": _calculate_weighted_total([format_reward, 0, 0, 0], weights)}
    
    # 3. L2标签奖励
    l2_reward = _calculate_l2_reward(gt, output)
    
    # 4. L3标签奖励
    l3_reward = _calculate_l3_reward(gt, output, l2_reward)
    
    # 5. 计算加权总奖励
    total_reward = _calculate_weighted_total([format_reward, binary_reward, l2_reward, l3_reward], weights)
    
    return {
        "format": format_reward,
        "binary": binary_reward,
        "l2": l2_reward,
        "l3": l3_reward,
        "total": total_reward
    }

def _check_format(output, union_abnormal_labels):
    """检查输出格式是否符合规范，返回0或1"""
    # 首先检查输入类型
    if isinstance(output, dict):
        # 已经是字典类型，无需解析
        pass
    elif isinstance(output, str):
        # 尝试将字符串转换为字典
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            print(f"JSON解析失败: {output[:300]}...")  # 打印前100个字符用于调试
            return 0
    else:
        # 既不是字典也不是字符串
        print(f"输入类型错误: {type(output)}")
        return 0
        
    # 确保现在output是字典类型
    if not isinstance(output, dict):
        print(f"解析后不是字典类型: {type(output)}")
        return 0

    # print(output)
    # print(type(output))
    # 1. 必须有Whether Normal键
    if "Whether Normal" not in output:
        return 0

    # 2. "Whether Normal"为True时，不能有其他key
    if output["Whether Normal"] is True:
        if len(output) != 1:
            return 0
        return 1

    # 3. "Whether Normal"为False时，必须有"Type of Deformity"且只能有这两个key
    if output["Whether Normal"] is False:
        # 必须只有两个key
        if set(output.keys()) != {"Whether Normal", "Type of Deformity"}:
            return 0
        # Type of Deformity必须是字典
        if not isinstance(output["Type of Deformity"], dict):
            return 0
        # 二级标签必须是union_abnormal_labels的子集
        for l2 in output["Type of Deformity"].keys():
            if l2 not in union_abnormal_labels:
                return 0
            # 如果union_abnormal_labels[l2]是True，允许任意内容/空内容
            if union_abnormal_labels[l2] is True:
                continue
            # 否则，l3必须是list且为union_abnormal_labels[l2]的子集
            l3s = output["Type of Deformity"][l2]
            if not isinstance(l3s, list):
                return 0
                
            # 检查l3s中的每个元素是否合法
            for item in l3s:
                if not isinstance(item, str) or item not in union_abnormal_labels[l2]:
                    return 0
        return 1

    # 4. 其他情况
    return 0

def _calculate_l2_reward(gt, output):
    """计算L2标签奖励，范围0-1"""
    # 如果是正常图像，L2奖励为0
    if gt["Whether Normal"] is True:
        return 0.0
    
    gt_types = gt.get("Type of Deformity", {})
    out_types = output.get("Type of Deformity", {})
    
    # 获取标准答案和预测结果中的L2标签集合
    gt_l2_set = set(gt_types.keys())
    out_l2_set = set(out_types.keys())
    
    # 计算正确识别的L2标签数量
    correct_l2 = len(gt_l2_set & out_l2_set)
    
    # 计算错报和漏报的L2标签数量
    missed_l2 = len(gt_l2_set - out_l2_set)
    extra_l2 = len(out_l2_set - gt_l2_set)
    
    # 按照[1.0, +0.6, -0.3]数组计算得分
    if correct_l2 > 0 and missed_l2 == 0 and extra_l2 == 0:
        score = 1.0  # 全部正确
    else:
        score = correct_l2 * 0.6 - (missed_l2 + extra_l2) * 0.3
    
    # 限制分数范围在0-1之间
    return max(0.0, min(1.0, score))

def _calculate_l3_reward(gt, output, l2_reward):
    """计算L3标签奖励，范围0-1"""
    # 如果L2奖励为0，L3奖励也为0
    if l2_reward == 0:
        return 0.0
    
    # 如果是正常图像，L3奖励为0
    if gt["Whether Normal"] is True:
        return 0.0
    
    gt_types = gt.get("Type of Deformity", {})
    out_types = output.get("Type of Deformity", {})
    
    # 只评估正确识别的L2标签下的L3标签
    common_l2 = set(gt_types.keys()) & set(out_types.keys())
    
    # 跟踪需要检查L3的L2标签数量
    l2_with_l3_tags = 0
    correct_l3 = 0
    missed_l3 = 0
    extra_l3 = 0
    
    for l2 in common_l2:
        # 如果这个L2标签不需要L3子标签，跳过
        if union_abnormal_labels[l2] is True:
            continue
            
        l2_with_l3_tags += 1
        
        gt_l3_set = set(gt_types[l2]) if isinstance(gt_types[l2], list) else set()
        out_l3_set = set(out_types[l2]) if isinstance(out_types[l2], list) else set()
        
        # 计算正确识别、错报和漏报的L3标签
        correct_l3 += len(gt_l3_set & out_l3_set)
        missed_l3 += len(gt_l3_set - out_l3_set)
        extra_l3 += len(out_l3_set - gt_l3_set)
    
    # 如果没有需要检查L3的L2标签，L3奖励为0
    if l2_with_l3_tags == 0:
        return 0.0
    
    # 按照[1.0, +0.6, -0.3]数组计算得分
    if correct_l3 > 0 and missed_l3 == 0 and extra_l3 == 0:
        score = 1.0  # 全部正确
    else:
        score = correct_l3 * 0.6 - (missed_l3 + extra_l3) * 0.3
    
    # 限制分数范围在0-1之间
    return max(0.0, min(1.0, score))

def _calculate_weighted_total(rewards, weights=[8, 4, 2, 1]):
    """计算加权总奖励"""
    assert len(rewards) == len(weights), "奖励和权重数量必须相同"
    return sum(r * w for r, w in zip(rewards, weights))













LABEL_COUNT_MAP0= {
        "Whether Normal": 0,
        "\'Whether Normal\': True": 0,
        "Type of Deformity": 0,
        "L2: Irrational Element Attributes": 0,
        "L2: Irrational Element Interaction": 0,
        "L2: Abnormal Human Anatomy": 0,
        "L2: Abnormal Animal Anatomy": 0,
        "L2: Abnormal Object Morphology": 0,
        "L2: Other Irrationalities": 0,
        "L3: Abnormal Material Texture": 0,
        "L3: Abnormal Detail Drawing": 0,
        "L3: Abnormal Element Proportion": 0,
        "L3: Abnormal Color Combination": 0,
        "L3: Abnormal Light and Shadow Effect": 0,
        "L3: Abnormal Element Overlap": 0,
        "L3: Abnormal Spatial Position": 0,
        "L3: Limb Structure Deformity": 0,
        "L3: Trunk Structure Deformity": 0,
        "L3: Hand Structure Deformity": 0,
        "L3: Foot Structure Deformity": 0,
        "L3: Facial Structure Deformity": 0,
        "L3: Abnormal Human Anatomy": 0,
        "L3: Abnormal and Uncoordinated Posture": 0,
        "L3: Abnormal Limb Structure": 0,
        "L3: Abnormal Posture Presentation": 0,
        "L3: Abnormal Head Structure": 0,
    }

from collections import defaultdict
import copy
# 创建两个副本：一个记录真实标签总数，一个记录正确预测的标签数

def count_labels(data, counter, is_gt=True):
    """
    统计标签出现次数
    
    参数:
        data: 标签数据（字典格式）
        counter: 计数器字典
        is_gt: 是否为真实标签（用于区分处理逻辑）
    """
    # 计数基本标签
    counter["Whether Normal"] += 1
    
    if data["Whether Normal"] is True:
        counter["'Whether Normal': True"] += 1
        return  # 如果是正常图像，不需要继续计数其他标签
    
    # 计数Type of Deformity
    counter["Type of Deformity"] += 1
    
    # 计数L2和L3标签
    for l2, l3_list in data.get("Type of Deformity", {}).items():
        counter[l2] += 1
        
        # 如果L3标签是列表，计数每个L3标签
        if isinstance(l3_list, list):
            for l3 in l3_list:
                counter[l3] += 1

def count_correct_labels(gt, pred, counter):
    """
    统计正确预测的标签数量
    
    参数:
        gt: 真实标签数据
        pred: 预测标签数据
        counter: 正确预测计数器
    """
    # 计数基本标签
    if "Whether Normal" in gt and "Whether Normal" in pred:
        if gt["Whether Normal"] == pred["Whether Normal"]:
            counter["Whether Normal"] += 1
            
            if gt["Whether Normal"] is True and pred["Whether Normal"] is True:
                counter["'Whether Normal': True"] += 1
    
    # 如果真实标签是正常图像，不需要继续计数其他标签
    if gt.get("Whether Normal", False) is True:

        if pred.get("Whether Normal", True) is False: # 统计extra标签
            pred_l2 = set(pred.get("Type of Deformity", {}).keys())
            for l2 in pred_l2:
                counter["extra_lable"][l2] += 1

        return
    
    # 如果预测为正常图像.   但实际不是，则其他标签都不正确
    if pred.get("Whether Normal", True) is True:

        if gt.get("Whether Normal", True) is False: # 统计miss标签
            gt_l2 = set(gt.get("Type of Deformity", {}).keys())
            for l2 in gt_l2:
                counter["miss_lable"][l2] += 1

        return
    
    # 计数Type of Deformity
    if "Type of Deformity" in gt and "Type of Deformity" in pred:
        counter["Type of Deformity"] += 1
    
    # 获取L2标签
    gt_l2 = set(gt.get("Type of Deformity", {}).keys())
    pred_l2 = set(pred.get("Type of Deformity", {}).keys())
    
    # 计数正确的L2标签
    for l2 in gt_l2 & pred_l2:
        counter[l2] += 1
        
        # 获取L3标签
        gt_l3 = set(gt["Type of Deformity"][l2]) if isinstance(gt["Type of Deformity"].get(l2), list) else set()
        pred_l3 = set(pred["Type of Deformity"][l2]) if isinstance(pred["Type of Deformity"].get(l2), list) else set()
        
        # 计数正确的L3标签
        for l3 in gt_l3 & pred_l3:
            counter[l3] += 1
    
    for l2 in pred_l2:
        if l2 not in (gt_l2 & pred_l2):
            counter["extra_lable"][l2] += 1

    for l2 in gt_l2:
        if l2 not in (gt_l2 & pred_l2):
            counter["miss_lable"][l2] += 1

def calculate_recall(gt_counts, correct_counts):
    """
    计算每个标签的召回率
    
    参数:
        gt_counts: 真实标签计数
        correct_counts: 正确预测计数
    
    返回:
        recall_dict: 每个标签的召回率字典
    """
    recall_dict = {}
    
    for label in gt_counts:
        if gt_counts[label] > 0:
            recall = correct_counts[label] / gt_counts[label]
            recall_dict[label] = recall
        else:
            recall_dict[label] = None  # 标记为无效（没有此类标签的样本）
    
    return recall_dict

from mathruler.grader import extract_boxed_content
def process_jsonl_file(file_path):
    """
    处理jsonl文件并计算标签召回率
    
    参数:
        file_path: jsonl文件路径
    
    返回:
        recall_dict: 每个标签的召回率字典
    """
    gt_label_counts = copy.deepcopy(LABEL_COUNT_MAP0)
    correct_label_counts = copy.deepcopy(LABEL_COUNT_MAP0)
    correct_label_counts["extra_lable"] = copy.deepcopy(LABEL_COUNT_MAP0)
    correct_label_counts["miss_lable"]  = copy.deepcopy(LABEL_COUNT_MAP0)
    num_lines = 0
    legal_nums = 0
    NUMS_OUT_DATA=0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            num_lines+=1
            try:
                data = json.loads(line.strip())
                gt = data.get('response', {})
                pred = data.get('output', {})

                # 确保gt和pred是字典格式
                if isinstance(gt, str) and isinstance(pred, str):
                    try:
                        pred = pred[-1000:]
                        gt = json.loads(gt)
                        # print(pred)
                        if "boxed{" in pred:
                            predict_str = extract_boxed_content(pred)
                        elif "{" in pred and "}" in pred:
                            predict_str = pred.split('{', 1)[1]
                            predict_str = predict_str[::-1].split('}', 1)[1]
                            predict_str = "{" + predict_str[::-1] + "}"

                        predict_str = parse_vlm_output_by_keywords(predict_str)
                        pred = json.loads(predict_str)
                        legal_nums+=1
                    except:
                        # 如果解析失败，跳过此样本
                        continue
                
                # 统计标签
                count_labels(gt, gt_label_counts)
                count_correct_labels(gt, pred, correct_label_counts)
                
            except Exception as e:
                print(f"处理行时出错: {e}")
                continue
    # print("num_lines: " + str(num_lines))
    # 计算召回率
    recall_dict = calculate_recall(gt_label_counts, correct_label_counts)
    print("legal_nums: " + str(legal_nums))
    
    return recall_dict, gt_label_counts, correct_label_counts


def display_results(recall_dict, gt_counts, correct_counts):
    """
    格式化显示结果
    """
    print("标签召回率统计:")
    print("-" * 80)
    print(f"{'标签':<40} | {'正确数':<10} | {'总数':<10} | {'召回率':<10}")
    print("-" * 80)
    
    # 按标签类型分组显示
    categories = [
        ("基本标签", ["Whether Normal", "'Whether Normal': True", "Type of Deformity"]),
        ("L2标签", [k for k in recall_dict.keys() if k.startswith("L2:")]),
        ("L3标签", [k for k in recall_dict.keys() if k.startswith("L3:")])
    ]
    
    for category_name, labels in categories:
        print(f"\n{category_name}:")
        for label in labels:
            if recall_dict[label] is not None:
                label_str = label
                if label == "Whether Normal":
                    label_str = "Whether Normal (二分类正确率)" 
                recall = recall_dict[label] * 100
                print(f"{label_str:<40} | {correct_counts[label]:<10} | {gt_counts[label]:<10} | {recall:.2f}%")
            else:
                print(f"{label_str:<40} | {correct_counts[label]:<10} | {gt_counts[label]:<10} | N/A")

    print("")
    cnt_extra_all = 0
    cnt_gt_all = 0
    for category_name, labels in categories:
        if category_name == "L2标签":
            print("extra_count:")
            for label in labels:
                label_str = label
                rate = correct_counts["extra_lable"][label] / gt_counts[label] * 100
                aaa = correct_counts["extra_lable"][label]
                cnt_extra_all   += aaa
                cnt_gt_all      += gt_counts[label]
                print(f"{label_str:<40} | {aaa:<10} | {gt_counts[label]:<10} | {rate:.2f}%")
    label_str = "Total L2 extra label"
    rate = cnt_extra_all / cnt_gt_all * 100
    print(f"{label_str:<40} | {cnt_extra_all:<10} | {cnt_gt_all:<10} | {rate:.2f}%")
    print("")

    cnt_miss_all = 0
    for category_name, labels in categories:
        if category_name == "L2标签":
            print("miss_count:")
            for label in labels:
                label_str = label
                rate = correct_counts["miss_lable"][label] / gt_counts[label] * 100
                aaa = correct_counts["miss_lable"][label]
                cnt_miss_all += aaa
                print(f"{label_str:<40} | {aaa:<10} | {gt_counts[label]:<10} | {rate:.2f}%")
    label_str = "Total L2 miss label"
    rate = cnt_miss_all / cnt_gt_all * 100
    print(f"{label_str:<40} | {cnt_miss_all:<10} | {cnt_gt_all:<10} | {rate:.2f}%")

    del correct_counts["extra_lable"]
    del correct_counts["miss_lable"]
    
    # 计算总体召回率
    total_correct = sum(correct_counts.values())
    total_gt = sum(gt_counts.values())
    overall_recall = total_correct / total_gt if total_gt > 0 else 0
    
    print("\n总体统计:")
    print(f"总正确预测标签数: {total_correct}")
    print(f"总标签数: {total_gt}")
    print(f"总体召回率: {overall_recall:.2f}%")
    

def parse_vlm_output_by_keywords(vlm_output_str: str) -> str:
    """
    通过关键字提取的方式，解析VLM输出的可能不规范的字符串，
    并将其格式化为标准的JSON字符串。

    Args:
        vlm_output_str: VLM模型输出的原始字符串。

    Returns:
        一个可以被json.loads()解析的JSON格式字符串。
    """
    union_abnormal_labels = {
        "L2: Irrational Element Attributes": ["L3: Abnormal Material Texture", "L3: Abnormal Detail Drawing", "L3: Abnormal Element Proportion", "L3: Abnormal Color Combination"],
        "L2: Irrational Element Interaction": ["L3: Abnormal Light and Shadow Effect", "L3: Abnormal Element Overlap", "L3: Abnormal Spatial Position"],
        "L2: Abnormal Human Anatomy": ["L3: Limb Structure Deformity", "L3: Trunk Structure Deformity", "L3: Hand Structure Deformity", "L3: Foot Structure Deformity", "L3: Facial Structure Deformity", "L3: Abnormal Human Anatomy", "L3: Abnormal and Uncoordinated Posture"],
        "L2: Abnormal Animal Anatomy": ["L3: Abnormal Limb Structure", "L3: Abnormal Posture Presentation", "L3: Abnormal Head Structure"],
        "L2: Abnormal Object Morphology": True,
        "L2: Other Irrationalities": True,
    }

    # 1. 判断字符串中是否有'true'或'false' (不区分大小写)
    # 简单处理：如果包含 'true'，则认为是正常
    sp_str = vlm_output_str.lower().split(",")[0][:40]
    # print(sp_str, end=".   ###\n")
    if 'true' in sp_str:
        output_dict = {"Whether Normal": True}
        # 使用 json.dumps 确保输出是标准的JSON格式
        return json.dumps(output_dict)
    
    # For Qwen2.5-VL-7B, 
    # e.g. "\n\nTherefore, the image is reasonable and does not exhibit any deformities.\n\n\\boxed{answer}"
    if ("answer" in sp_str) and (len(sp_str) < 8):
        output_dict = {"Whether Normal": True}
        return json.dumps(output_dict)

    # 非法回答，例如： answer
    if 'false' not in sp_str:
        return vlm_output_str

    # 如果没有 'true'，则认为是异常
    output_dict = {
        "Whether Normal": False,
        "Type of Deformity": {}  # 使用 "Type of Deformity" 以匹配您的统计函数
    }

    # 2. & 3. 逐个检查L2/L3标签是否存在，并按层级组织
    for l2_label, l3_options in union_abnormal_labels.items():
        # 检查L2标签是否存在于输出字符串中
        if l2_label in vlm_output_str:
            # 情况一：L2标签没有具体的L3子类别（值为True）
            if l3_options is True:
                output_dict["Type of Deformity"][l2_label] = True
            
            # 情况二：L2标签包含一个L3子类别列表
            elif isinstance(l3_options, list):
                found_l3_labels = []
                # 遍历所有可能的L3标签
                for l3_label in l3_options:
                    if l3_label in vlm_output_str:
                        found_l3_labels.append(l3_label)
                
                # 如果找到了至少一个对应的L3标签，才将这个L2类别加入字典
                if found_l3_labels:
                    output_dict["Type of Deformity"][l2_label] = found_l3_labels

    # 将最终构建的字典转换为标准的JSON字符串
    return json.dumps(output_dict, ensure_ascii=False)



