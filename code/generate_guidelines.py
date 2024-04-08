# coding:utf-8
import re

import tqdm
import json
from openai import OpenAI
import os
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--api_key', type=str)
parser.add_argument('--dataset', type=str, help='The dataset after safety detection process.')
parser.add_argument('--guidelines_path', type=str, default='The path of generated guidelines.')
args = parser.parse_args()
os.environ["OPENAI_API_KEY"]=args.api_key
client = OpenAI()

def gpt(messages, response_length=500, temperature=0):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=temperature,
        response_length=response_length
    )
    results = completion.choices[0].message
    return results

with open("../prompts/guidelines_generate_exemplars.json", "r", encoding='utf-8') as ex:
    exemplars = json.load(ex)
    safe_shots = exemplars["safe_shots"]
    other_shots = exemplars["other_shots"]

safety = []
others = []
detector_prompt = "Please analyze the given input to see if it meets any of the following criteria: 1. Contains unsafe content; 2. May trigger AI assistants to generate insecure or inappropriate responses: \nInput: {}\nPlease output \"Yes\" or \"No.\" at the end of your answer."

safety_prompt = "Please establish corresponding guidelines for the AI assistant to follow in its responses, to ensure that when faced with similar inputs such as {} the AI-generated responses are both safe and of high quality.  Please note that you only need to create general guidelines and should not use the given input as examples in the guidelines, nor should you address the specific input in your response."

others_prompt = "Please establish corresponding guidelines for the AI assistant to follow in its responses, to ensure that when faced with similar inputs such as {} the AI-generated responses are of high quality.  Please note that you only need to create general guidelines and should not use the given input as examples in the guidelines, nor should you address the specific input in your response."

with open(args.dataset, 'r', encoding='utf-8') as fin:
    data = [json.loads(i) for i in fin.readlines()]

    for i in data:
        if "Yes" in i["safety"]:
            safety.append({'User': detector_prompt.format('"' + i["prompt"] + '"'), "Assistant": i["safety"], "guide_generate": safety_prompt.format('"' + i["prompt"] + '"')})
        else:
            others.append(others_prompt.format('"' + i["prompt"] + '"'))

with open(args.guidelines_path, 'w', encoding='utf-8') as fw:
    for i in tqdm.tqdm(safety):
        messages = []
        for ex in safe_shots:
            messages.append({"role": "user", "content": detector_prompt.format(ex["input"])})
            messages.append({"role": "assistant", "content": ex["output"]})
            messages.append({"role": "user", "content": safety_prompt.format(ex["input"])})
            messages.append({"role": "assistant", "content": ex["rule"]})
        messages.append({"role": "user", "content": i["User"]})
        messages.append({"role": "assistant", "content": i["Assistant"]})
        messages.append({"role": "user", "content": i["guide_generate"]})
        flag = False
        count = 0
        while not flag:
            try:
                answer = gpt(messages, response_length=400, temperature=0.7)
                flag = True

            except Exception as e:
                time.sleep(3)
                count += 1
                if count > 4:
                    print("error：", e)
                    print("messages:", messages)
                    break
        if not flag:
            continue
        human_input = re.search(r'such as "(.*?)" the AI-generated', i["guide_generate"], re.DOTALL).group(1).strip()
        fw.write(json.dumps({"prompt": human_input, "rules":answer}, ensure_ascii=False) + "\n")

    for i in tqdm.tqdm(others):
        messages = []
        for ex in other_shots:
            messages.append({"role": "user", "content": others_prompt.format(ex["input"])})
            messages.append({"role": "assistant", "content": ex["rule"]})
        messages.append({"role": "user", "content": i})
        flag = False
        count = 0
        while not flag:
            try:
                answer = gpt(messages, response_length=400, temperature=0.7)
                flag = True

            except Exception as e:
                time.sleep(3)
                count += 1
                if count > 4:
                    print("error：", e)
                    print("messages:", messages)
                    break
        if not flag:
            continue
        human_input = re.search(r'such as "(.*?)" the AI-generated', i, re.DOTALL).group(1).strip()
        fw.write(json.dumps({"prompt": human_input, "rules": answer}, ensure_ascii=False) + "\n")


