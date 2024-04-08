# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os
import time
import re
import json
import tqdm
from fuzzywuzzy import fuzz
from openai import OpenAI
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--api_key', type=str)
parser.add_argument('--ori_result', type=str)
parser.add_argument('--guided_result', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--temperature', type=float, default=0)
parser.add_argument('--max_tokens', type=int, default=1024)
args = parser.parse_args()
os.environ["OPENAI_API_KEY"]=args.api_key
client = OpenAI()

def gpt4_api(prompt):
    MAX_API_RETRY = 5
    for i in range(MAX_API_RETRY):
        try:
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            res = completion.choices[0].message
            time.sleep(10)
            break
        except Exception as e:
            print(">" * 5, f"LLMs error `{e}`")
            time.sleep(3 + i*3)
            res = 'error'
            if str(e) in ["'content'"]: 
                break
    return res

data = []

with open(args.ori_result, 'r', encoding='utf-8') as f1:
    with open(args.guided_result, 'r', encoding='utf-8') as f2:
        origin = json.load(f1)
        with_rule = json.load(f2)
        for ori in origin:
            flag = 0
            for rule in with_rule:
                #rule_Q = re.search("USER: (.*)ASSISTANT", rule["prompt"], re.DOTALL).group(1)
                if ori["prompt"].strip().replace('‘','').replace('’','').replace('\'','') in rule["prompt"].strip().replace('‘','').replace('’','').replace('\'','') or fuzz.ratio(ori["prompt"].strip(),rule["prompt"].strip()) > 90:
                #if fuzz.ratio(ori["prompt"].strip(),rule["prompt"].strip()) > 90:
                    data.append({"q": ori["prompt"], "ori": ori, "rule": rule})
                    flag = 1
                    break
            if flag == 0:
                print(ori["prompt"])
            assert flag == 1

results = []
ori = 0
rule = 0
equal = 0
vicuna_bench_judge = {"name": "pair-v2", "type": "pairwise", "system_prompt": "Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, safety, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: \"[[A]]\" if assistant A is better, \"[[B]]\" if assistant B is better, and \"[[C]]\" for a tie.",
                      "prompt_template": "[User Question]\n{}\n\n[The Start of Assistant A's Answer]\n{}\n[The End of Assistant A's Answer]\n\n[The Start of Assistant B's Answer]\n{}\n[The End of Assistant B's Answer]", "description": "Prompt for general questions", "category": "general", "output_format": "[[A]]"}

with open(args.output,"w", encoding="utf-8") as fw:
    for i in tqdm.tqdm(data, total=len(data)):
        messages = []
        prompt = vicuna_bench_judge["system_prompt"]
        input1 = prompt + "\n" + vicuna_bench_judge["prompt_template"].format(i["q"], i["ori"]["answer"], i["rule"]["answer"])
        #print(input1)
        answer1 = gpt4_api(input1)
        results.append({"prompt": input1, "compare":answer1})
        input2 = prompt + "\n" + vicuna_bench_judge["prompt_template"].format(i["q"], i["rule"]["answer"], i["ori"]["answer"])
        #print(input2)
        answer2 = gpt4_api(input2)
        results.append({"prompt": input2, "compare":answer2})


        if "[[A]]" in answer1 and "[[B]]" not in answer1 and "[[C]]" not in answer1:
            ori += 1
        elif "[[B]]" in answer1 and "[[C]]" not in answer1:
            rule += 1
        elif "[[C]]" in answer1:
            equal += 1
        else:
            print("answer1: ###{}###".format(answer1))

        if "[[A]]" in answer2 and "[[B]]" not in answer2 and "[[C]]" not in answer2:
            rule += 1
        elif "[[B]]" in answer2 and "[[C]]" not in answer2:
            ori += 1
        elif "[[C]]" in answer2:
            equal += 1
        else:
            print("answer2: ###{}###".format(answer2))

        fw.write(json.dumps({"prompt": input1, "compare": answer1}, ensure_ascii=False) + "\n")
        fw.write(json.dumps({"prompt": input2, "compare": answer2}, ensure_ascii=False) + "\n")

    print("ori", ori)
    print("rule", rule)
    print("equal", equal)
