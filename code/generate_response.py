from vllm import LLM, SamplingParams
import json
import tqdm
import argparse
import re
from torch.utils.data import Dataset, DataLoader
from fuzzywuzzy import fuzz

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--model', type=str, default='', help='the model file path')
parser.add_argument('--k', type=int, default=6, help='use top k rules')
parser.add_argument('--batchsize', type=int)
parser.add_argument('--tensor_parallel_size', type=int, default=1)
parser.add_argument('--max_seq_len', type=int, default=512, required=False)
parser.add_argument('--max_num_batched_tokens', type=int, default=4096, required=False)

args = parser.parse_args()


if "vicuna" in args.model:
    seps = [" ","</s>"]
    roles = ["USER", "ASSISTANT"]
    system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful and detailed answers to the user's questions." + seps[0]

def generate_prompt(messages):
    ret = system_prompt
    for i, (role, message) in enumerate(messages):
        if message:
            if role == roles[0]:
                rules = re.search(r"####.*?####", message, re.DOTALL)
                if rules:
                    rules = rules.group()
                    message = message.replace(rules, '')
                    ret += rules + " " + role + ": " + message + seps[i % 2]
                else:
                    ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ": " + message
        else:
            ret += role + ":"
    return ret

sampling_params = SamplingParams(temperature=0, max_tokens=args.max_seq_len)
# Create an LLM.
llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size, max_num_batched_tokens=args.max_num_batched_tokens)

def create_input(inp, rules):
    input_story = """####{}####{}
    """
    output_prefix = None
    input_story_1 = input_story.format(rules, inp)
    message1 = [(roles[0], input_story_1), (roles[1], output_prefix)]
    input1 = generate_prompt(message1)
    return input1

class DataPrecessForSentence(Dataset):
    """
    对文本进行处理
    """

    def __init__(self, args):
        self.data = self.get_input(args)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = {}
        data["input"] = self.data["input"][idx]
        data["rules"] = self.data["rules"][idx]
        replacement_text = "Here are some guidelines for the AI assistant to follow:\n"
        data["rules"] = replacement_text + "\n".join(data["rules"])
        return data

    def get_input(self, args):
        with open(args.dataset, "r", encoding="utf-8") as fr:
            data = json.load(fr)
        examples = {"input": [], "rules": []}
        for i in data:
            examples["input"].append(i["question"])

            #De-duplication
            chosen_rules = [j["text"] for j in i["ctxs"]]
            final_rules = [chosen_rules[0]]
            for item in chosen_rules[1:]:
                max_similarity = max([fuzz.ratio(item, result_item) / 100 for result_item in
                                  final_rules]) < 0.5
                #print("max_similarity: ", max_similarity)
                if max_similarity:
                    final_rules.append(item)
            final_rules = final_rules[:args.k] if len(final_rules) > args.k else final_rules
            #print("final_rules: ", final_rules)
            examples["rules"].append(final_rules)
        self.length = len(examples["input"])
        return examples

all_predictions = []
dataset = DataPrecessForSentence(args)
d = DataLoader(dataset, batch_size=args.batchsize)
count = 0
acc = 0
for i in tqdm.tqdm(d, total=len(d)):
    input = [create_input(inp, rules) for inp, rules in zip(i["input"], i["rules"])]
    #print(input)
    #input = [j for i in input for j in i]
    outputs = llm.generate(input, sampling_params)
    for id, out in enumerate(outputs):
        count += 1
        generated_text = out.outputs[0].text
        #print("prompt:###{}###".format(out.prompt))
        #print("output:###{}###".format(generated_text))
        all_predictions.append({"answer": generated_text, "prompt": out.prompt})

with open(args.output, 'w', encoding='utf8') as fw:
    json.dump(all_predictions, fw, ensure_ascii=False, indent=4)




