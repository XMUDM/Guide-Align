from vllm import LLM, SamplingParams
import json
import tqdm
import random
import argparse
from rouge import Rouge
import re

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str)
parser.add_argument('--output', type=str, help='the output file path')
parser.add_argument('--model', type=str, default='', help='the model file path')
parser.add_argument('--batchsize', type=int)
parser.add_argument('--tensor_parallel_size', type=int, default=1)
parser.add_argument('--max_seq_len', type=int, default=512, required=False)
parser.add_argument('--max_num_batched_tokens', type=int, default=4096, required=False)
parser.add_argument('--instructions_num', type=int, default=10000, required=True, help='The number of instructions needs to be generated.')

args = parser.parse_args()

seps = [" ", "</s>"]
def generate_prompt(messages):
    ret = "A chat between a curious user and an artificial intelligence assistant.  The assistant gives helpful and detailed answers to the user's questions." + seps[0]
    for i, (role, message) in enumerate(messages):
        if message:
            if role == "USER":
                ret += role + ": " + message + seps[i % 2]
            else:
                ret += role + ": " + message
        else:
            ret += role + ":"
    return ret

sampling_params = SamplingParams(temperature=1, max_tokens=args.max_seq_len)
# Create an LLM.
llm = LLM(model=args.model, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size, max_num_batched_tokens=args.max_num_batched_tokens)

rouge = Rouge()
all_instructions = []
new_instructions = []
#load seed instructions
with open("../prompts/seed_prompt.json", "r", encoding="utf-8") as f:
    data = json.load(f)
seed_instructions = data[args.type]["example"]
all_instructions.extend(seed_instructions)

def return_random_prompt():
    user_prompt = data[args.type]["prompt"]
    if len(new_instructions) < 2:
        examples = random.sample(seed_instructions, 4)
    else:
        examples = random.sample(seed_instructions, 2)
        examples.extend(random.sample(new_instructions, 2))
  # generate random topics
    ai_answer = "1.{}\n2.{}\n3.{}\n4.{}\n5.".format(examples[0], examples[1], examples[2], examples[3])
    message = [("USER", user_prompt), ("ASSISTANT", ai_answer)]
    return generate_prompt(message)


with open(args.type+"_instructions.txt", "w", encoding="utf-8") as output_file:
    pattern = r'\d+\.(.*?)'
    progress_bar = tqdm.tqdm(total=args.instructions_num)
    while len(new_instructions) < args.instructions_num:
        input = [return_random_prompt()]
        #print("input",input)
        output = llm.generate(input, sampling_params)
        answer = output[0].outputs[0].text
        answers = answer.split("\n")
        answers = [i for i in answers if re.match(r'^\d+\.', i) is not None]
        answers = [re.sub(r'^\d+\.', '', i) for i in answers]
        answers = [i for i in answers if i != '']
        answers = answers[:6]
        scores = [[rouge.get_scores(i, ai)[0]['rouge-l']['f'] for ai in all_instructions] for i in answers]

        keep_list = [max(i) <= 0.7 for i in scores]
        result = [ans for ans, keep in zip(answers, keep_list) if keep]
        new_instructions.extend(result)
        all_instructions.extend(result)
        res = "\n".join(result)
        output_file.write(res + '\n')
        progress_bar.update(len(result))