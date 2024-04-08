from transformers import pipeline
import tqdm
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str)
parser.add_argument('--result', type=str)
parser.add_argument('--batchsize', default=20, type=int)
parser.add_argument('--model', type=str)
args = parser.parse_args()

classifier = pipeline(model=args.model) # ro means response-only model

def chunk_list(input_list, chunk_size):
    """将列表分割为多个子列表"""
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


with open(args.file, 'r', encoding='utf-8') as f:
    with open(args.result, 'w', encoding='utf-8') as out:
        data = json.load(f)
        data = [i['answer'] for i in data]
        data = chunk_list(data, args.batchsize)
        results = []
        for i in tqdm.tqdm(data):
            label = classifier(i)
            results.extend(label)
            print(label)
        json.dump(results, out)
    