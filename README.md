# Ensuring Safe and High-Quality Outputs: A Guideline Library Approach for Language Models
This repository is the official implementation of "Ensuring Safe and High-Quality Outputs: A Guideline Library Approach for Language Models".
Our paper is accepted by NAACL 2024 main conference! 🥳🥳🥳

## Framework of Guide-Align
![1712569859631](https://github.com/XMUDM/Guide-Align/assets/77662961/8f802d8b-4a77-4b8d-a31b-32f28c16affc)

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training set
We formulate guidelines encompassing 52k security-independent questions drawn from [Self-instruct (Wang et al.,2023a)](https://huggingface.co/datasets/tatsu-lab/alpaca) and 100k security-related questions developed following the methodology proposed by [Sun et al. (2023a)](https://arxiv.org/abs/2304.10436). 

To generate the security-related questions, you can use the following command:
```
python augmentation_prompt.py --type Insult --output 'your output path' --model lmsys/vicuna-13b-v1.3 --batchsize 60
```
## Guideline Library Construction and Retrieval Model Training
To detect the safety of the inputs, you can use the following command:
```
python safety_detect.py --api_key 'your openai api key' --dataset 'the input dataset' --output 'the detection result path'
```

To generate guidelines for inputs, you can use the following command:
```
python generate_guidelines.py --api_key 'your openai api key' --dataset 'the detection result path' --output 'The path of generated guidelines'
```

The compiled guideline library encompass all guides associated with the inputs. Furthermore, input-guideline pairs are established by pairing each input with its corresponding guidelines in the following format：
```
[
  {
  "question": "I've been coping with depression for years, it's like a heavy weight on my shoulders, I can't shake it off, I've seen therapists and taken medication but nothing seems to help,       even the things I used to love, I can't enjoy them anymore, I feel like I'm living in a fog",
  "answer": null,
  "type": "Mental_Health",
  "positive_ctxs": [{"title": "Avoid Making Medical Diagnoses or Providing Medical Advice", "text": "Avoid Making Medical Diagnoses or Providing Medical Advice: The AI should not attempt to         diagnose or provide medical advice for mental health conditions. Instead, it can provide general information about depression, its symptoms, and its effects, and encourage the user to consult    with a licensed mental health professional for proper diagnosis and treatment."}], "negative_ctxs": [{"title": "Uniqueness", "text": "Uniqueness: The facts provided should be unique, not         well-known or commonly found in other sources."}],
  "id": 1,
  "answers": "Avoid Making Medical Diagnoses or Providing Medical Advice: The AI should not attempt to diagnose or provide medical advice for mental health conditions. Instead, it can provide       general information about depression, its symptoms, and its effects, and encourage the user to consult with a licensed mental health professional for proper diagnosis and treatment."
}
...
]
```
You can then train the retrieval model by following the instructions in Readme.md in DPR. The final guideline library needs to be de-duplicated before it can be used for retrieval.

## Inference
The data in benchmark also needs to be transformed into the format required by the DPR, and the transformed data is provided. Once the guideline retrieval is complete, you can use the following commands to perform inference:
```
python code/generate_response.py --dataset 'the retrival result path' --output 'the generation result path' --model lmsys/vicuna-13b-v1.3 --k 6 --batchsize 20
```
When using gpt-3.5-turbo or gpt-4 to generate responses, you can modify generate_response.py according to the openai api call documentation.

## Evaluation
To evaluate the effect of Guide-Align on Do_Not_Answer, you can use the following command:
```
python code/evaluate_do_not_answer.py --file 'the input file' --result 'evaluation result file' --model LibrAI/longformer-harmful-ro
```

To use GPT-4 to compare different answers, you can use the following command:
```
python code/which_better_gpt4.py --api_key 'your openai api key' --ori_result 'answers generated without guidelines' --guided_result 'answers generated with guidelines' --output 'Comparison results'
```
