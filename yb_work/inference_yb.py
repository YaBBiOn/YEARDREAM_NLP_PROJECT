import os
from tqdm import tqdm

import pandas as pd 
import torch
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

import argparse 
from transformers import set_seed
from datasets import Dataset, DatasetDict


# Define your evaluation function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(logits, dim=-1)
    return {
        'accuracy': (predictions == labels).float().mean().item()
    }

def generate_dict(df, cluster_prompts): # 20240825 수정

    # Cluster별로 다른 프롬프트 설정 # 20240825 수정
    cluster_prompts = {
        0: ("Cluster 0 : Please carefully read the following question. Based on your understanding, knowledge, and reasoning, choose the most appropriate answer or provide the best possible explanation.\n"
        "Cluster 0 : 다음 질문을 신중하게 읽고, 자신의 이해, 지식, 그리고 논리를 바탕으로 가장 적절한 답을 선택하거나 최선의 설명을 제시하세요.\n"),
        1: ("Cluster 1 : Please read the following question carefully and choose the most accurate answer based on your understanding and knowledge.\n"
        "Cluster 1 : 다음 질문을 주의 깊게 읽고, 자신의 이해와 지식을 바탕으로 가장 정확한 답을 선택하세요.\n"),
        2: ("Cluster 2 : Consider the moral implications of the following scenarios and choose the one where the action is most clearly wrong according to common moral standards.\n"
        "Cluster 2 : 다음 시나리오들의 도덕적 함의를 고려하고, 일반적인 도덕 기준에 따라 가장 명백히 잘못된 행동이 있는 시나리오를 선택하세요.\n"),
        3: ("Cluster 3 : Complete the following statement by choosing the most appropriate term or concept from the options provided.\n"
        "Cluster 3 : 다음 문장을 완성하기 위해 주어진 선택지 중 가장 적절한 용어 또는 개념을 선택하세요.\n"),
        4: ("Cluster 4 : Evaluate the truth of the following propositions and determine which statements are true and which are false.\n"
        "Cluster 4 :다음 명제들의 진위를 평가하고, 참과 거짓을 판단하세요.\n"),
        5: ("Cluster 5 : Based on the information provided in the following sources, select the most accurate answer to the question posed.\n"
        "Cluster 5 :  제공된 자료의 정보를 바탕으로, 제시된 질문에 가장 정확한 답을 선택하세요.\n"),
        }


    prompt_ca = "You are given the following multiple choices and supposed to output the index of the correct answer."
    prompt_ex = "For instance, if the choices are ['a', 'b', 'c', 'd'] and the answer is 'b' the correct output should be 1."
    
    instruction_list = [
        [
            f"{cluster_prompts.get(row['cluster'], 'Provide me an answer to the following question.')} {prompt_ca} {prompt_ex}"
        ] 
        for _, row in df.iterrows()
    ]
    # instruction_list = [[prompt_q + ' ' + prompt_ca + ' ' + prompt_ex] for _ in range(len(df))]와 대응 # 20240825 수정
    
    question_list = df['문제'].tolist() 
    choices_list = df['__선택지'].tolist()  # 20240825 수정
    dataset_dict = {'instruction': instruction_list, 'question': question_list, 'choices': choices_list}
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset


def create_datasets(df, tokenizer, apply_chat_template=False):
    def preprocess(samples):
        batch = []
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context.\n"
                "아래는 작업을 설명하는 명령어와 추가적 맥락을 제공하는 입력이 짝을 이루는 예제입니다.\n\n"
                "Write a response that appropriately completes the request.\n요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{instruction}\n\n### Input(입력):\n{input}\n\n### Response(응답):"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task.\n"
                "아래는 작업을 설명하는 명령어입니다.\n\n"
                "Write a response that appropriately completes the request.\n명령어에 따른 요청을 적절히 완료하는 응답을 작성하세요.\n\n"
                "### Instruction(명령어):\n{instruction}\n\n### Response(응답):"
            ),
        }

        for instruction, question, choices in zip(samples["instruction"], samples["question"], samples["choices"]):
            user_input = question + '<|sep|>' + choices  
            conversation = PROMPT_DICT['prompt_input'].replace('{instruction}', instruction[0]).replace('{input}', user_input)
            batch.append(conversation)

        return {"content": batch}

    dataset = generate_dict(df, cluster_prompts)
    
    raw_datasets = DatasetDict()
    raw_datasets["test"] = dataset

    raw_datasets = raw_datasets.map(
        preprocess,
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
    )

    test_data = raw_datasets["test"]
    print(
        f"Size of the test set: {len(test_data)}"
    )
    print(f"A sample of test dataset: {test_data[1]}")

    return test_data


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(__file__)
    print(f"Available GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', required=False, default=42, help='add seed number')
    parser.add_argument('--response_split', required=False, default='\nThe answer is', help='add response splitter')
    parser.add_argument('--model_path', required=False, default='', help='add pretrained model path')

    args = parser.parse_args()
    set_seed(args.seed)
    
    model_path = os.path.join(BASE_DIR, args.model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    df = pd.read_csv(os.path.join(BASE_DIR, 'data/test_data.csv'), encoding='utf-8')
    
    test_dataset = create_datasets(
        df,
        tokenizer,
        apply_chat_template=False
    )
        
    device = "cuda" if torch.cuda.is_available else "cpu"
    model = model.to(device)
    
    df_submission = pd.DataFrame()
    id_list, answer_list = list(), list()
    
    for i, test_data in enumerate(tqdm(test_dataset)): 
        text = test_data['content']
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        model_inputs.pop('token_type_ids', None)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=9,
                eos_token_id=tokenizer.eos_token_id, 
                pad_token_id=tokenizer.pad_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        
        answer = response.split(args.response_split)[1].strip()
        id_list.append(i)
        answer_list.append(answer)
        
    df_submission['id'] = id_list
    df_submission['answer'] = answer_list
    df_submission.to_csv(os.path.join(BASE_DIR, 'submission.csv'), index=False)
