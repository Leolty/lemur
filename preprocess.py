import json
import fire
import os
from datasets import load_dataset
from variables import SYSTEM, HUMAN, AI, NAME, ORGANIZATION
from tqdm import tqdm

def main(dataset: str):
    # dataset can only be alpaca, quora, stackoverflow, sharegpt or identity
    # ignore cases
    assert dataset.lower() in ['alpaca', 'quora', 'stackoverflow', 'sharegpt', 'identity'] \
        , "dataset can only be alpaca, quora, stackoverflow, sharegpt or identity"

    os.makedirs(f'./data', exist_ok=True)
    
    # load data
    if dataset.lower() == 'alpaca':
        data = load_dataset('vicgalle/alpaca-gpt4')

        for i, d in tqdm(enumerate(data['train']), total=len(data['train'])):
            if d['input']:
                input = f"{SYSTEM}\n{HUMAN} {d['instruction']}\n{d['input']}\n{AI} {d['output']}"
            else:
                input = f"{SYSTEM}\n{HUMAN} {d['instruction']}\n{AI} {d['output']}"

            save_dic = {
                'idx': i,
                'input': input
            }
            with open(f'./data/{dataset.lower()}.jsonl', 'a') as f:
                f.write(json.dumps(save_dic) + '\n')
        
    
    elif dataset.lower() in ['quora', 'stackoverflow']:

        with open(f'./baize-chatbot/data/{dataset.lower()}_chat_data.json', 'r') as f:
            data = json.load(f)
        
        for i, d in tqdm(enumerate(data), total=len(data)):
            input = d['input']

            # change it to our format
            input = input.replace('The conversation between human and AI assistant.', SYSTEM)
            input = input.replace('[|Human|]', HUMAN)
            input = input.replace('[|AI|]', AI)
    
            save_dic = {
                'idx': i,
                'input': input
            }

            with open(f'./data/{dataset.lower()}.jsonl', 'a') as f:
                f.write(json.dumps(save_dic) + '\n')
    
    elif dataset.lower() == 'sharegpt':
        #  download the data online to ./temp
        # link here: https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/blob/main/ShareGPT_V3_unfiltered_cleaned_split.json

        with open(f'./temp/ShareGPT_V3_unfiltered_cleaned_split.json', 'r') as f:
            data = json.load(f)

        for i, d in tqdm(enumerate(data), total=len(data)):
            input = f"{SYSTEM}\n"
            for conv in d['conversations']:
                if conv['from'] == 'human':
                    input += f"{HUMAN} {conv['value']}\n"
                elif conv['from'] == 'gpt':
                    input += f"{AI} {conv['value']}\n"
            
            save_dic = {
                'idx': i,
                'input': input
            }
        
            with open(f'./data/{dataset.lower()}.jsonl', 'a') as f:
                f.write(json.dumps(save_dic) + '\n')
    
    elif dataset.lower() == 'identity':
        # this is for AI self-idnetity
        with open(f'./FastChat/data/dummy_conversation.json', 'r') as f:
            data = json.load(f)
        
        for i, d in tqdm(enumerate(data), total=len(data)):
            input = f"{SYSTEM}\n"
            for conv in d['conversations']:
                value = conv['value']

                # submit Vicuna to Lemur
                value = value.replace('Vicuna', NAME)

                # change the organization to UC San Diego
                value = value.replace('Large Model Systems Organization (LMSYS)', ORGANIZATION)

                if conv['from'] == 'human':
                    input += f"{HUMAN} {value}\n"
                elif conv['from'] == 'gpt':
                    input += f"{AI} {value}\n"
            
            save_dic = {
                'idx': i,
                'input': input
            }

            with open(f'./data/{dataset.lower()}.jsonl', 'a') as f:
                f.write(json.dumps(save_dic) + '\n')
if __name__ == '__main__':
    fire.Fire(main)

        







        




