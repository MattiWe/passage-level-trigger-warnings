import logging
from functools import partial
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from typing import Tuple
import json
import torch
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForCausalLM
)
from sequence_level_trigger_warning_assignment.dataset.prompt import Prompt
from sequence_level_trigger_warning_assignment.evaluation.evaluation import _score

logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.ERROR)
  

def _tokenize_function(example, tokenizer):
    return tokenizer(
        example["text"],
        padding=False,
        truncation=True,
    )
        

def _load_dataset(
    checkpoint: str, test_path: str, cache_dir: str
) -> Tuple:
    """load and tokenize the dataset. 

    :param checkpoint: _description_
    :param train_path: _description_
    :param validation_path: _description_
    :param test_path: _description_
    :param mode: _description_
    :return: _description_
    """
    # "mistralai/Mistral-7B-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    tokenize_function = partial(_tokenize_function, tokenizer=tokenizer)

    data_files = {"test": test_path}
    dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
    dataset.set_format("torch")

    logger.info("tokenizing datasets")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets_trainable = tokenized_datasets.remove_columns(["id", "text", "warning"])
    return tokenized_datasets_trainable, tokenized_datasets["test"], tokenizer


def _get_model(checkpoint, cache_dir):
    if checkpoint == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        return AutoModelForCausalLM.from_pretrained(
            checkpoint, device_map='auto', load_in_8bit=True, bnb_4bit_compute_dtype=torch.float16,
            cache_dir=cache_dir
        )
    elif checkpoint == "meta-llama/Llama-2-13b-chat-hf":
        return AutoModelForCausalLM.from_pretrained(
            checkpoint, device_map='auto', 
            # load_in_8bit=True, bnb_4bit_compute_dtype=torch.float16,
            cache_dir=cache_dir
        )
    else:
        model =  AutoModelForCausalLM.from_pretrained(
            checkpoint, device_map='auto', # load_in_4bits=True
            cache_dir=cache_dir
        )
        # model.to('cuda')
        return model

def run_generative_experiment(
    checkpoint, test, savepoint, cache_dir: str, batches, name, max_new_tokens=50
):
    logger.info("load dataset and tokenizer")
    tokenized_datasets, test_dataset_with_metadata, tokenizer = \
        _load_dataset(checkpoint, test, cache_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    logger.info("define models and parameters")
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=batches, collate_fn=data_collator
    )

    model = _get_model(checkpoint, cache_dir)
    
    labels = []
    responses = []
    predictions = []

    for batch in test_dataloader:
        batch_example_length = len(batch["input_ids"][0])
        batch.to(model.device)
        generated_ids = model.generate(**batch, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        results = tokenizer.batch_decode(generated_ids[:,batch_example_length:], skip_special_tokens=True)
        responses.extend(results)
        predictions.extend([Prompt.parse(r) for r in results])
        labels.extend(batch["labels"].cpu().numpy())   

    metrics = _score(labels, predictions)
    
    open(f"{savepoint}/scores-{name}.json", 'w').write(json.dumps(metrics))

    test_dataset_with_metadata = test_dataset_with_metadata.add_column("predictions", predictions)   
    test_dataset_with_metadata = test_dataset_with_metadata.add_column("responses", responses)   
    test_dataset_with_metadata = test_dataset_with_metadata.remove_columns(["input_ids", "attention_mask"])
    test_dataset_with_metadata.to_json(f"{savepoint}/predictions-{name}.jsonl")
    

def run_chat_experiment(
    checkpoint, test, savepoint, cache_dir: str, batches, name, max_new_tokens=50
):
    logger.info("load dataset and tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token

    data_files = {"test": test}
    dataset = load_dataset("json", data_files=data_files, cache_dir=cache_dir)
    dataset.set_format("torch")

    logger.info("define models and parameters")

    model = _get_model(checkpoint, cache_dir)
    
    labels = []
    responses = []
    predictions = []

    for example in dataset["test"]:
        if checkpoint == "meta-llama/Llama-2-13b-chat-hf" or checkpoint == "meta-llama/Llama-2-7b-chat-hf":
            messages = [
                    {"role": "system", "content": "You are a classification model that only answers with 'yes' or 'no'."},
                    {"role": "user", "content": example["text"]}
                    ]
        else:
            messages = [
                    {"role": "user", "content": "You are a classification model that only answers with 'yes' or 'no'.\n" + example["text"]}
                    ]
        inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")
        example_length = len(inputs[0])
        generated_ids = model.generate(inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)
        result = tokenizer.batch_decode(generated_ids[:,example_length:], skip_special_tokens=True)[0]
        responses.append(result)
        predictions.append(Prompt.parse(result))
        labels.append(example["labels"].cpu().numpy())   

    metrics = _score(labels, predictions)
    
    open(f"{savepoint}/scores-{name}.json", 'w').write(json.dumps(metrics))

    test_dataset_with_metadata = dataset["test"]
    test_dataset_with_metadata = test_dataset_with_metadata.add_column("predictions", predictions)   
    test_dataset_with_metadata = test_dataset_with_metadata.add_column("responses", responses)   
    test_dataset_with_metadata.to_json(f"{savepoint}/predictions-{name}.jsonl")
    

if __name__ == "__main__":
    exit("Call this via main.py")
