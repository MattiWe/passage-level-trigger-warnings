import logging
from functools import partial
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from typing import Tuple, Dict, List
import numpy as np
import json
import wandb
from dataclasses import dataclass
from accelerate import Accelerator
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    DataCollatorWithPadding,
    get_scheduler
)
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from gc import collect

logger = logging.getLogger("evaluation_logger")
logger.setLevel(logging.INFO)


def _tokenize_function(example, tokenizer):
        return tokenizer(
            example["text"],
            padding=False,
            truncation=True,
        )
        

def _load_dataset(
    checkpoint: str, train_path: str, validation_path: str, test_path: str, mode: str, cache_dir: str
) -> Tuple:
    """load and tokenize the dataset. 

    :param checkpoint: _description_
    :param train_path: _description_
    :param validation_path: _description_
    :param test_path: _description_
    :param mode: _description_
    :return: _description_
    """
    if cache_dir:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    tokenize_function = partial(_tokenize_function, tokenizer=tokenizer)

    data_files = {"train": train_path, "test": test_path, "validation": validation_path}
    dataset = load_dataset("json", data_files=data_files)
    dataset.set_format("torch")

    logger.info("tokenizing datasets")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    if mode == "multilabel":
        tokenized_datasets = tokenized_datasets.map(
            lambda x: {"float_labels": x["labels"].to(torch.float)}, remove_columns=["labels"]
        ).rename_column("float_labels", "labels")

    tokenized_datasets_trainable = tokenized_datasets.remove_columns(["id", "text"])
    return tokenized_datasets_trainable, tokenized_datasets["test"], tokenizer


def _score(logits, mode, multilabel_threshold=0.5):
    if mode == "multilabel":
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(logits))
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= multilabel_threshold)] = 1
    elif mode == 'multiclass' or mode == 'binary':
        y_pred = torch.argmax(logits, dim=-1).numpy()
    return y_pred


def _compute_metric(y_true, y_pred, mode) -> Dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "f1": f1_score(y_true=y_true, y_pred=y_pred, average="micro", zero_division=0),
        "f1_macro": f1_score(y_true=y_true, y_pred=y_pred, average="macro", zero_division=0),
        "f1_cls1": f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1) if mode=="binary" else 0,
        "f1_cls0": f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=0) if mode=="binary" else 0,
    }


def _eval(model, dataloader, epoch=0, wandb_prefix="eval", mode="binary", no_wandb=False) -> Tuple[Dict, List]:
    """Run the evaluation, push scores to wandb, and return the predictions
    of the elements in the dataloader

    :param model: _description_
    :param dataloader: _description_
    :param epoch: _description_, defaults to 0
    :param wandb_prefix: _description_, defaults to "eval"
    :return: _description_
    """
    labels = []
    predictions = []
    model.eval()
    for batch in dataloader:
        # batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits.cpu()
        prediction = _score(logits=logits, mode=mode)
        predictions.extend(prediction)

        labels.extend(batch["labels"].cpu().numpy())

    metrics = _compute_metric(labels, predictions, mode=mode)
    if not no_wandb:
        wandb.log({
            f"{wandb_prefix}_epoch": epoch,
            f"{wandb_prefix}_f1": metrics["f1"], 
            f"{wandb_prefix}_f1_macro": metrics["f1_macro"], 
            f"{wandb_prefix}_f1_cls1": metrics["f1_cls1"],
            f"{wandb_prefix}_f1_cls0": metrics["f1_cls0"],
            f"{wandb_prefix}_accuracy": metrics["accuracy"],
            f"{wandb_prefix}_precision": metrics["precision"],
            f"{wandb_prefix}_recall": metrics["recall"],
            })
    # logger.info("Metrics: ", metrics)
    return metrics, predictions


def _train(model, epochs, optimizer, lr_scheduler, num_training_steps, 
           train_dataloader, eval_dataloader, accelerator, mode, no_wandb):
    progress_bar = tqdm(range(num_training_steps))

    model.train()
    for epoch in range(epochs):
        for batch in train_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}  disabled for accelerate
            outputs = model(**batch)
            loss = outputs.loss
            # loss.backward()  disabled for accelerate
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            if not no_wandb:
                wandb.log({"train_loss": loss.cpu()})
        logger.info(f"Evaluate after epoch {epoch}")
        _eval(model, eval_dataloader, epoch=epoch, mode=mode, no_wandb=no_wandb)
    return model

def run_bert_experiment(
    checkpoint, tokenizer_checkpoint, training, validation, test,
    savepoint, epochs, batches, lr, name,
    num_labels=2, mode='binary', cache_dir: str|None = None, no_wandb: bool = False
):  
    if not no_wandb:
        run = wandb.init(
            project=f"triggers-acl24-segment-classification-sweep",
            name=name,
                config={
                    "learning_rate": lr,
                    "epochs": epochs
                },
            )

    logger.info("load dataset and tokenizer")
    tokenized_datasets, test_dataset_with_metadata, tokenizer = \
        _load_dataset(tokenizer_checkpoint, training, validation, test, mode, cache_dir=cache_dir)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accelerator = Accelerator()

    logger.info("define models and parameters")
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=batches, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=batches, collate_fn=data_collator
    )
    test_dataloader = DataLoader(
        tokenized_datasets["test"], batch_size=batches, collate_fn=data_collator
    )
    if cache_dir:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, 
            num_labels=num_labels, 
            cache_dir=cache_dir,
            problem_type="multi_label_classification" if mode == "multilabel" else "single_label_classification"
        )
    else: 
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, 
            num_labels=num_labels, 
            cache_dir=cache_dir,
            problem_type="multi_label_classification" if mode == "multilabel" else "single_label_classification"
        )
    optimizer = AdamW(model.parameters(), lr=lr)
    # on MacOS, use mps to avoid CUDA out of memory error
    # use 'export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0' command to disable
    # device = "cpu" if use_cpu else torch.device("cuda")  # disabled for use with accelerate
    # model.to(device)  # disabled for use with accelerate
    num_training_steps=epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    logger.info("start model training")
    # Accelerate dataloading
    train_dataloader, eval_dataloader, test_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader, model, optimizer
    ) 
    model = _train(model, epochs, optimizer, lr_scheduler, 
                   num_training_steps, train_dataloader, eval_dataloader, 
                   accelerator, mode=mode, no_wandb=no_wandb)
    
    logger.info("save trained model")
    model.save_pretrained(savepoint)
    # model = AutoModelForSequenceClassification.from_pretrained(savepoint) 
    logger.info("evlauate on test")
    test_metrics, test_predictions = _eval(model, test_dataloader, wandb_prefix="test", mode=mode, no_wandb=no_wandb)
    open(savepoint + "/test-results.json", 'w').write(json.dumps(test_metrics))

    test_dataset_with_metadata = test_dataset_with_metadata.add_column("predictions", test_predictions)   
    test_dataset_with_metadata = test_dataset_with_metadata.remove_columns(["input_ids", "attention_mask"])

    if mode == "multilabel":
        test_dataset_with_metadata = test_dataset_with_metadata.map(
            lambda x: {"int_labels": x["labels"].to(torch.int), 
                       "int_predictions": x["predictions"].to(torch.int)}, 
                       remove_columns=["labels", "predictions"]
        ).rename_column("int_labels", "labels").rename_column("int_predictions", "predictions")

    test_dataset_with_metadata.to_json(f"{savepoint}/test-predictions.jsonl")
    # probably not neccessary
    # torch.cuda.empty_cache()
    if not no_wandb:
        wandb.finish(quiet=True)
    

if __name__ == "__main__":
    exit("Call this via main.py")