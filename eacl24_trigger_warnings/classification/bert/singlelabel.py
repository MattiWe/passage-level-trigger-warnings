import logging

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _load_dataset(
    checkpoint: str, train_path: str, validation_path: str, test_path: str
):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

    data_files = {"train": train_path, "test": test_path, "validation": validation_path}
    dataset = load_dataset("json", data_files=data_files)
    dataset.set_format("torch")

    tokenized_data = dataset.map(
        tokenize_function, batched=True
    )

    return tokenized_data, tokenizer


def _compute_metric(eval_pred):
    logits, labels = eval_pred
    y_pred = np.argmax(logits, axis=1)

    return {
        "f1": f1_score(y_true=labels, y_pred=y_pred, average="micro"),
        "f1_cls1": f1_score(y_true=labels, y_pred=y_pred, average="binary", pos_label=1),
        "f1_cls0": f1_score(y_true=labels, y_pred=y_pred, average="binary", pos_label=0),
        "accuracy": accuracy_score(labels, y_pred),
        "precision": precision_score(labels, y_pred, average="micro"),
        "recall": recall_score(labels, y_pred, average="micro"),
        "roc_auc": roc_auc_score(labels, y_pred, average="micro"),
    }


def run_single_label_experiment(
    checkpoint, tokenizer_checkpoint, training, validation, test, savepoint, epochs, batches, lr, use_cpu, name
):
    logging.info("load dataset and tokenizer")
    ds, tokenizer = _load_dataset(tokenizer_checkpoint, training, validation, test)

    ds_train = ds["train"]
    ds_validation = ds["validation"]
    ds_test = ds["test"]
    num_labels = 2

    logging.info("start model training")

    args = TrainingArguments(
        savepoint,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batches,
        per_device_eval_batch_size=batches,
        num_train_epochs=epochs,
        load_best_model_at_end=True,
        weight_decay=0.01,
        metric_for_best_model="eval_f1_cls1",
        push_to_hub=False,
        report_to="wandb",
        run_name=name,
    )

    # on MacOS, use mps to avoid CUDA out of memory error
    # use 'export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0' command to disable
    device = "cpu" if use_cpu else "cuda:0"

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels,
    ).to(device)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_validation,
        tokenizer=tokenizer,
        compute_metrics=_compute_metric,
        # resume_from_checkpoint=True,
    )

    trainer.evaluate()
    trainer.train()

    logging.warning("save trained model")
    trainer.save_model(savepoint)

    tokenizer_kwargs = {"padding": True, "truncation": True}
    classifier = pipeline(
        model=trainer.model_wrapped,
        tokenizer=tokenizer,
        task="text-classification",
        device=device,
    )

    def classify(examples):
        p = classifier(examples["text"], **tokenizer_kwargs)
        p = [1 if example["label"] == 'LABEL_1' else 0 for example in p]
        return {"prediction": p}

    logging.warning("make predictions")
    dataset_with_predictions = ds_test.map(
        classify, batched=True, batch_size=batches, remove_columns=["text", "input_ids", "attention_mask"]
    )

    dataset_with_predictions.to_json(f"{savepoint}/test-predictions.jsonl")


if __name__ == "__main__":
    exit("Call this via main.py")
