import json
import logging
from tqdm import tqdm
from time import sleep
from typing import List, Tuple
from statistics import mean

from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

from pathlib import Path
import openai

from openai.error import ServiceUnavailableError, Timeout, RateLimitError
from sequence_level_trigger_warning_assignment.evaluation.evaluation import _score
from sequence_level_trigger_warning_assignment.dataset.prompt import Prompt


warning_idx = {0: "misogyny", 1: "racism", 2: "ableism", 3: "homophobia",
               4: "death", 5: "violence", 6: "abduction", 7: "war"}


def _call(example: dict, checkpoint: str) -> dict:
    """Call the API until it succeeds, whatever the cost."""
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=checkpoint,
                messages=[{"role": "user", "content": example['text']}],
                timeout=30,
                request_timeout=30,
                temperature=0
            )
        except ServiceUnavailableError as e:
            logging.error(f"{e} - sleeping for 10")
            sleep(10)
        except Timeout as e:
            logging.error(f"{e}")
        except RateLimitError as e:
            logging.error(f"{e} - sleeping for 60")
            sleep(60)
        else:
            return response


def run_gpt_experiment(test: Path, savepoint: Path, key: str, checkpoint: str, name: str) -> None:
    """
    :param test: path to the directory with `test-<prompt>.jsonl` files
    :param savepoint:
    :param key:
    :param model:
    :return:
    """
    logging.info("load dataset")
    openai.api_key = key

    dataset = [json.loads(line) for line in open(test)]
    
    results = []
    with open(savepoint / f"predictions-{name}.jsonl", 'w') as of:
        for example in tqdm(dataset, desc=f"examples for {name}", total=len(dataset)):
            response = _call(example, checkpoint)
            text = response.choices[0].message.content.lower().strip()
            result = {**example, "predictions": Prompt.parse(text)}
            results.append(result)
            of.write(f"{json.dumps(result)}\n")

    scores = _score(
        [example["labels"] for example in results],
        [example["predictions"] for example in results]
        )
    open(savepoint / f"scores-{name}.json", 'w').write(json.dumps(scores))


if __name__ == "__main__":
    exit("Call this via main.py")
