import json
import logging
from tqdm import tqdm
from time import sleep
from typing import List
from statistics import mean

from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np

from pathlib import Path
import openai
from openai.error import ServiceUnavailableError, Timeout, RateLimitError
import vertexai
from vertexai.language_models import TextGenerationModel
from google.api_core.exceptions import InternalServerError


warning_idx = {0: "misogyny", 1: "racism", 2: "ableism", 3: "homophobia",
               4: "death", 5: "violence", 6: "abduction", 7: "war"}


class Prompt:
    def __init__(self, persona: bool = False, definition: bool = False, extended_instruction: bool = False,
                 demonstration: bool = False, explanation: bool = False,
                 num_unanimous: int = 1, num_non_unanimous: int = 1):
        """

        :param persona:
        :param definition:
        :param extended_instruction: Add the prompt part `extended-instruction`
        :param demonstration:
        :param explanation:
        :param num_unanimous:
        :param num_non_unanimous:
        """
        self.has_persona = persona
        self.has_definition = definition
        self.has_demonstration = demonstration
        self.has_explanation = explanation
        self.num_unanimous = num_unanimous
        self.num_non_unanimous = num_non_unanimous
        self.has_extended_instruction = extended_instruction
        self.prompt_parts = json.loads(open("../../../resources/gpt-prompt-parts.json").read())
        self.demonstrations_unanimous = json.loads(open("../../../resources/gpt-prompt-unanimous-demonstrations.json").read())
        self.demonstrations_non_unanimous = json.loads(open("../../../resources/gpt-prompt-non-unanimous-demonstrations.json").read())
        self.demonstrations_selected = json.loads(open("../../../resources/gpt-prompt-selected-demonstrations.json").read())

    def __str__(self) -> str:
        return f"{'extended-' if self.has_extended_instruction else ''}instruction" \
               f"{'-persona' if self.has_persona else ''}" \
               f"{'-definition' if self.has_definition else ''}" \
               f"{'-explanation' if self.has_explanation else ''}" \
               f"{'-' + str(self.num_unanimous) + '-unanimous' if self.has_demonstration else ''}" \
               f"{'-' + str(self.num_non_unanimous) + '-non-unanimous' if self.has_demonstration else ''}"

    def __call__(self, text: str, warning: str) -> str:
        """
        Construct the query from
        :param text:
        :param warning:
        :return:
        """
        query = ""
        if self.has_persona:
            query += f"{self.prompt_parts[warning]['persona']} " \
                     f"{self.prompt_parts[warning]['definition'] if self.has_definition else ''}\n\n" \

        query += f"{self.prompt_parts[warning]['instruction']} " \
                 f"{self.prompt_parts['extended-instruction'] if self.has_extended_instruction else ''} " \
                 f"{self.prompt_parts['explanation'] if self.has_explanation else ''} " \
                 f"\n\n"

        if self.has_demonstration:
            for idx in range(0, max(self.num_non_unanimous, self.num_unanimous)):
                if idx < self.num_unanimous:
                    query += f"{self.prompt_parts['text_prefix']} {self.demonstrations_unanimous[warning]['positive'][idx]}\n" \
                             f"{self.prompt_parts['tw_prompt']} {self.prompt_parts['yes']}\n\n"
                    query += f"{self.prompt_parts['text_prefix']} {self.demonstrations_unanimous[warning]['negative'][idx]}\n" \
                             f"{self.prompt_parts['tw_prompt']} {self.prompt_parts['no']}\n\n"

                if idx < self.num_non_unanimous:
                    query += f"{self.prompt_parts['text_prefix']} {self.demonstrations_non_unanimous[warning]['positive'][idx]}\n" \
                             f"{self.prompt_parts['tw_prompt']} {self.prompt_parts['yes']}\n\n"
                    query += f"{self.prompt_parts['text_prefix']} {self.demonstrations_non_unanimous[warning]['negative'][idx]}\n" \
                             f"{self.prompt_parts['tw_prompt']} {self.prompt_parts['no']}\n\n"

        query += f"{self.prompt_parts['text_prefix']} {text}\n{self.prompt_parts['tw_prompt']} "
        return query

    def parse(self, response, api='gpt3'):
        if api == 'bison':
            if response.is_blocked:
                text = ""
            else:
                text = response.text.lower().strip()
        elif api == 'gpt3' or api == 'gpt4':
            text = response.choices[0].message.content.lower().strip()

        if not (text.startswith("yes") or
                text.startswith("warning: yes") or
                text.startswith("warning: no") or
                text.startswith("no")):
            logging.info(f'invalid response: {text}')
        if self.prompt_parts['yes'] in text:
            return 1
        elif self.prompt_parts['no'] in text:
            return 0
        elif text == '':
            # Here the generation was probably blocked for safety reasons
            logging.warning('empty text')
            return 1
        else:
            logging.error(f"error parsing response {response}")
            return 0


def _score(truth: List[list], predictions: List[list]) -> dict:
    """
    The order of the lists should follow `warning_idx`

    :param truth: list of lists - one list for each class, each index is the true score (1, 0) for the class
    :param predictions: list of lists - one list for each class, each index is the predicted score (1, 0) for the class
    :return: scores
    """
    results = {"f1": {}, "precision": {}, "recall": {}}

    for idx, label in enumerate(warning_idx.values()):
        y_true = np.asarray(truth[idx])
        y_predicted = np.asarray(predictions[idx])

        results["f1"][label] = round(
            f1_score(y_true, y_predicted, average='binary', labels=[1], zero_division=0), 2)
        results["precision"][label] = round(
            precision_score(y_true, y_predicted, average='binary', labels=[1], zero_division=0), 2)
        results["recall"][label] = round(
            recall_score(y_true, y_predicted, average='binary', labels=[1], zero_division=0), 2)

    results["f1"]["mean"] = round(mean([r for r in results["f1"].values()]), 2)
    results["precision"]["mean"] = round(mean([r for r in results["precision"].values()]), 2)
    results["recall"]["mean"] = round(mean([r for r in results["recall"].values()]), 2)

    return results


def _load_dataset(test: Path):
    """ load examples for the `binary` test case, where there is a file per label
    which contains all examples for that class
    """
    for _file in test.glob("*.jsonl"):
        warning = _file.stem.removeprefix("test-").removeprefix("responses-").removesuffix(".jsonl")
        if warning not in warning_idx.values():
            logging.warning(f"Encountered unknown warning: {warning} - skip the file {_file.stem}")
            continue
        # if warning in {"ableism", "war", "homophobia", "racism", "violence", "abduction"}:  # TODO remove "misogyny", "death"
        #     logging.warning(f"Skipping warning: {warning} - skip the file {_file.stem}")
        #     continue
        for line in open(_file).readlines():
            l = json.loads(line)
            yield warning, l


def _call_vertex(_warning: str, _example: dict, _prompt: Prompt) -> dict:
    """Ideation example with a Large Language Model"""
    p = _prompt(text=_example['text'], warning=_warning)
    parameters = {
        "temperature": 0,  # Temperature controls the degree of randomness in token selection.
        "max_output_tokens": 20,  # Token limit determines the maximum amount of text output.
        # "top_p": 0.8,  # Tokens are selected from most probable to least until the sum of their probabilities equals the top_p value.
        # "top_k": 40,  # A top_k of 1 means the selected token is the most probable among all tokens.
    }

    model = TextGenerationModel.from_pretrained("text-bison")
    try:
        response = model.predict(p, **parameters,)
    except InternalServerError as e:
        logging.error(e)
        response = {"is_blocked": False, "text": "error"}
    return {
        "id": _example["id"],
        "response": response
    }


def _call(_warning: str, _example: dict, _prompt: Prompt, model) -> dict:
    """Call the API until it succeeds, whatever the cost."""
    if model == 'gpt3':
        model = "gpt-3.5-turbo"
    elif model == 'gpt4':
        model = "gpt-4"
    p = _prompt(text=_example['text'], warning=_warning)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": p}],
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
            return {
                "id": _example["id"],
                "response": response
            }


def run_gpt_experiment(test: Path, savepoint: Path, key: str, prompts: List[Prompt], model='gpt3') -> None:
    """
    5 Settings:
    - Instruction only
    - Instruction + Persona
    - Instruction + Persona + Definitions
    - Instruction + ICL
    - Instruction + Explanations
    ---
    - Instruction + ICL + Explanations
    - Instruction + Persona + Definitions + Explanations
    - Instruction + Persona + Definitions + ICL
    - Instruction + Persona + Definitions + ICL + Explanations

    :param test: path to the directory with `test-label.jsonl` files
    :param savepoint:
    :param key:
    :param prompts: A list of Prompt elements that should be run
    :param model:
    :return:
    """
    logging.info("load dataset")
    openai.api_key = key
    vertexai.init(project='each23-trigger-promts', location='us-central1')

    dataset = {}
    for warning, example in _load_dataset(test):
        dataset.setdefault(warning, []).append(example)

    for prompt in prompts:
        logging.warning(f"Testing for prompt {prompt}")
        (savepoint / f"{prompt}").mkdir(exist_ok=True, parents=True)
        open(savepoint / f"{prompt}" / f"prompt.txt", 'w').write(prompt("some text", "racism"))

        results = {}
        for warning, examples in dataset.items():
            if model == 'bison':
                responses = [_call_vertex(warning, example, prompt)
                             for example in tqdm(examples, desc=f"examples for {warning}", total=len(examples))]
            else:
                responses = [_call(warning, example, prompt, model)
                             for example in tqdm(examples, desc=f"examples for {warning}", total=len(examples))]
            results[warning] = [{
                "id": response["id"],
                "prediction": prompt.parse(response["response"], api=model),
            } for response in responses]

            open(savepoint / f"{prompt}" / f"responses-{warning}.jsonl", 'w').writelines([
                f"{json.dumps(_)}\n" for _ in results[warning]
            ])

        # evaluation
        scores = _score(
            [[example["labels"] for example in dataset[warning]] for warning in warning_idx.values()],
            [[example["prediction"] for example in results[warning]] for warning in warning_idx.values()])
        open(savepoint / f"{prompt}" / f"{model}-{prompt}.json", 'w').write(json.dumps(scores))


def score(examples: Path, predictions: Path):
    """
    Only score the stored results

    :param examples: Path to the truth (binary test files)
    :param predictions: Path to the directory with the responses-<warning>-.jsonl files
    """

    dataset = {}
    for warning, example in _load_dataset(Path(examples)):
        dataset.setdefault(warning, []).append(example)

    results = {}
    for warning, example in _load_dataset(Path(predictions)):
        results.setdefault(warning, []).append(example)

    # evaluation
    scores = _score(
        [[example["labels"] for example in dataset[warning]] for warning in warning_idx.values()],
        [[example["prediction"] for example in results[warning]] for warning in warning_idx.values()])
    open(Path(predictions) / f"scores.json", 'w').write(json.dumps(scores))


if __name__ == "__main__":
    exit("Call this via main.py")
