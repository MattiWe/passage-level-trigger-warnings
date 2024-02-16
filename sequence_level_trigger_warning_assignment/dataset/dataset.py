import wasabi
from typing import List, Dict, Tuple
from collections import namedtuple
from pathlib import Path
from statistics import mode
from random import shuffle, seed, choice
import json

from sequence_level_trigger_warning_assignment.utility import major_warning, warning_map, preprocess, infer_label
from sequence_level_trigger_warning_assignment.dataset.prompt import Prompt

msg = wasabi.Printer(ignore_warnings=True)

TriggerExample = namedtuple('TriggerExample', ('id', 'text', 'unanimous', 'warning', 'answer'))


def _format_label(date, label_mode):
    keys = list(major_warning.keys())
    inferred_warning = date["labels"]
    target_warning = date["warning"]

    if label_mode == 'multilabel':
        return [inferred_warning if target_warning == warning else 0 for warning in keys]
    elif label_mode == 'multiclass':
        if inferred_warning == 1:
            return keys.index(target_warning)
        else:
            return len(keys)
    elif label_mode == 'binary':
        return inferred_warning
    elif label_mode == 'null':
        return 0


def _write_sample(sample, path: Path, name, label_mode='multilabel'):
    path.mkdir(exist_ok=True, parents=True)

    s = [{
        "id": _["example_id"],
        "text": _["text"],
        "labels": _format_label(_, label_mode),
    } for _ in sample]

    open(path / name, 'w').writelines([f"{json.dumps(_)}\n" for _ in s])


def _write_prompts(sample, output: Path, prompt_path: Path):
    output.mkdir(exist_ok=True, parents=True)   
    prompts = [
        # Prompt(prompt_resource_dir=prompt_path),
        # Prompt(persona=True, prompt_resource_dir=prompt_path), 
        # Prompt(definition=True, prompt_resource_dir=prompt_path), 
        # Prompt(demonstration=True, prompt_resource_dir=prompt_path),  
        # Prompt(persona=True, definition=True, prompt_resource_dir=prompt_path),
        # Prompt(persona=True, demonstration=True, prompt_resource_dir=prompt_path),  
        # Prompt(definition=True, demonstration=True, prompt_resource_dir=prompt_path), 
        Prompt(persona=True, definition=True, demonstration=True, prompt_resource_dir=prompt_path),  
    ]
    for prompt in prompts:
        with open(output / f"test-{prompt}.jsonl", 'w') as of:
            for example in sample:
                p = prompt(text=example['text'], warning=example['warning'])
                e = {"id": example["example_id"], 
                     "text": p, 
                     "warning": example['warning'], 
                     "labels": example['labels']
                     }
                of.write(f"{json.dumps(e)}\n")


def _make_training_settings(train: list, output_path: Path):
    """ create the 3 settings from the train-test-split """
    # setting A
    _write_sample(train, output_path / "multilabel", "train.jsonl")

    train_a = [d for d in train if d["labels"] == 1]
    _write_sample(train_a, output_path / "multilabel-negatives", "train.jsonl")

    # setting B - this requires two separate training sets for technical reasons
    #  one classifies the violence and the other the discrimination classes
    train_b_discrimination = [d for d in train
                              if (d["labels"] == 1 and major_warning[d["warning"]] == "discrimination")
                              or (d["labels"] == 0 and major_warning[d["warning"]] == "violence")]
    _write_sample(train_b_discrimination, output_path / "multilabel-extended", "train-discrimination.jsonl")
    train_b_violence = [d for d in train
                        if (d["labels"] == 1 and major_warning[d["warning"]] == "violence")
                        or (d["labels"] == 0 and major_warning[d["warning"]] == "discrimination")]
    _write_sample(train_b_violence, output_path / "multilabel-extended", "train-violence.jsonl")

    # setting C - here we create a training dataset for each class
    for warning in major_warning.keys():
        _ = [d for d in train if d["warning"] == warning]
        _write_sample(_, output_path / "binary", f"train-{warning}.jsonl", label_mode='binary')

        _ = [d for d in train
             if (d["warning"] == warning)
             or (major_warning[d["warning"]] != major_warning[warning])]
        _write_sample(_, output_path / "binary-extended", f"train-{warning}.jsonl", label_mode='binary')

    # setting D - multiclass
    _write_sample(train, output_path / "multiclass", "train.jsonl", label_mode='multiclass')


def _make_test_setting(test: list, output_path: Path, prompt_path: Path):
    """
    :param test: _description_
    :param output_path: _description_
    """

    _write_sample(test, output_path / "test-multilabel", "test.jsonl", label_mode='multilabel')
    _write_sample(test, output_path / "test-binary", f"test.jsonl", label_mode='binary')
    _write_sample(test, output_path / "test-multiclass", f"test.jsonl", label_mode='multiclass')
    _write_prompts(test, output_path / "test-prompts", prompt_path)

    # binary-extended test cases (to evaluate training of the
    for warning in major_warning.keys():
        _ = [d for d in test
             if (d["warning"] == warning)]
             # or (major_warning[d["warning"]] != major_warning[warning])]  # this add the negative examples from other cases too.
        _write_sample(_, output_path / "test-binary", f"test-{warning}.jsonl", label_mode='binary')


def compose_text(annotation):
    if not isinstance(annotation['before_context'], list):
        annotation['before_context'] = ["", ""]
    annotation['before_context'].extend(["", ""])
    return f"{preprocess(annotation['before_context'][0])} " \
           f"{preprocess(annotation['before_context'][1])} " \
           f"{preprocess(annotation['sentence'][0])} " \
           f"{preprocess(annotation['after_context'][0])} " \
           f"{preprocess(annotation['after_context'][1])}".strip()


def _extract_data(ls_annotations: List[dict], primary_rater: int) -> Tuple[str, dict]:
    for annotation in ls_annotations:
        a_id = int(annotation.get("source_id", annotation["id"]))
        if annotation["annotator"] != primary_rater:
            msg.warn(f"not primary rater for {a_id}")
            continue
        if len(annotation['sentence']) == 0:
            msg.warn(f"sentence is None for {a_id}")
            continue
        elif "annotator" not in annotation:
            msg.warn(f"annotator is None for {a_id}")
            continue
        elif "trigger" not in annotation:
            msg.warn(f"trigger is None for {a_id}")
            continue
        example = {"warning": warning_map.get(annotation["tw"]), 
                   "labels": [-1, -1, -1],
                   "text": compose_text(annotation),
                   "work_id": annotation.get("ref_if", annotation.get("ref_id", None))[0],
                   "example_id": a_id,
                   "set_id": 1 if annotation["tw"].endswith("Test") else 0}
        yield a_id, example


def _extract_votes(ls_annotations: List[dict], annotator_indices: Dict[str, str]):
    for annotation in ls_annotations:
        a_id = int(annotation.get("source_id", annotation["id"]))
        a_idx = annotator_indices.get(annotation["annotator"], 2)
        # Skip the evil annotator
        if annotation["annotator"] == 68 and (annotation["tw"] == "Racism - Test" or annotation["tw"] == "Racism - Train"):
            continue
        try:
            vote = 1 if annotation["trigger"] == "Positive" else 0
        except KeyError:
            msg.warn(f"trigger is None for {a_id}")
            vote = 0
        yield a_id, a_idx, vote


def compile_dataset(ls_primary: List[dict], ls_additional: List[List[dict]], known_ids: tuple):
    """ This method loads the annotation results (from Label Studio) and complies the raw dataset
    cf. `main.py#parse_ls` for additional documentation.    
    """
    # determine, which annotator gets which position in the array of annotations.
    annotator_indices = {int(annotator_id): idx for idx, annotator_id in enumerate(known_ids)}
    msg.info(annotator_indices)
    # parse primary (find examples without labels)
    msg.good("Loading primary annotations")
    dataset = {example_id: data for example_id, data in _extract_data(ls_primary, primary_rater=63)}
    msg.info(f"number of examples: {len(dataset)}")

    # add additional annotations from primary and additional ls files
    msg.good("Loading additional annotations")

    for a_id, a_idx, vote in _extract_votes(ls_primary, annotator_indices):
        if a_id not in dataset:
            msg.warn(f"annotation {a_id} from primary not in primary list")
            continue
        dataset[a_id]["labels"][a_idx] = vote

    for additional in ls_additional:
        for a_id, a_idx, vote in _extract_votes(additional, annotator_indices):
            if a_id not in dataset:
                msg.warn(f"annotation {a_id} from additional not in primary list")
                continue
            dataset[a_id]["labels"][a_idx] = vote

    # TODO temporarily replace missing votes with a random choice
    for d in dataset.values():
        for idx, vote in enumerate(d["labels"]):
            if vote == -1:
                _ = [v for v in d["labels"] if v != -1]
                d["labels"][idx] = choice(_)

    return list(dataset.values())


def _sample(dataset: List, balance="20") -> Tuple:
    """_summary_

    :param dataset: List of dicts with {"warning", "labels", "text", "work_id", "example_id", "set_id"}
    :param balance: how many elements of both classes to include
    :return: a tuple (list of the samples element, list of the remaining elements)
    """
    positives = {}
    negatives = {}
    sample = []
    rest = []
    for date in dataset:
        if date["labels"] == 1:
            positives.setdefault(date["warning"], []).append(date)
        else:
            negatives.setdefault(date["warning"], []).append(date)
    
    for warning in major_warning.keys():
        shuffle(positives[warning])
        shuffle(negatives[warning])
        sample.extend(positives[warning][:balance])
        sample.extend(negatives[warning][:balance])
        rest.extend(positives[warning][balance:])
        rest.extend(negatives[warning][balance:])
    return sample, rest


def create_experiment_datasets(dataset_path: Path, output_path: Path, prompt_path: Path, inference="majority", rng_seed: int = 42):
    """ This method creates the experiment files from the dataset complied by `compile_dataset`
    cf. `main.py#prepare_experiments` for additional documentation.

    - in-distribution: 40 test, rest train. sample test and validation balanced
    - out-of-distribution: set 2 is test
    """
    seed(rng_seed)
    dataset = [json.loads(d) for d in open(dataset_path).readlines()]
    dataset = [infer_label(d, method=inference) for d in dataset]

    # 1. create the out-of-distribution train/test split
    # take set 2 for testing, because set 1 has more positive examples (at least in the majority inference)
    ood_train = [d for d in dataset if d["set_id"] == 0]
    ood_test = [d for d in dataset if d["set_id"] == 1]
    _make_training_settings(ood_train, output_path / f"ood-{inference}")
    ood_test_undersampled, _ = _sample(ood_test, balance=20)  # downsample and balance the test dataset
    print([_["example_id"] for _ in ood_test_undersampled[:5]])
    _make_test_setting(ood_test_undersampled, output_path / f"ood-{inference}", prompt_path)

    # 2. create the in-distribution train/test split
    id_test, id_train = _sample(dataset, balance=20)
    print([_["example_id"] for _ in id_test[:5]])
    _make_training_settings(id_train, output_path / f"id-{inference}")
    _make_test_setting(id_test, output_path / f"id-{inference}", prompt_path)

    # # save calibration examples - 20 positive and 20 negative of each label
    # calibration = id_train[:train_half_index]

    # for warning in major_warning.keys():
    #     examples_with_warning = [d for d in calibration if d["warning"] == warning]
    #     calibration_selection = [_ for _ in examples_with_warning if _label_inference(_["labels"], warning, "binary")][:20]
    #     calibration_selection.extend([_ for _ in examples_with_warning if not _label_inference(_["labels"], warning, "binary")][:20])
    #     _write_sample(calibration_selection, output_path / "calibration", f"test-{warning}.jsonl", label_mode='binary')


def create_few_shot_ablation_datasets(dataset: Path, output: Path, prompt_path: Path, inference='majoity'):
    """ Create dataset for the few shot/prompt experiments
    1. dataset for the prompt ablation
        - id-majority training dataset 

    :param dataset: Path to the dataset path
    """
    output.mkdir(exist_ok=True)
    data = [json.loads(line) for line in open(dataset, 'r')]
    data = [infer_label(d, method=inference) for d in data]
    sample, _ = _sample(data, 30)
    prompts = [
        Prompt(prompt_resource_dir=prompt_path),
        Prompt(persona=True, prompt_resource_dir=prompt_path), 
        Prompt(definition=True, prompt_resource_dir=prompt_path), 
        Prompt(demonstration=True, prompt_resource_dir=prompt_path),  
        Prompt(persona=True, definition=True, prompt_resource_dir=prompt_path),
        Prompt(persona=True, demonstration=True, prompt_resource_dir=prompt_path),  
        Prompt(definition=True, demonstration=True, prompt_resource_dir=prompt_path), 
        Prompt(persona=True, definition=True, demonstration=True, prompt_resource_dir=prompt_path),  
    ]
    for prompt in prompts:
        with open(output / f"{prompt}.jsonl", 'w') as of:
            for example in sample:
                p = prompt(text=example['text'], warning=example['warning'])
                e = {"id": example["example_id"], 
                     "text": p, 
                     "warning": example['warning'], 
                     "labels": example['labels']
                     }
                of.write(f"{json.dumps(e)}\n")


def collect_icl_examples(dataset_path: Path, output_path: Path):
    seed(42)

    examples = []
    dataset = [json.loads(d) for d in open(dataset_path).readlines()]

    # Sort examples by unanimous decisions
    for d in dataset:
        examples.append(TriggerExample(id=d["example_id"], text=d['text'],
                                       unanimous=sum(d['labels']) == 0 or sum(d['labels']) == 3,
                                       warning=d['warning'],
                                       answer='positive' if mode(d['labels']) == 1 else 'negative'
                                       ))

    shuffle(examples)
    # write examples file
    unanimous_results = {}
    non_unanimous_results = {}
    for example in examples:
        if example.unanimous:
            unanimous_results.setdefault(example.warning, {})\
                .setdefault(example.answer, [])\
                .append(example.text)
        else:
            non_unanimous_results.setdefault(example.warning, {})\
                .setdefault(example.answer, [])\
                .append(example.text)

    for warning, _w in unanimous_results.items():
        unanimous_results[warning]["negative"] = _w["negative"][:10]
        unanimous_results[warning]["positive"] = _w["positive"][:10]

    for warning, _w in non_unanimous_results.items():
        non_unanimous_results[warning]["negative"] = _w["negative"][:10]
        non_unanimous_results[warning]["positive"] = _w["positive"][:10]

    open(output_path / "gpt-prompt-unanimous-demonstrations.json", 'w').write(json.dumps(unanimous_results))
    open(output_path / "gpt-prompt-non-unanimous-demonstrations.json", 'w').write(json.dumps(non_unanimous_results))
