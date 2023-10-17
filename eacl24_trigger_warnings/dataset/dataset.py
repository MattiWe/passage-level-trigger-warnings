import wasabi
from typing import List, Dict, Tuple
from collections import namedtuple
from pathlib import Path
from statistics import mode
from random import shuffle, seed
import json

from eacl24_trigger_warnings.utility import major_warning, warning_map, preprocess

msg = wasabi.Printer()

TriggerExample = namedtuple('TriggerExample', ('id', 'text', 'unanimous', 'warning', 'answer'))


def _label_inference(votes, target_warning, label_mode):
    keys = list(major_warning.keys())
    inferred_warning = mode(votes) if mode(votes) >= 0 else votes[0]

    if label_mode == 'multi':
        return [inferred_warning if target_warning == warning else 0 for warning in keys]
    elif label_mode == 'binary':
        return inferred_warning
    elif label_mode == 'null':
        return 0


def _write_sample(sample, path: Path, name, label_mode='multi'):
    path.mkdir(exist_ok=True, parents=True)

    s = [{
        "id": _["example_id"],
        "text": _["text"],
        "labels": _label_inference(_["labels"], _["warning"], label_mode),
    } for _ in sample]

    open(path / name, 'w').writelines([f"{json.dumps(_)}\n" for _ in s])


def _make_training_settings(train: list, output_path: Path):
    """ create the 3 settings from the train-test-split """
    # setting A
    train_a = [d for d in train if mode(d["labels"]) == 1]
    _write_sample(train_a, output_path / "oneclass-strict", "train.jsonl")
    # setting B - this requires two separate training sets for technical reasons
    #  one classifies the violence and the other the discrimination classes
    train_b_discrimination = [d for d in train
                              if (mode(d["labels"]) == 1 and major_warning[d["warning"]] == "discrimination")
                              or (mode(d["labels"]) == 0 and major_warning[d["warning"]] == "violence")]
    _write_sample(train_b_discrimination, output_path / "oneclass-lenient", "train-discrimination.jsonl")
    train_b_violence = [d for d in train
                        if (mode(d["labels"]) == 1 and major_warning[d["warning"]] == "violence")
                        or (mode(d["labels"]) == 0 and major_warning[d["warning"]] == "discrimination")]
    _write_sample(train_b_violence, output_path / "oneclass-lenient", "train-violence.jsonl")

    # setting C - here we create a training dataset for each class
    for warning in major_warning.keys():
        _ = [d for d in train
             if (d["warning"] == warning)
             or (major_warning[d["warning"]] != major_warning[warning])]
        _write_sample(_, output_path / "binary-extended", f"train-{warning}.jsonl", label_mode='binary')


def _make_test_setting(test: list, output_path: Path):
    _write_sample(test, output_path / "test", "test.jsonl")
    _write_sample(test, output_path / "test-binary", f"test-all.jsonl", label_mode='null')

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


def extract_data(ls_annotations: List[dict], annotator_indices: Dict[str, str]) -> Tuple[str, dict]:
    for annotation in ls_annotations:
        a_id = int(annotation.get("source_id", annotation["id"]))
        if len(annotation['sentence']) == 0:
            msg.warn(f"sentence is None for {a_id}")
            continue
        elif "annotator" not in annotation:
            msg.warn(f"annotator is None for {a_id}")
            continue
        elif "trigger" not in annotation:
            msg.warn(f"trigger is None for {a_id}")
            continue
        example = {"warning": warning_map.get(annotation["tw"]), "labels": [-1, -1, -1],
                   "text": compose_text(annotation),
                   "work_id": annotation.get("ref_if", annotation.get("ref_id", None))[0],
                   "example_id": a_id,
                   "set_id": 1 if annotation["tw"].endswith("Test") else 0}
        a_idx = annotator_indices.get(annotation["annotator"], 2)
        example["labels"][a_idx] = 1 if annotation["trigger"] == "Positive" else 0
        # Set all labels that are still -1 to the value of the primary (index 0) vote
        example["labels"] = [vote if vote >= 0 else example["labels"][0] for vote in example["labels"]]

        yield a_id, example


def compile_dataset(ls_primary: List[dict], ls_additional: List[List[dict]], known_ids: tuple):
    """ This method loads the annotation results (from Label Studio) and complies the raw dataset
    cf. `main.py#parse_ls` for additional documentation.
    """
    # determine, which annotator gets which position in the array of annotations.
    annotator_indices = {int(annotator_id): idx for idx, annotator_id in enumerate(known_ids)}
    print(annotator_indices)
    # parse primary
    msg.good("Loading primary annotations")
    dataset = {example_id: data for example_id, data in extract_data(ls_primary, annotator_indices)}

    # add additional annotations from primary and additional ls files
    msg.good("Loading additional annotations")
    for additional in ls_additional:
        for idx, data in extract_data(additional, annotator_indices):
            if idx not in dataset:
                msg.warn(f"annotation {idx} from additional not in primary list")
                continue
            dataset[idx]["labels"] = [previous_vote if current_vote == -1 else current_vote
                                      for previous_vote, current_vote in zip(dataset[idx]["labels"], data["labels"])]
    return list(dataset.values())


def create_experiment_datasets(dataset_path: Path, output_path: Path):
    """ This method creates the experiment files from the dataset complied by `compile_dataset`
    cf. `main.py#prepare_experiments` for additional documentation.
    """
    seed(42)
    dataset = [json.loads(d) for d in open(dataset_path).readlines()]

    # 1. create the in-distribution train/test split
    ood_train = [d for d in dataset if d["set_id"] == 0]
    ood_test = [d for d in dataset if d["set_id"] == 1]

    _make_test_setting(ood_train, output_path / "ood")
    _make_training_settings(ood_test, output_path / "ood")

    # 2. create the out-of-distribution train/test split
    shuffle(ood_test)
    shuffle(ood_train)
    train_half_index = int(round(len(ood_train) * 0.8, 0))
    test_half_index = int(round(len(ood_test) * 0.8, 0))
    id_train = ood_train[:train_half_index] + ood_test[:test_half_index]
    id_test = ood_train[train_half_index:] + ood_test[test_half_index:]
    _make_test_setting(id_test, output_path / "id")
    _make_training_settings(id_train, output_path / "id")

    # save calibration examples - 20 positive and 20 negative of each label
    calibration = id_train[:train_half_index]

    for warning in major_warning.keys():
        examples_with_warning = [d for d in calibration if d["warning"] == warning]
        calibration_selection = [_ for _ in examples_with_warning if _label_inference(_["labels"], warning, "binary")][:20]
        calibration_selection.extend([_ for _ in examples_with_warning if not _label_inference(_["labels"], warning, "binary")][:20])
        _write_sample(calibration_selection, output_path / "calibration", f"test-{warning}.jsonl", label_mode='binary')


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
