from pathlib import Path
import json
import shutil

from eacl24_trigger_warnings.utility import major_warning


def collect_1cs(source_dir: Path, output_dir: Path, model_name: str):
    shutil.copyfile(source_dir / "test-predictions.jsonl",
                    output_dir / f"{model_name}.jsonl")


def collect_1cl(source_dir: Path, output_dir: Path, model_name: str):
    # merge oneclass-lenient
    predictions_violence_path = source_dir / "violence" / "test-predictions.jsonl"
    predictions_violence = [json.loads(line) for line in open(predictions_violence_path).readlines()]
    predictions_violence = {_["id"]: _ for _ in predictions_violence}

    predictions_discrimination_path = source_dir / "discrimination" / "test-predictions.jsonl"
    predictions_discrimination = [json.loads(line) for line in open(predictions_discrimination_path).readlines()]
    predictions_discrimination = {_["id"]: _ for _ in predictions_discrimination}
    labels = {}
    for idx, l in enumerate(major_warning.items()):
        label, major_label = l
        for example_id in predictions_violence.keys():
            if major_label == 'violence':
                labels.setdefault(example_id, []).append(predictions_violence[example_id]["prediction"][idx])
            elif major_label == 'discrimination':
                labels.setdefault(example_id, []).append(predictions_discrimination[example_id]["prediction"][idx])

    open(output_dir  /f"{model_name}.jsonl", 'w').writelines([
        f"{json.dumps({'id': example_id, 'prediction': l})}\n" for example_id, l in labels.items()
    ])


def collect_bin(source_dir: Path, output_dir: Path, model_name: str):
    labels = {}
    for idx, l in enumerate(major_warning.items()):
        label, major_label = l
        predictions_path = source_dir / label / "test-predictions.jsonl"
        predictions = [json.loads(line) for line in open(predictions_path).readlines()]
        predictions = {_["id"]: _ for _ in predictions}
        for example_id in predictions.keys():
            labels.setdefault(example_id, []).append(predictions[example_id]["prediction"])

    open(output_dir / f"{model_name}.jsonl", 'w').writelines([
        f"{json.dumps({'id': example_id, 'prediction': l})}\n" for example_id, l in labels.items()
    ])


if __name__ == "__main__":
    exit("Call this via main.py")
