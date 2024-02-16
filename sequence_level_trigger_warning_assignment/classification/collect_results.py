from pathlib import Path
import json
import shutil
from typing import List

from sequence_level_trigger_warning_assignment.utility import major_warning, multilabel_to_multiclass


def collect_1cl(violence_predictions: Path, discrimination_predictions: Path, 
                output_file: Path):
    predictions_violence = [json.loads(line) for line in open(violence_predictions).readlines()]
    # predictions_violence = {_["id"]: _ for _ in predictions_violence}

    predictions_discrimination = [json.loads(line) for line in open(discrimination_predictions).readlines()]
    # predictions_discrimination = {_["id"]: _ for _ in predictions_discrimination}
    with open(output_file, 'w') as of:
        for violence_pred, discrimination_pred in zip(predictions_violence, predictions_discrimination):
            # sanity check if they're the same examples
            if violence_pred["id"] != discrimination_pred["id"]:
                print("Ids are not the same")
                exit()
            # we take predictions on index 0--3 from discrimination, rest from violence
            violence_pred["predictions"] = discrimination_pred["predictions"][:3] + violence_pred["predictions"][3:]
            of.write(f"{json.dumps(violence_pred)}\n")


def collect_mc(source_dir: Path, output_file: Path):
    """ Convert the output predictions of a muliclass classifier 
        {"id": int, "labels":int, "prediction": int}
        into the multi-label format used by the scoring function.

    :param source_dir: The `savedir` that contains the test-predictions.jsonl. 
    :param output_dir: Directory where to save the `model-name.jsonl` with the scores
    :param model_name: name of the model used to name the output file
    """
    # merge oneclass-lenient
    predictions_path = source_dir / "test-predictions.jsonl"
    predictions = [json.loads(line) for line in open(predictions_path).readlines()]

    def _to_array_form(label: int):
        new_label = [0] * len(major_warning)
        if label < len(major_warning):
            new_label[label] = 1
        return new_label

    with open(output_file, 'w') as of:
        for p in predictions:
            output_line = {'id': p['id'], 'predictions': _to_array_form(p['predictions'])}
            of.write(f"{json.dumps(output_line)}\n")


def collect_bin(source_dir: Path, output_file: Path):
    with open(output_file, 'w') as of:
        for warning in major_warning.keys():
            predictions_path = source_dir / warning / "test-predictions.jsonl"
            for line in open(predictions_path).readlines():
                of.write(line)


def collect_sampling_experiments(source_dirs: List[Path], target_dir: Path):
    """This is a utility to collect the predictions files in the same place
    
    acl24-fanbert-binary-id-minority-extended-1e-5lr-20ep-44
    acl24-fanbert-multilabel-ood-majority-2e-5lr-20ep-45
    acl24-fanbert-binary-ood-minority-extended-2e-5lr-20ep-45
    acl24-fanbert-binary-ood-minority-extended-2e-5lr-20ep-47
    """
    prompt_file_name = "predictions-instruction-persona-definition-1-unanimous-0-non-unanimous.jsonl"
    for source_dir in source_dirs:
        for model_dir in source_dir.glob("*"):
            name = model_dir.stem
            if (model_dir / f"{name}.jsonl").exists():
                shutil.copyfile(model_dir / f"{name}.jsonl", target_dir / f"{name}.jsonl")
            elif (model_dir / prompt_file_name).exists():
                shutil.copyfile(model_dir / prompt_file_name, target_dir / f"{name}.jsonl")
            else:
                print(name)
                

if __name__ == "__main__":
    # exit("Call this via main.py")
    collect_sampling_experiments([
        Path("/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/triggering-sentences/acl24/models-43"),
        Path("/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/triggering-sentences/acl24/models-44"),
        Path("/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/triggering-sentences/acl24/models-45"),
        Path("/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/triggering-sentences/acl24/models-46"),
        Path("/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/triggering-sentences/acl24/models-47"),
    ], Path("/home/mike4537/code/git-webis-de/code-research/computational-ethics/sequence-level-trigger-warning-assignment/resources/classification-results/predictions") )
