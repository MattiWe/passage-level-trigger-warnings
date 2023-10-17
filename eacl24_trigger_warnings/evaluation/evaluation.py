import krippendorff
from typing import List
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
from pathlib import Path
import json
import shutil
from statistics import mode, mean, median
from eacl24_trigger_warnings.utility import major_warning
from scipy import stats

import plotly.graph_objects as go


####################################################
# Evaluation of the datasets (counts/agreements)
####################################################
def _load_raters_class_wise(dataset_path: Path):
    dataset = [json.loads(d) for d in open(dataset_path).readlines()]
    raters = {}
    for date in dataset:
        raters.setdefault(date["warning"], []).append(date["labels"])

    for warning, raters in raters.items():
        r1 = [r[0] for r in raters]
        r2 = [r[1] for r in raters]
        r3 = [r[2] for r in raters]
        yield warning, r1, r2, r3


def print_krippendorff_table(dataset_path: Path):
    print("Krippendorff's alpha (nominal)")
    for warning, r1, r2, r3 in _load_raters_class_wise(dataset_path):
        print(f"{warning} & {krippendorff.alpha(reliability_data=[r1, r2, r3], level_of_measurement='nominal')} \\\\")


def anno_stats(dataset_path: Path):
    """
    TODO add mean example length in words
    :param dataset_path:
    :return:
    """
    def _infer(votes):
        return mode(votes) if mode(votes) >= 0 else votes[0]
    dataset = [json.loads(d) for d in open(dataset_path).readlines()]

    counts = {}
    length = {}
    _raters = {}
    for date in dataset:
        counts.setdefault(date["warning"], {}).setdefault(date["set_id"], []).append(_infer(date["labels"]))
        counts.setdefault("total", {}).setdefault(date["set_id"], []).append(_infer(date["labels"]))
        _raters.setdefault(date["warning"], []).append(date["labels"])
        _raters.setdefault("total", []).append(date["labels"])
        words = len(date["text"].split(" "))
        length.setdefault(date["warning"], []).append(words)
        length.setdefault("total", []).append(words)

    print(f"warning & total & length & set 0 positives & set 0 negatives  & set 1 positives & set 1 negatives & alpha")
    for warning, r in _raters.items():
        r1 = [r[0] for r in r]
        r2 = [r[1] for r in r]
        r3 = [r[2] for r in r]
        print(f"{warning} \t& "
              f"{len(counts[warning][0]) + len(counts[warning][1])} \t& "
              f"{round(mean(length[warning]), 0)} \t& "
              f"{sum(counts[warning][0])} \t& {len(counts[warning][0]) - sum(counts[warning][0])} \t& "
              f"{sum(counts[warning][1])} \t& {len(counts[warning][1]) - sum(counts[warning][1])} \t& "
              f"{round(krippendorff.alpha(reliability_data=[r1, r2, r3], level_of_measurement='nominal'), 2)} \\\\")


def keyphrase_stats(keyphrase_file: Path):
    """
    bitch and btch
    Examples:
    :param keyphrase_file:
    :return:
    """
    keyphrases = json.loads(open(keyphrase_file).read())
    print(f"Warning \t& Source Phrase Count \t& cleaned up phrases \t& set 1 \t& set 2")
    for warning_dict in keyphrases:
        print(f"{warning_dict['trigger']} \t& {len(warning_dict['phrases'])} \t& {len(warning_dict['cleaned_up_phrases'])}"
              f"\t& {len(warning_dict['set_1'])} \t& {len(warning_dict['set_2'])}")
    print(f"Total \t& "
          f"{sum([len(_['phrases']) for _ in keyphrases])} \t& "
          f"{sum([len(_['cleaned_up_phrases']) for _ in keyphrases])} \t& "
          f"{sum([len(_['set_1']) for _ in keyphrases])} \t& "
          f"{sum([len(_['set_2']) for _ in keyphrases])}")


def count_test_data(classes: dict,
                    id_test_data_path: Path, ood_test_data_path: Path,
                    id_train_data_path: Path, ood_train_data_path: Path):

    sizes = {"id_test": {}, "ood_test": {}, "id_train": {}, "ood_train": {}}
    for test_file in id_test_data_path.glob("*.jsonl"):
        warning = test_file.stem.removeprefix("test-")
        length = len(open(test_file).readlines())
        sizes['id_test'][warning] = length

    for test_file in ood_test_data_path.glob("*.jsonl"):
        warning = test_file.stem.removeprefix("test-")
        length = len(open(test_file).readlines())
        sizes['ood_test'][warning] = length

    for train_file in id_train_data_path.glob("*.jsonl"):
        warning = train_file.stem.removeprefix("train-")
        length = len([1 for line in open(train_file).readlines() if classes.get(json.loads(line)['id']) == warning])
        sizes['id_train'][warning] = length

    for train_file in ood_train_data_path.glob("*.jsonl"):
        warning = train_file.stem.removeprefix("train-")
        length = len([1 for line in open(train_file).readlines() if classes.get(json.loads(line)['id']) == warning])
        sizes['ood_train'][warning] = length

    print(sizes)
    print("warning \t& id-train \t& id-test \t& ood-train \t& ood-test")
    for warning in sizes["id_train"]:
        print(f"{warning} \t& {sizes['id_train'][warning]} \t& {sizes['id_test'][warning]} "
              f"\t& {sizes['ood_train'][warning]} \t& {sizes['ood_test'][warning]}")

    print(f"Total \t& {sum([l for w, l in sizes['id_train'].items()])} \t& "
          f"{sum([l for w, l in sizes['id_test'].items()])} \t& "
          f"{sum([l for w, l in sizes['ood_train'].items()])} \t& "
          f"{sum([l for w, l in sizes['ood_test'].items()])}")


####################################################
# Evaluation of the experiments
####################################################

def _score(truth: List[list], predictions: List[list]) -> dict:
    """ TODO: add 'correted' F1 where models are only evaluated on those examples for which they're also annotated.
    The order of the lists should follow `warning_idx`

    :param truth: list of lists - one list for each class, each index is the true score (1, 0) for the class
    :param predictions: list of lists - one list for each class, each index is the predicted score (1, 0) for the class
    :return: scores
    """
    results = {"f1": {}, "precision": {}, "recall": {}}

    for idx, label in enumerate(major_warning.keys()):
        y_true = np.asarray(truth[idx], dtype=int)
        y_predicted = np.asarray(predictions[idx], dtype=int)

        results["f1"][label] = round(
            f1_score(y_true, y_predicted, average='binary', zero_division=0), 2)
        results["precision"][label] = round(
            precision_score(y_true, y_predicted, average='binary', zero_division=0), 2)
        results["recall"][label] = round(
            recall_score(y_true, y_predicted, average='binary', zero_division=0), 2)

    results["f1"]["mean"] = round(mean([r for r in results.get("f1").values()]), 2)
    results["precision"]["mean"] = round(mean([r for r in results.get("precision").values()]), 2)
    results["recall"]["mean"] = round(mean([r for r in results.get("recall").values()]), 2)

    return results


def _evaluate_bert_results(dataset: Path, truth_path: Path, predictions_path: Path):
    """ Evaluate the results from the BERT predictions.

    We only evaluate the performance on examples we actually annotated.
    However, for convenience, we made a prediction for each class on all examples (even those that we're not annotated
    for that class)

    Here, we separate the predictions again: For each class, we extract the predictions for only that class from the
    combined output and pass those to _score().
    """
    dataset = [json.loads(example) for example in open(dataset)]

    dataset_by_warning = {}
    for example in dataset:
        dataset_by_warning.setdefault(example["warning"], set()).add(example["example_id"])

    p_data = {json.loads(line)['id']: json.loads(line) for line in open(predictions_path)}
    t_data = {json.loads(line)['id']: json.loads(line) for line in open(truth_path)}

    predictions = [[p_data[example_id]["prediction"][idx]
                    for example_id in t_data.keys() if example_id in dataset_by_warning[warning]]
                   for idx, warning in enumerate(major_warning)]

    truth = [[t_data[example_id]["labels"][idx]
              for example_id in t_data.keys() if example_id in dataset_by_warning[warning]]
             for idx, warning in enumerate(major_warning)]

    return _score(truth, predictions)


def evaluate_classification_results(dataset_path: Path, truth_path: Path, predictions_dir: Path):
    """
    This is to get the scores for the bert predictions.

    :param dataset_path:
    :param truth_path:
    :param predictions_dir:
    :return:
    """
    output_path = predictions_dir / 'scores'
    output_path.mkdir(exist_ok=True)
    for prediction_file in predictions_dir.glob("*.jsonl"):
        scores = _evaluate_bert_results(dataset_path, truth_path, prediction_file)
        open(output_path / f"scores-{prediction_file.stem}.json", 'w').write(json.dumps(scores))
        print(prediction_file, scores)


def _barcharts(scores_path: Path, output_path: Path, figure_name: str):
    def _plot(x: list, y: list, trace_name: list, fig_name: str):
        fig = go.Figure()
        for t_x, t_y, t_name in zip(x, y, trace_name):
            fig.add_trace(go.Bar(x=t_x, y=t_y, name=t_name))

        fig.update_layout(barmode='group', title_text=fig_name,
                          xaxis={'categoryorder': 'total descending'})
        fig.show()
        fig.write_image(output_path / f"{figure_name}.svg")

    # make the f1 mean plot
    scores = {"model": [], "f1": []}
    for model_scores in scores_path.glob("*.json"):
        model_name = model_scores.stem
        _s = json.loads(open(model_scores).read())
        scores["model"].append(model_name)
        scores["f1"].append(_s["f1"]["mean"])
    _plot([scores["model"]], [scores["f1"]], ["f1"], figure_name)

    # make the f1 classwise plot
    scores = {}
    model_names = []
    for model_scores in scores_path.glob("*.json"):
        model_name = model_scores.stem
        model_names.append(model_name)
        _s = json.loads(open(model_scores).read())

        for warning in major_warning:
            scores.setdefault(model_name, {}).setdefault("warning", []).append(warning)
            scores[model_name].setdefault("f1", []).append(_s["f1"][warning])

    _plot(
        [scores[model]["warning"] for model in model_names],
        [scores[model]["f1"] for model in model_names],
        model_names,
        f"{figure_name}-classwise-by-warning"
    )

    scores = {}
    model_names = []
    for model_scores in scores_path.glob("*.json"):
        model_name = model_scores.stem
        model_names.append(model_name)
        _s = json.loads(open(model_scores).read())

        for warning in major_warning:
            scores.setdefault(model_name, []).append(_s["f1"][warning])

    _plot(
        [model_names for model in model_names],
        [scores[model] for model in model_names],
        list(major_warning.keys()),
        f"{figure_name}-classwise-by-model"
    )


def scores_presentation(scores_path: Path, output_path: Path, figure_name: str, plot=True):
    def _plot(y: list, trace_name: list, fig_name: str):
        fig = go.Figure()
        for t_y, t_name in zip(y, trace_name):
            fig.add_trace(go.Box(y=t_y, name=t_name, boxmean=True, boxpoints='all'))

        fig.update_layout(title_text=fig_name,
                          # barmode='group',
                          # xaxis={'categoryorder': 'total descending'}
                          )
        fig.show()
        fig.write_image(output_path / f"{fig_name}.svg")

    # # make the f1 mean plot
    # scores = {"model": [], "f1": []}
    # for model_scores in scores_path.glob("*.json"):
    #     model_name = model_scores.stem
    #     _s = json.loads(open(model_scores).read())
    #     scores["model"].append(model_name)
    #     scores["f1"].append(_s["f1"]["mean"])
    # _plot([scores["model"]], [scores["f1"]], ["f1"], figure_name)

    # make the f1 classwise plot - a box for each model
    if plot:
        scores = {}
        model_names = []
        for model_scores in scores_path.glob("*.json"):
            model_name = model_scores.stem
            model_names.append(model_name)
            _s = json.loads(open(model_scores).read())
            scores[model_name] = [_s["f1"][warning] for warning in major_warning]

        _plot(
            [scores[model] for model in model_names],
            model_names,
            f"{figure_name}-models-box"
        )

        scores = {}
        model_names = []
        for model_scores in scores_path.glob("*.json"):
            model_name = model_scores.stem
            model_names.append(model_name)
            _s = json.loads(open(model_scores).read())

            for warning in major_warning:
                scores.setdefault(warning, []).append(_s["f1"][warning])

        _plot(
            [scores[warning] for warning in major_warning],
            list(major_warning.keys()),
            f"{figure_name}-warnings-box"
        )

    # print table
    model_names = []
    scores = {}
    print("model & mean & ", " & ".join(list(major_warning.keys())))
    for model_scores in scores_path.glob("*.json"):
        model_name = model_scores.stem
        model_names.append(model_name)
        _s = json.loads(open(model_scores).read())
        line = f"{model_name} \t& {_s['f1']['mean']} "
        scores.setdefault("mean", []).append(_s['f1']['mean'])
        for warning in major_warning:
            scores.setdefault(warning, []).append(_s['f1'][warning])
            line += f"\t& {_s['f1'][warning]}"
        line += " \\\\"
        print(line)
    # mean + median
    print(f"mean \t& {round(mean(scores['mean']),2)} \t& " + " \t& ".join([str(round(mean(scores[warning]),2)) for warning in major_warning]))
    print(f"median \t& {round(median(scores['mean']),2)} \t& " + " \t& ".join([str(round(median(scores[warning]),2)) for warning in major_warning]))

    # IRQ
    print(f"IRQ \t& {round(stats.iqr(scores['mean']),2)} \t& " + " \t& ".join([str(round(stats.iqr(scores[warning]),2)) for warning in major_warning]))

