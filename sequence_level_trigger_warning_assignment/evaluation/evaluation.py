import krippendorff
from typing import List
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, cohen_kappa_score
import numpy as np
from pathlib import Path
import json
import shutil
from statistics import mode, mean, median
from sequence_level_trigger_warning_assignment.utility import major_warning, infer_label, multilabel_to_multiclass
from scipy import stats
from typing import Dict

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


def anno_stats(classes: Dict, dataset_file: Path, annotations_path: Path):
    """
    kappa + time + annotations after infrence

    :param dataset_file:
    :return:
    """
    dataset = [json.loads(d) for d in open(dataset_file).readlines()]

    counts_maj = {}
    counts_min = {}
    _raters = {}
    for date in dataset:
        counts_maj.setdefault(date["warning"], {}).setdefault(date["set_id"], []).append(infer_label(date, method="majority")["labels"])
        counts_maj.setdefault("total", {}).setdefault(date["set_id"], []).append(infer_label(date, method="majority")["labels"])
        counts_min.setdefault(date["warning"], {}).setdefault(date["set_id"], []).append(infer_label(date, method="minority")["labels"])
        counts_min.setdefault("total", {}).setdefault(date["set_id"], []).append(infer_label(date, method="minority")["labels"])
        _raters.setdefault(date["warning"], []).append(date["labels"])
        _raters.setdefault("total", []).append(date["labels"])

    # annotation time 
    lead_by_class = {}
    for ls in annotations_path.glob("ls-*.json"):
        ls = json.loads(open(ls).read())
        for _ in ls:
            _id = int(_.get("source_id", _["id"]))
            if _["lead_time"] > 300:
                continue
            lead_by_class.setdefault(classes.get(_id, "dup"), []).append(_["lead_time"])
    lead_by_class.pop("dup")
    
    lead_by_class["total"] = [t for times in lead_by_class.values() for t in times]

    # unanimouns negative - non-unanimous - unanimouns positive
    # majority-minority delta in percentages of total for each warning

    # rater characteristics 
    # (sensitity of the rater) for each warning and rater: many positives or negatives? 
    # (deviating rater) is one rated systematically different (i.e. has less overlap, pairwise cohens kappa?)

    # print(lead_by_class)
    print(f"warning &"
          f"maj-pos & maj-neg & min-pos (positive ratio) & min-neg \t&" 
          f"time & alpha \t&" 
          f"\\\\")
    
    w = list(major_warning.keys()) + ["total"]
    for warning in w:
        r1 = [r[0] for r in _raters[warning]]
        r2 = [r[1] for r in _raters[warning]]
        r3 = [r[2] for r in _raters[warning]]
        total = len(counts_min[warning][0]) + len(counts_min[warning][1])
        maj_pos = sum(counts_maj[warning][0]) + sum(counts_maj[warning][1])
        maj_neg = total - maj_pos
        min_pos = sum(counts_min[warning][0]) + sum(counts_min[warning][1])
        min_neg = total - min_pos
        print(f"{warning.title()} \t& "
              f"{maj_pos} \t& {maj_neg} \t& "
              f"{min_pos} ({int(round(min_pos / total,2)*100)}) \t& {min_neg} \t& "
              f"{round(mean(lead_by_class[warning]), 0)} \t&"
              f"{round(krippendorff.alpha(reliability_data=[r1, r2, r3], level_of_measurement='nominal'), 2)}"
              f"\t\\\\"
              )
    
    print(f"warning &"
          f"r1-2 kappa & r1-3 kappa & r2-3 kappa \t&" 
          f"r1-2 overlap & r1-3 overlap & r2-3 overlap \t&" 
          f"r1 mean kappa & r2 mean kappa & r3 mean kappa \t&" 
          f"r1 pos rate & r2 pos rate & r3 pos rate \t" 
          f"\\\\")
    
    for warning in w:
        r1 = [r[0] for r in _raters[warning]]
        r2 = [r[1] for r in _raters[warning]]
        r3 = [r[2] for r in _raters[warning]]
        unanimous_pos = len([1 for _1, _2, _3 in zip(r1, r2, r3) if _1 == 1 and _2 == 1 and _3 == 1])
        unanimous_neg = len([1 for _1, _2, _3 in zip(r1, r2, r3) if _1 == 0 and _2 == 0 and _3 == 0])
        r12_kappa = round(cohen_kappa_score(r1, r2), 2)
        r13_kappa = round(cohen_kappa_score(r1, r3), 2)
        r23_kappa = round(cohen_kappa_score(r2, r3), 2)
        r12_overlap = round(len([1 for _1, _2 in zip(r1, r2) if _1 == _2]) / len(r1), 2)
        r13_overlap = round(len([1 for _1, _3 in zip(r1, r3) if _1 == _3]) / len(r1), 2)
        r23_overlap = round(len([1 for _2, _3 in zip(r2, r3) if _2 == _3]) / len(r1), 2)


        print(f"{warning.title()} \t& "
              f"{r12_kappa} & {r13_kappa} & {r23_kappa} \t& "
              f"{r12_overlap} & {r13_overlap} & {r23_overlap} \t& "
              f"{round(mean([r12_kappa, r13_kappa]), 2)} & {round(mean([r12_kappa, r23_kappa]), 2)} & {round(mean([r23_kappa, r13_kappa]), 2)} \t& "
              f"{round(sum(r1)/len(r1), 2)} & {round(sum(r2)/len(r2), 2)} & {round(sum(r3)/len(r3), 2)}" 
              f"\t\\\\"
              )

    print(f"warning &"
          f"maj-0-pos & maj-1-pos \t&" # set 0 positive count + rate
          f"min-0-pos & min-1-pos \t&" 
          f"unan-pos & unan neg & non-unan  \t&" 

          f"\\\\")
    
    for warning in w:
        r1 = [r[0] for r in _raters[warning]]
        r2 = [r[1] for r in _raters[warning]]
        r3 = [r[2] for r in _raters[warning]]
        total = len(r1)
        all_neg = len([1 for _ in zip(r1, r2, r3) if sum(_) == 0])
        pos_1 = len([1 for _ in zip(r1, r2, r3) if sum(_) == 1])
        pos_2 = len([1 for _ in zip(r1, r2, r3) if sum(_) == 2])
        pos_3 = len([1 for _ in zip(r1, r2, r3) if sum(_) == 3])

        total_set0 = len(counts_min[warning][0])
        total_set1 = len(counts_min[warning][1])
        maj_set0_pos = sum(counts_maj[warning][0])
        maj_set1_pos = sum(counts_maj[warning][1])
        min_set0_pos = sum(counts_min[warning][0])
        min_set1_pos = sum(counts_min[warning][1])

        print(f"{warning.title()} \t& "
              f"{maj_set0_pos} ({int(round(maj_set0_pos/total_set0,2)*100)}\,\%) \t& "
              f"{maj_set1_pos} ({int(round(maj_set1_pos/total_set1,2)*100)}\,\%) \t& "
              f"{min_set0_pos} ({int(round(min_set0_pos/total_set0,2)*100)}\,\%) \t& "
              f"{min_set1_pos} ({int(round(min_set1_pos/total_set1,2)*100)}\,\%) \t& "
              f"{all_neg} ({int(round(all_neg/total,2)*100)}\,\%) \t& "
              f"{pos_1} ({int(round(pos_1/total,2)*100)}\,\%) \t& "
              f"{pos_2} ({int(round(pos_2/total,2)*100)}\,\%) \t& "
              f"{pos_3} ({int(round(pos_3/total,2)*100)}\,\%) \t& "
              f"\t\\\\"
              )
    

def segment_stats(dataset_file: Path, keyphrase_file: Path):
    """
    Examples:
    :param keyphrase_file:
    :return:
    """
    keyphrases = json.loads(open(keyphrase_file).read())
    dataset = [json.loads(d) for d in open(dataset_file).readlines()]
    length = {}

    for date in dataset:
        words = len(date["text"].split(" "))
        length.setdefault(date["warning"], []).append(words)
        length.setdefault("total", []).append(words)

    print(f"Warning \t&  Len. \t&  Num. \t& kw source \t& kw cleaned \t& set 1 \t& set 2")
    for warning in major_warning.keys():
        print(f"{warning.title()} \t& "
              f"{round(mean(length[warning]), 0)} \t& {len(length[warning])} \t&"
              f"{len(keyphrases[warning]['phrases'])} \t& {len(keyphrases[warning]['cleaned_up_phrases'])} \t& "
              f"{len(keyphrases[warning]['set_1'])} \t& {len(keyphrases[warning]['set_2'])}"
              f"\\\\"
              )
    print(f"Total \t& "
          f"{round(mean(length['total']), 0)} \t& {len(length['total'])} \t&"
          f"{sum([len(_['phrases']) for _ in keyphrases.values()])} \t& "
          f"{sum([len(_['cleaned_up_phrases']) for _ in keyphrases.values()])} \t& "
          f"{sum([len(_['set_1']) for _ in keyphrases.values()])} \t& "
          f"{sum([len(_['set_2']) for _ in keyphrases.values()])}")


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
        print(f"{warning.title()} \t& {sizes['id_train'][warning]} \t& {sizes['id_test'][warning]} "
              f"\t& {sizes['ood_train'][warning]} \t& {sizes['ood_test'][warning]}")

    print(f"Total \t& {sum([l for w, l in sizes['id_train'].items()])} \t& "
          f"{sum([l for w, l in sizes['id_test'].items()])} \t& "
          f"{sum([l for w, l in sizes['ood_train'].items()])} \t& "
          f"{sum([l for w, l in sizes['ood_test'].items()])}")


####################################################
# Evaluation of the experiments
####################################################

def _score(y_true: List[int], y_pred: List[int]) -> dict:
    """ 
    :param truth: list of lists - one list for each class, each index is the true score (1, 0) for the class
    :param predictions: list of lists - one list for each class, each index is the predicted score (1, 0) for the class
    :return: scores
    """
    results = {"accuracy": round(accuracy_score(y_true, y_pred), 2),
               "f1": round(f1_score(y_true, y_pred, average='micro', zero_division=0), 2), 
               "precision": round(precision_score(y_true, y_pred, average='micro', zero_division=0), 2), 
               "recall": round(recall_score(y_true, y_pred, average='micro', zero_division=0), 2),
               "f1_cls1": f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=1),
               "f1_cls0": f1_score(y_true=y_true, y_pred=y_pred, average="binary", pos_label=0),
               "pr": sum(y_pred) / len(y_pred),
               }

    return results


def _evaluate_bert_results(predictions_file: Path, warning_map: Dict[str, str]) -> dict:
    """ Evaluate the results from the BERT predictions.

    We only evaluate the performance on examples we actually annotated.
    However, for convenience, we made a prediction for each class on all examples (even those that we're not annotated
    for that class)

    Here, we separate the predictions again: For each class, we extract the predictions for only that class from the
    combined output and pass those to _score().
    """
    examples = [json.loads(line) for line in open(predictions_file)]
    if isinstance(examples[0]['labels'], list):
        mode = 'multilabel'
    elif 2 in {_["labels"] for _ in examples}:  # multiclass file
        mode = 'multiclass'
    else:  # multiclass file
        mode = 'binary'
        
    results = {"total": {"y_true": [], "y_pred": []}}
    for example in examples:
        warning = warning_map[example["id"]]
        results.setdefault(warning, {"y_true": [], "y_pred": []})
        if mode == 'multilabel':
            results[warning]["y_true"].append(
                1 if multilabel_to_multiclass(example["labels"], level='minor') == warning else 0
                )
            results[warning]["y_pred"].append(
                1 if multilabel_to_multiclass(example["predictions"], level='minor') == warning else 0
                )
        elif mode == 'multiclass':
            results[warning]["y_true"].append(0 if example["labels"] == 8 else 1)
            results[warning]["y_pred"].append(1 if example["predictions"] == example["labels"] and example["predictions"] != 8 else 0)
        else: # binary file
            results[warning]["y_true"].append(example["labels"])
            results[warning]["y_pred"].append(example["predictions"])
            
    for warning, res in results.items():
        results["total"]["y_true"].extend(res["y_true"])
        results["total"]["y_pred"].extend(res["y_pred"])

    scores = {}
    for warning, values in results.items():
        scores[warning] = _score(
            np.asarray(values["y_true"], dtype=int), 
            np.asarray(values["y_pred"], dtype=int)
            )

    return scores


def score_classification_results(dataset_file: Path, predictions_dir: Path):
    """
    This is to get the scores for the bert predictions.

    :param dataset_file: the original dataset path; this lists the trigger with which each example was annotated.
    :param predictions_dir: directory with the test-predictions.jsonl files. Files must be of the same setting. The prediction files should contain the true `labels` and the `predictions` keys. 
    :return:
    """
    warning_map = {json.loads(example)["example_id"]: json.loads(example)["warning"] for example in open(dataset_file)}
    
    output_path = predictions_dir.parent / 'scores'
    output_path.mkdir(exist_ok=True)
    for prediction_file in predictions_dir.glob("*.jsonl"):
        scores = _evaluate_bert_results(prediction_file, warning_map)
        open(output_path / f"{prediction_file.stem}.json", 'w').write(json.dumps(scores))
        print(prediction_file.stem, scores['total']["accuracy"])


def _aggregate_scores(scores_dir: Path):
    """aggregate the scores 

    :param scores_dir: {
        "model": {"distribution": {"aggregation": {"dsi:" {"warning": {"measure": [values over samples]}}}}}}
    }
    """
    results = {}
    for scores_file in scores_dir.glob("*"):
        scores = json.loads(open(scores_file).read())
        dsi = scores_file.stem[-2:]
        model = scores_file.stem[:-2].removeprefix("acl24-")\
        .removeprefix("fanbert-").removesuffix("-20ep-")\
        .removesuffix("-5lr").removesuffix("-1e").removesuffix("-2e").removesuffix("-5e")
        print(model, dsi)
        aggregation = "minority" if model.endswith("minority") else "majority"
        model = model.removesuffix("-minority").removesuffix("-majority")
        distribution = "id" if model.endswith("id") else "ood"
        model = model.removesuffix("-id").removesuffix("-ood")
        for warning, score_dict in scores.items():
            for measure, value in score_dict.items():
                results.setdefault(model, {}).setdefault(distribution, {}).setdefault(aggregation, {}).setdefault(warning, {}).setdefault(measure, []).append(value)
        return scores


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


def evaluate_prompt_ablation(models_dir: Path):
    result_dirs = [
        models_dir / "mistralai/Mistral-7B-Instruct-v0.2-chat/prompt-ablation",
        models_dir / "mistralai/Mixtral-8x7B-Instruct-v0.1-chat/prompt-ablation",
        models_dir / "meta-llama/Llama-2-13b-chat-hf/prompt-ablation",
        models_dir / "meta-llama/Llama-2-7b-chat-hf/prompt-ablation",
    ]
    prompt_scores = {}
    for result_dir in result_dirs:
        for scores in result_dir.glob("scores*.json"):
            s = json.loads(open(scores).read())["accuracy"]
            model = str(scores.parents[1]).removeprefix("/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/triggering-sentences/acl24/models/")
            prompt = scores.stem.removeprefix("scores-instruction-").removeprefix("scores-")
            print(s, prompt, model)
            prompt_scores.setdefault(prompt, []).append(s)
    
    for p, s in prompt_scores.items():
        print(p, mean(s))


if __name__ == "__main__":
    evaluate_prompt_ablation(Path("/mnt/ceph/storage/data-in-progress/data-research/computational-ethics/trigger-warnings/triggering-sentences/acl24/models"))