import json

import click
from statistics import mean
from pathlib import Path

from eacl24_trigger_warnings.dataset.dataset import compile_dataset, create_experiment_datasets, collect_icl_examples
from eacl24_trigger_warnings.evaluation.evaluation import anno_stats, evaluate_classification_results, \
    scores_presentation, keyphrase_stats, count_test_data
from eacl24_trigger_warnings.classification.bert.singlelabel import run_single_label_experiment
from eacl24_trigger_warnings.classification.bert.multilabel import run_multi_label_experiment
from eacl24_trigger_warnings.classification.gpt.gpt_experiments import Prompt, run_gpt_experiment
from eacl24_trigger_warnings.classification.bert.collect_results import collect_bin, collect_1cl, collect_1cs


@click.group()
def data():
    pass


@click.group()
def eval():
    pass


@click.group()
def models():
    pass


@click.option('--ls-primary', type=click.Path(exists=True), help="Path to a label studio output file")
@click.option('--ls-additional', type=click.Path(exists=True),
              help="List of Paths to label studio output files where the source_id field corresponds to the id field "
                   "of ls-primary", multiple=True)
@click.option('--output-path', type=click.Path(), help="Where to write the output to.")
@click.option('--known-ids', type=str, help="Ids of known annotators", multiple=True, default=['63', '29'])
@data.command()
def parse_ls(ls_primary: str, ls_additional: list, output_path: str, known_ids: tuple):
    """ Parse the Annotations as downloaded from Label Studio and compile the dataset file

    The produced data file has the following keys:
        "warning": the warning labels this annotation is for,
        "labels": a list of all the votes of the (3) annotations (1 true, 0 false),
        "text": The text as a string,
        "work_id": which fic this text is taken from,
        "set_id": 0 or 1 - which keyword split this is taken from
    """
    dataset = compile_dataset(json.loads(open(ls_primary).read()),
                              [json.loads(open(ls).read()) for ls in ls_additional],
                              known_ids)
    open(output_path, 'w').writelines([f"{json.dumps(data)}\n" for data in dataset])


@click.option('--dataset-path', type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help="The complied dataset.jsonl as produced by `parse_ls`")
@click.option('--output-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Where to write the output to.")
@data.command()
def prepare_experiments(dataset_path: str, output_path: str):
    """ Process the dataset file to create all experimental sets:
    1. In-distribution split (training set_id = 0, test set_id = 1)
        - Training :
            (A) strict one-class setting:
            - 1-v-all classification with positive and w/o negative examples
            (B) lenient one-class setting:
            - 1-v-all classification with positive and with negative examples - the all-class contains negative examples from the 1-class and from other major trigger types
            (C) expanded binary setting:
            - individual classifiers, using in-class positives, in-class negatives, and negatives from other major trigger types.
            Possible Expansions:
                # (D) explicit one-class setting (reminiscent of two-stage classification - determine triggering and topic differently):
                # - 1-v-all classification; negative examples are their own class
                # (E) strict binary setting:
                # - individual classifiers, using in-class positives and negatives (when possible)
        - Test:
            - All annotated, positive examples
            - All annotated, negative examples
            - Random examples without keywords

    2. Out-of-distribution split (same sets as above, but data are sampled uniformly from both set_ids)
    """
    create_experiment_datasets(Path(dataset_path), Path(output_path))


@click.option('--dataset-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="root of the dataset path")
@eval.command()
def annotation_statistics(dataset_path: str):
    """ Print the annotation statistics table"""
    classes = {json.loads(line)["example_id"]: json.loads(line)["warning"]
               for line in open(Path(dataset_path) / "dataset.jsonl")}
    anno_stats(Path(dataset_path) / "dataset.jsonl")
    keyphrase_stats(Path("../resources/anno-keywords.json"))
    count_test_data(classes,
                    Path(dataset_path) / "datasets" / "id" / "test-binary",
                    Path(dataset_path) / "datasets" / "ood" / "test-binary",
                    Path(dataset_path) / "datasets" / "id" / "binary-extended",
                    Path(dataset_path) / "datasets" / "ood" / "binary-extended"
                    )

    lead_by_class = {}
    for ls in Path(dataset_path).glob("ls-*.json"):
        ls = json.loads(open(ls).read())
        for _ in ls:
            _id = int(_.get("source_id", _["id"]))
            if _["lead_time"] > 300:
                continue
            lead_by_class.setdefault(classes.get(_id, "dup"), {}).setdefault(_id, []).append(_["lead_time"])

    print("warning \t& mean \\")

    lead_by_class.pop("dup")
    for warning, examples in lead_by_class.items():
        print(f"{warning} \t& "
              # f"{mean([ex[0] for idx, ex in examples.items()])} \t& "
              # f"{mean([ex[1] for idx, ex in examples.items()])} \t& "
              # f"{mean([ex[2] for idx, ex in examples.items()])} \t& "
              f"{round(mean([mean(ex) for idx, ex in examples.items()]), 0)}")
    total_mean = mean([mean([mean(ex) for idx, ex in examples.items()]) for _, examples in lead_by_class.items()])
    print(f"total {round(total_mean, 0)}")
    # print(lead_by_class["abduction"])


@click.option('--dataset-path', type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help="The complied dataset as produced by `parse_ls`")
@click.option('--output-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="The complied dataset as produced by `parse_ls`")
@data.command()
def collect_demonstrations(dataset_path: str, output_path: str):
    """ Print the annotation statistics table """
    collect_icl_examples(Path(dataset_path), Path(output_path))


@click.option('--dataset-path', type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help="Path to the dataset.jsonl file")
@click.option('--truth-path', type=click.Path(exists=True, dir_okay=False, file_okay=True),
              help="Path to the test.jsonl file with the multi-label style truth values for all examples")
@click.option('--bert-predictions-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Path to the dir with all the prediction.jsonl files")
@eval.command()
def evaluate(dataset_path: str, truth_path: str, bert_predictions_path: str):
    """ Evaluate the results """
    evaluate_classification_results(Path(dataset_path), Path(truth_path), Path(bert_predictions_path))


@click.option('--scores-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Path to the directory with the scores.json")
@click.option('--output-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="")
@click.option('--figure-name', type=str)
@eval.command()
def plot_scores(scores_path: str, output_path: str, figure_name: str):
    """ Evaluate the results """
    scores_presentation(Path(scores_path), Path(output_path), figure_name)


@click.option("-c", "--checkpoint", type=click.STRING, default="roberta-base",
              help="base checkpoint for model and tokenized",)
@click.option("--tokenizer-checkpoint", type=click.STRING, default="roberta-base",
              help="base checkpoint for model and tokenized",)
@click.option("--training", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to the training dataset jsonl file",)
@click.option("--validation", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to the validation dataset jsonl file",)
@click.option("--test", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to the test dataset jsonl file")
@click.option("-s", "--savepoint", type=click.Path(), default="./models", help="Path to save the model in.",)
@click.option("--epochs", type=click.INT, default=5, help="Path to save the model in.")
@click.option("--batches", type=click.INT, default=1, help="Batch size")
@click.option("--lr", type=click.FLOAT, default=0.00001, help="Initial learning rate")
@click.option("--use-cpu", is_flag=True, show_default=True, default=False,
              help="If set, run the model on the CPU (for local testing w/o GPU)",)
@click.option("-n", "--name", type=click.STRING, default="develop", help="base name of the model (for wandb)")
@models.command()
def single_label_experiments(checkpoint: str, tokenizer_checkpoint: str, training: str, validation: str, test: str,
                             savepoint: str, epochs: int, batches: int, lr: float, use_cpu: bool, name: str):
    """
    Run the (neural baseline) experiments.
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    run_single_label_experiment(checkpoint, tokenizer_checkpoint, training, validation, test, savepoint,
                                epochs, batches, lr, use_cpu, name)
    collect_bin(Path(savepoint), Path(savepoint), name)


@click.option("-c", "--checkpoint", type=click.STRING, default="roberta-base",
              help="base checkpoint for model and tokenized",)
@click.option("--tokenizer-checkpoint", type=click.STRING, default="roberta-base",
              help="base checkpoint for model and tokenized")
@click.option("--training", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to the training dataset jsonl file",)
@click.option("--validation", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to the validation dataset jsonl file")
@click.option("--test", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="Path to the test dataset jsonl file")
@click.option("-s", "--savepoint", type=click.Path(), default="./models", help="Path to save the model in.")
@click.option("--epochs", type=click.INT, default=5, help="Path to save the model in.")
@click.option("--batches", type=click.INT, default=1, help="Batch size")
@click.option("--lr", type=click.FLOAT, default=0.00001, help="Initial learning rate")
@click.option("--use-cpu", is_flag=True, show_default=True, default=False,
              help="If set, run the model on the CPU (for local testing w/o GPU)",)
@click.option("-n", "--name", type=click.STRING, default="develop", help="base name of the model (for wandb)")
@click.option("--setting", type=str, help="`oneclass-strict` or `oneclass-lenient`")
@models.command()
def multi_label_experiment(checkpoint: str, tokenizer_checkpoint: str, training: str, validation: str, test: str,
                           savepoint: str, epochs: int, batches: int, lr: float, use_cpu: bool, name: str,
                           setting: str):
    """
    Run the (neural baseline) experiments.
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    run_multi_label_experiment(checkpoint, tokenizer_checkpoint, training, validation, test,
                               savepoint, epochs, batches, lr, use_cpu, name)
    if setting == "oneclass-strict":
        collect_1cs(Path(savepoint), Path(savepoint), name)
    elif setting == "oneclass-lenient":
        collect_1cl(Path(savepoint), Path(savepoint), name)


@click.option("--test", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default="/Users/matti/Documents/data/trigger-sentences/datasets/id/test/test.jsonl",
              help="Path to the test dataset jsonl file")
@click.option("--savepoint", type=click.Path(), default="./gpt-results", help="Path to save the results in.")
@click.option("--keyfile", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              default="../../../resources/gpt-secret-sentence-triggers.txt",
              help="File with the openai API key")
@click.option("--model", type=str, default="gpt3", help="gpt3, gpt4, bison")
@models.command()
def fewshot_experiment(test: str, savepoint: str, keyfile: str, model: str):
    """
    Run the gpt experiments.
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)
    prompts = [
        Prompt(),  # Instruction only
        Prompt(definition=True, extended_instruction=True),  # Definition
        Prompt(persona=True, extended_instruction=True),  # Persona + Definition
        Prompt(persona=True, definition=True, extended_instruction=True),  # Persona + Warning Definition  + Extended Instruction
        Prompt(demonstration=True, num_unanimous=2, num_non_unanimous=0),  # only ICL
        Prompt(persona=True, definition=True, extended_instruction=True, demonstration=True, num_unanimous=2, num_non_unanimous=0),  # complete set
    ]
    run_gpt_experiment(Path(test), Path(savepoint), open(keyfile).read(), prompts, model)


@click.option("--examples", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Path to the calibration test jsonl files")
@click.option("--savepoint", type=click.Path(),
              help="Path to save the results in.")
@click.option("--keyfile", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="File with the openai API key")
@models.command()
def fewshot_calibration(examples: str, savepoint: str, keyfile: str):
    """
    "Calibrate" the prompts - run and evaluate on the training data

    - Instruction only
    - Instruction + Persona
    - Instruction + Definition
    - Instruction + Persona + Definition
    - Instruction + ICL (different versions)
    - (Instruction + Explanations)
    ---
    - Instruction + Persona + Definitions + ICL
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)

    prompts = [
        Prompt(),  # Instruction only
        Prompt(persona=True),  # Persona
        Prompt(definition=True),  # Definition
        Prompt(persona=True, definition=True),  # Persona + Definition
        Prompt(persona=True, extended_instruction=True),   # Persona + Extended Instruction
        Prompt(definition=True, extended_instruction=True),  # Definition + Extended Instruction
        Prompt(persona=True, definition=True, extended_instruction=True),   # Persona + Warning Definition  + Extended Instruction
        Prompt(persona=False, definition=False, demonstration=True, num_unanimous=2, num_non_unanimous=0),  # ICL - unanimous
        Prompt(persona=False, definition=False, demonstration=True, num_unanimous=0, num_non_unanimous=2),  # ICL - non-unanimous
        Prompt(persona=False, definition=False, demonstration=True, num_unanimous=1, num_non_unanimous=1),  # ICL - half n half
        Prompt(persona=False, definition=False, demonstration=True, num_unanimous=2, num_non_unanimous=2),  # ICL - half n half with more examples
    ]
    run_gpt_experiment(Path(examples), Path(savepoint), open(keyfile).read(), prompts)


cli = click.CommandCollection(sources=[data, eval, models])

if __name__ == "__main__":
    cli()
