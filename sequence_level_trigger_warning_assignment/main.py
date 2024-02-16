import json

import click
from statistics import mean
from pathlib import Path
import wasabi
from shutil import copyfile
from gc import collect
from torch import cuda

from sequence_level_trigger_warning_assignment.dataset.dataset import compile_dataset, create_experiment_datasets, collect_icl_examples, create_few_shot_ablation_datasets
from sequence_level_trigger_warning_assignment.evaluation.evaluation import anno_stats, score_classification_results, \
    scores_presentation, segment_stats, count_test_data
from sequence_level_trigger_warning_assignment.classification.bert import run_bert_experiment
# from sequence_level_trigger_warning_assignment.classification.bert.multilabel import run_multi_label_experiment
from sequence_level_trigger_warning_assignment.classification.gpt_api import run_gpt_experiment
from sequence_level_trigger_warning_assignment.classification.generative import run_generative_experiment, run_chat_experiment
from sequence_level_trigger_warning_assignment.classification.collect_results import collect_bin, collect_1cl,  collect_mc
from sequence_level_trigger_warning_assignment.utility import major_warning


msg = wasabi.Printer()


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
@click.option('--output-path', type=click.Path(dir_okay=True, file_okay=False),
              help="Where to write the output to.")
@click.option('--prompt-path', type=click.Path(dir_okay=True, file_okay=False),
              help="Directory with the prompt parts.")
@click.option('--seed', type=int,
              help="Number to initialize the RNG with.")
@data.command()
def prepare_experiments(dataset_path: str, output_path: str, prompt_path: str, seed: int):
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
    output_path = Path(output_path) 
    output_path.mkdir(exist_ok=True)
    prompt_path = Path(prompt_path)
    prompt_path.mkdir(exist_ok=True)

    create_experiment_datasets(Path(dataset_path), output_path, prompt_path, inference='majority', rng_seed=seed)
    create_experiment_datasets(Path(dataset_path), output_path, prompt_path, inference='minority', rng_seed=seed)
    create_few_shot_ablation_datasets(Path(dataset_path), output_path / "prompt-ablation", prompt_path, inference='majority')


@click.option('--dataset-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="root of the dataset path")
@click.option('--keywords-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="dir with the keyword files")
@click.option('--annotations-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="dir with the annotation files ls-*.json")
@eval.command()
def annotation_statistics(dataset_path: str, keywords_path: str, annotations_path: str):
    """ Print the annotation statistics table"""

    classes = {json.loads(line)["example_id"]: json.loads(line)["warning"]
               for line in open(Path(dataset_path) / "dataset.jsonl")}
    
    msg.info("Annotation Statistics")
    anno_stats(classes, Path(dataset_path) / "dataset.jsonl", annotations_path=Path(annotations_path))

    msg.info("Segment Statistics")
    segment_stats(Path(dataset_path) / "dataset.jsonl", keyphrase_file=Path(keywords_path) / "anno-keywords.json")

    msg.info("Dataset Statistics")    
    count_test_data(classes,
                    Path(dataset_path) / "datasets" / "id-minority" / "test-binary",
                    Path(dataset_path) / "datasets" / "ood-minority" / "test-binary",
                    Path(dataset_path) / "datasets" / "id-minority" / "binary-extended",
                    Path(dataset_path) / "datasets" / "ood-minority" / "binary-extended"
                    )
    count_test_data(classes,
                    Path(dataset_path) / "datasets" / "id-majority" / "test-binary",
                    Path(dataset_path) / "datasets" / "ood-majority" / "test-binary",
                    Path(dataset_path) / "datasets" / "id-majority" / "binary-extended",
                    Path(dataset_path) / "datasets" / "ood-majority" / "binary-extended"
                    )

    # TODO (qual/quant) evaluation of disagreement


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
@click.option('--bert-predictions-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Path to the dir with all the prediction.jsonl files")
@eval.command()
def evaluate_classification(dataset_path: str, bert_predictions_path: str):
    """ Evaluate the results """
    score_classification_results(Path(dataset_path), Path(bert_predictions_path))


@click.option('--scores-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="Path to the directory with the scores.json")
@click.option('--output-path', type=click.Path(exists=True, dir_okay=True, file_okay=False),
              help="")
@click.option('--figure-name', type=str, default="scores")
@eval.command()
def plot_scores(scores_path: str, output_path: str, figure_name: str):
    """ Evaluate the results """
    scores_presentation(Path(scores_path), Path(output_path), figure_name)


@click.option("-c", "--checkpoint", type=click.STRING, default="roberta-base",
              help="base checkpoint for model and tokenized",)
@click.option("--tokenizer-checkpoint", type=click.STRING, default="roberta-base",
              help="base checkpoint for model and tokenized",)
@click.option("--training", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Path to the directory with the training dataset jsonl files",)
@click.option("--validation", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Path to the directory with the validation dataset jsonl files",)
@click.option("--test", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Path to the directory with the test dataset jsonl files")
@click.option("-s", "--savepoint", type=click.Path(), default="./models", help="Path to save the model in.",)
@click.option("--epochs", type=click.INT, default=5, help="Path to save the model in.")
@click.option("--batches", type=click.INT, default=1, help="Batch size")
@click.option("--lr", type=click.FLOAT, default=0.00001, help="Initial learning rate")
@click.option("-n", "--name", type=click.STRING, default="develop", help="base name of the model (for wandb)")
@click.option("--mode", type=str, help="`multilabel` or `multilabel-extended` or `binary` or `multiclass`")
@click.option("--cache-dir", type=click.Path(), default=None, help="Path to save the results in.")
@click.option("--no-wandb", is_flag=True)
@models.command()
def finetune_experiments(checkpoint: str, tokenizer_checkpoint: str, training: str, validation: str, test: str, savepoint: str, epochs: int, batches: int, lr: float, name: str, mode: str, cache_dir: str|None, no_wandb: bool):
    """
    Run the (neural baseline) experiments.
    """
    savepoint = Path(savepoint)
    savepoint.mkdir(parents=True, exist_ok=True)

    msg.info(f"Run BERT experiments {name} - {mode} - {lr}")

    if mode == "multilabel":
        run_bert_experiment(checkpoint, tokenizer_checkpoint, 
                            f"{training}/train.jsonl", 
                            f"{validation}/test.jsonl", 
                            f"{test}/test.jsonl",
                            savepoint.absolute().as_posix(), epochs, batches, lr, name, num_labels=8, 
                            mode='multilabel', cache_dir=cache_dir, no_wandb=no_wandb)
        copyfile(savepoint / "test-predictions.jsonl",
                savepoint / f"{name}.jsonl")
        
    elif mode == "multilabel-extended":
        # this has a violence and a discrimination split
        for warning in {"violence", "discrimination"}:
            (savepoint / warning).mkdir(exist_ok=True)
            run_bert_experiment(checkpoint, tokenizer_checkpoint, 
                                f"{training}/train-{warning}.jsonl", 
                                f"{validation}/test.jsonl", 
                                f"{test}/test.jsonl",
                                (savepoint / warning).absolute().as_posix(), epochs, batches, lr, 
                                f"{name}-{warning}", num_labels=8, mode='multilabel', cache_dir=cache_dir, no_wandb=no_wandb)
        
        collect_1cl(violence_predictions=savepoint / "violence" / "test-predictions.jsonl",
                    discrimination_predictions=savepoint / "discrimination" / "test-predictions.jsonl", 
                    output_file=savepoint / f"{name}.jsonl")

    elif mode == "binary":
        for warning in major_warning.keys():
            run_bert_experiment(checkpoint, tokenizer_checkpoint, 
                                f"{training}/train-{warning}.jsonl", 
                                f"{validation}/test-{warning}.jsonl", 
                                f"{test}/test-{warning}.jsonl", 
                                (savepoint / warning).absolute().as_posix(),
                                epochs, batches, lr, f"{name}-{warning}", 
                                num_labels=2, mode='binary', cache_dir=cache_dir, no_wandb=no_wandb)
        collect_bin(savepoint, output_file=savepoint / f"{name}.jsonl")

    elif mode == "multiclass":
        run_bert_experiment(checkpoint, tokenizer_checkpoint, 
                            f"{training}/train.jsonl", 
                            f"{validation}/test.jsonl", 
                            f"{test}/test.jsonl",
                            savepoint.absolute().as_posix(), epochs, batches, lr, name, num_labels=9, 
                            mode='multiclass', cache_dir=cache_dir, no_wandb=no_wandb)
        copyfile(savepoint / "test-predictions.jsonl", savepoint / f"{name}.jsonl")


@click.option("--checkpoint", type=str, default="gpt3", help="`gpt3-5-turbo`, `gpt4-turbo`, `mistralai/Mistral-7B-Instruct-v0.2`, `meta-llama/Llama-2-13b-chat-hf`, `meta-llama/Llama-2-7b-chat-hf`,`mistralai/Mixtral-8x7B-Instruct-v0.1`, `tiiuae/falcon-7b-instruct`")
@click.option("--test", type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help="Path to the directory with the test dataset jsonl files (where `text` is the prompt)")
@click.option("--savepoint", type=click.Path(), default="./gpt-results", help="Path to save the results in.")
@click.option("--cache-dir", type=click.Path(), help="Path to save the results in.")
@click.option("--keyfile", type=click.Path(exists=True, file_okay=True, dir_okay=False),
              help="File with the openai API key")
@click.option("--batches", type=click.INT, default=12, help="Batch size")
@click.option("--decoding-strategy", type=str, default="default", help="`default` or `chat`")
@models.command()
def fewshot_experiment(checkpoint: str, test: str, savepoint: str, cache_dir: str, keyfile: str, batches: int, decoding_strategy: str):
    """
    Run the gpt experiments. Uses the ChatGPT API or a local huggingface implementation, based on the selected checkpoints. 
    """
    Path(savepoint).mkdir(parents=True, exist_ok=True)

    for prompt_file in Path(test).glob("*.jsonl"):
        name = prompt_file.stem.removeprefix("test-")
        msg.info(f"Run Generative experiments {name}")
        if checkpoint in {"gpt-3.5-turbo-0125", "gpt-4-0125-preview"}:
            run_gpt_experiment(prompt_file, Path(savepoint), open(keyfile).read(), 
                               checkpoint, name)
            exit()
        elif decoding_strategy == 'default':
            run_generative_experiment(checkpoint, 
                                      prompt_file.absolute().as_posix(), 
                                      savepoint, cache_dir, batches, name)
        elif decoding_strategy == 'chat':
            run_chat_experiment(checkpoint, 
                                prompt_file.absolute().as_posix(), 
                                savepoint, cache_dir, batches, name)


cli = click.CommandCollection(sources=[data, eval, models])


if __name__ == "__main__":
    cli()
