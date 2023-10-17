# eacl24-trigger-warnings

This repository contains the data and code for **Passage-level Trigger Warning Assignment**. 

For all commands, you can read the docstrings or add `--help` for usage explanations. 

## Get the Data

Note: This is obfuscated for review. Please consider the data samples in `/samples/`

~~1. Download the data from zenodo: .~~

## Setup

1. This project is build with poetry, see here for installation: `https://python-poetry.org/docs/`
2. Run `poetry install`
3. Run `poetry run python3 main.py --help` for usage instructions of the individual methods. 

## Reproduce dataset creation

Build the `dataset.jsonl` file from the annotation result files from label studio. Contact the authors to get the source files. 
   ```
   poetry run python3 main.py data parse_ls
   ```

Compile the training and test data splits from the `dataset.jsonl`:
   ```
   poetry run python3 main.py data prepare_experiments
   ```

## Experiments

Re-run the experiments:
   ```
   poetry run python3 main.py models single_label_experiments
   poetry run python3 main.py models multi_label_experiment
   poetry run python3 main.py models fewshot_calibration
   poetry run python3 main.py models fewshot_experiment
   ```

## Evaluation

Re-run the evaluation:
   ```
   poetry run python3 main.py eval annotation_statistics
   poetry run python3 main.py eval evaluate
   poetry run python3 main.py eval plot_scores
   ```

