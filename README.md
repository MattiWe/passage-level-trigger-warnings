# acl24-trigger-warnings

This repository contains the data and code for **If there's a Trigger Warning, then where's the Trigger? Investigating Trigger Warnings at the Passage Level**. 

For all commands, you can read the docstrings or add `--help` for usage explanations. 

## Get the Data

Note: This is obfuscated for review. Please consider the dataset or the excerpt in `/samples/`
Data will be published via Zenodo for the final release.


## Setup

1. Run `pip install .`
2. Run `python3 main.py --help` for usage instructions of the individual methods. 

## Dataset

Unzip the dataset `dataset.jsonl` in `resources`. 

To reconstruct this file from the annotation result files from label studio, run:

   ```
   poetry run python3 main.py data parse_ls
   ```

Contact the authors to get the source files. 

## Experiments

`sequence_level_trigger_warning_assignment/experiment-runners` contains scripts to run every part of the experiments. In particular:

- `run-compile-datasets.sh` creates the cross-validation folds. 
- `run-evaluate-dataset.sh` created the dataset statistics.
- `run-experiments-*` runs the respective experiments. Those create `score-` and `prediction-` files for each run under `resources/models-*`. Copy these prediction files to a shared directory for evaluation (default: `resources/classification-results/predictions`). 
- `run-evaluate-classification.sh` scores the predictions in the evaluation directory in a binary fashion. 

Note: For the GPT experiments, you need to add your OpenAI API key in a file `resources/prompt/gpt-secret-sentence-triggers.txt`

## Evaluation

`sequence_level_trigger_warning_assignment/notebooks` contains jupyter notebooks used to evaluate and plot the classification results. Run the cells in order. 