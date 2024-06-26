# Predicting Code Comprehension: A Novel Approach to Align Human Gaze with Code Using Deep Neural Networks

![cc-eye-movement-hq](https://github.com/Taremeh/predicting-code-comprehension-eye-tracking/assets/1512652/e9701966-2ed1-4a02-833e-13d89f4a367b)


This repository contains the code snippets and comprehension tasks of the code comprehension experiment, the anonymized raw and processed eye-tracking data, the nested cross-validation script containing the neural network, helper scripts, and scripts to run the baselines mentioned in the paper. Detailed information can be found in the README files in each of the repository's subdirectories:

- `code_snippets_and_comprehension_tasks`: All code snippets and corresponding comprehension tasks as single Python files, task answer options and mapping files.
- `processed_data`: Processed data used to train the neural networks*
- `raw_fixation_reports`: Raw eye-tracking data.
- `scripts`: Scripts to run our model, the baseline models, and various helper scripts.

You can also find the following files in this root directory:

- `hyperparameter_search_results.txt`: Containing the results of the nested cross-validation hyperparameter search.
- `raw_em_data.rar`: Raw high-frequency eye movement data (not required to run our model)*

## Quick-Start: Run our model

`python scripts/evaluate_cross_validation.py --problem-setting {accuracy, subjective_difficulty} --split {subject, code-snippet} --mode {bimodal, fixations, code}`

For more details, refer to `scripts/README.md`.

## *Full Data Download
Due to large file sizes you can download the entire package including the `processed_data` and `raw_em_data.rar` via [Zenodo](https://zenodo.org/doi/10.5281/zenodo.11123100).
