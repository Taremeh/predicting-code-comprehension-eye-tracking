# Scripts

This directory contains scripts to run our model, the baseline models, and various helper scripts. 

## Run our model

`python evaluate_cross_validation.py --problem-setting {accuracy, subjective_difficulty} --split {subject, code-snippet} --mode {bimodal, fixations, code}`

Parameter Explanation:

- **problem setting**: `accuracy` (predict code comprehension), `subjective_difficulty` (predict perceived difficulty to understand code)
- **split**: `subject` (split validation by subjects), `code-snippet` (split validation by code snippets)
- **mode**: `bimodal` (use fixation data and code embeddings for training & predictions), `fixations` (use only fixation data for training & predictions), `code` (use only code embeddings for training & predictions)

## Run baseline models
`python {al_madi,fritz,hanaka}.py --problem-setting {accuracy, subjective_difficulty} --split {subject, code-snippet}`

Parameters to run the baselines are analog to running our model (with the exception of providing a mode)

*Please note that depending on your machine, CUDA version etc. version incompatibilities might occur. Please adjust the gpu_selection method to your needs or comment out the respective functions. We provide a `./utils/requirements.txt` file (using Python 3.10 and Anaconda), however, in our experience incompatibilities with the underlying CUDA platform or other dependencies have still occurred when running the scripts on different machines.*