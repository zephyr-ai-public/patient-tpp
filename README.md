# patient-tpp
Zephyr AI extensions to the EasyTPP framework, December 2025

This repository contains resources for building Temporal Point Process models from electronic health records (EHR), as described in [preprint](https://khaaan.com).

## Installation

If you have access to [poetry](https://python-poetry.org/) on your system, clone this repo and run

    poetry install
    
from the repository root. If `poetry` is not available, use the supplied requirements file with an installation of Python 3.12:

    pip install -r requirements.txt

## Contents

- `easy_tpp_zai` contains Zephyr's extensions to [EasyTPP](https://github.com/ant-research/EasyTemporalPointProcess), most importantly the `AttNHP` variant.
- A synthetic dataset is provided under `test/data`, comprising ~310k patients in a 75/20/5 train/test/validation split.
    - Indicative features (diagnoses) and numeric features (lab results) are provided in separate streams. Patient-tpp consumes both simultaneously.  
    - Time is in units of _quarters_. 
    - Records span at most 76 quarters (19 years). t = 0 corresponds to 2005-Q4.
    - Mappings for integer event types and diagnostic codes are provided in the resource file `test/data/enc2thingnamerval.pkl`.
    
## Usage
A model can be invoked for training or generation using notebook commands or the `driver.py` script and an experiment id from a config file,
typically under `configs`. Point the config at some json sequence data and issue commands like

    python driver.py --config configs/config.yaml --experiment_id AttNHP_zai_train   # Train an attention-based NHP model
    python driver.py --config configs/config.yaml --experiment_id AttNHP_zai_gen     # Generate predictions from the configured test set

Note to run in generate mode (i.e., to make predictions) you will need to supply a specific `pretrained_model_dir` under the `model_config` node.
An entry might look like

    `model_config:
        pretrained_model_dir: ./checkpoints/27175_139998201468736_250130-181341/models/saved_model`
        
Specifying a `pretrained_model_dir` for a training experiment will resume training from that checkpoint, provided the model configurations are compatible.

See `demo.ipynb` for in-notebook training and generation.
