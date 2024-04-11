# Safeguards

The experiments are implemented on a computer equipped with an AMD EPYC 64-Core CPU, Tesla T4 GPU, and 100 GB of RAM running Ubuntu 20.04.6. All VizDoom experiments in this paper are conducted using [SampleFactory](https://github.com/alex-petrenko/sample-factory).

## Setup

Please follow the instructions on the SampleFactory page to set up the framework (installing VizDoom, and dependencies of SampleFactory).

## Running Experiments

### ViZDoom

To reproduce the baseline, zero shot, and safeguarded results, use `run_trial.sh`.

Please keep in mind that running all of the trials will take several hours depending on your hardware.

### MineCraft

To run PCL on MineCraft, run `minecraft.py`.

### GPT-2 Finetuned Models

GPT-2 finetuned models can be found under `gpt-2-ft` directory. After unzipping the shards, each model can be invoked using the provided script `talk.py`.

