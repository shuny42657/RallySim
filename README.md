# RallySim
Project Repository for the paper "RallySim : Simulated Environment with Continuous Control for Turn-based Multi-agent Reinforcement Learning"

## Requirements
ml-agents (https://github.com/Unity-Technologies/ml-agents/tree/release_20).
mlagents-envs (https://github.com/Unity-Technologies/ml-agents/tree/release_20).
torch 1.8.1.

## Usage
Assuiming the repository is cloned into a local device and all required libiraries are installed.

## Low-level
Go to ./low_level_training and run
```
python3 training.py --total-step-num=TOTAL_STEP_NUM --model-save-freq=MODEL_SAVE_FREQ

##TOTAL_STEP_NUM : Total number of training steps
##MODEL_SAVE_FREQ : How many training steps there are from saving model parameters at one time to the next.
```


