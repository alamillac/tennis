[//]: # (Image References)

[image1]: tennis.png "Trained Agent"
[image2]: soccer.png "Trained Agent"

# Project 2: Tennis

## Introduction

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

![Trained Agent][image1]

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

## Getting Started

### Requirements
- Python 3.6 or later
- Conda

### Installation
1. Clone the repository
2. Create a conda environment
```bash
conda create --name drlnd python=3.11
```
3. Activate the environment
```bash
conda activate drlnd
```
4. Install the required packages
```bash
pip install -r requirements.txt
```
5. Install the requirements from the `python` folder
```bash
cd python
pip install .
```
6. Download the Unity environments from one of the links below. You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

7. Unzip the file and place it in the root directory of the repository

### (Optional) Play Soccer

The goal is to train a small team of agents to play soccer.

![Trained Agent][image1]

You can read more about this environment in the ML-Agents GitHub [here](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md).

You need only select the environment that matches your operating system:

- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Linux.zip)
- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer.app.zip)
- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86.zip)
- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Soccer/Soccer_Windows_x86_64.zip)

Unzip the file and place it in the root directory of the repository

### Testing the Agent

There are already pre-trained weights in the `tennis.pth` file.

To test the agent, run the `test.py` script.

```bash
python src/test.py
```

### Training the Agent

To train the agent, run the `train.py` script.

```bash
python src/train.py
```

## Implementation

In this project, I used TD3 algorithm with Priority Experience Replay (PER) to solve the environment.

## Report

The report for this project can be found in the `src/Report.ipynb` file.

To view the report, you need to have Jupyter Notebook installed.

```bash
jupyter notebook src/Report.ipynb
```

## Tensorboard

To visualize the training process, you can use Tensorboard.

```bash
tensorboard --logdir=runs
```
