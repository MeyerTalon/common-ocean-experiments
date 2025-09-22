# common-ocean-experiments
This repo holds the code created to complete the take home task for the NSF IRES Project on Autonomous Marine Operations for the Arctic.

## Table of Contents
- [Getting Started](#getting-started)
- [Implementation Details](#implementation-details)
- [Citation](#citation)


## Getting Started
Create a new Anaconda environment for Python 3.8:

```bash
conda create −n foo python=3.8
conda activate foo
```

Install the packages outlines in requirements.txt:
```bash
pip install -r requirements.txt
```

## Research Questions

- Reinforcement learning model (current favorite) vs rule based model vs some secret third option?
- For an RL model, do we output the complete trajectory given an input state or output $u(t) \in \mathbb{R}^m$ the control input at each time step? Or a combination of both?
- How do we integrate the model with the CommonOcean-IO framework?
- For scenario generation, do we want to use/train on one set scenario, or any randomly generated scenario within the a given scenario bounds?
- How do we randomly generate a scenario?


## Implementation Details

**Three Degrees of Freedom Model**

State of the ego vessel at time $t$ is $x(t) \in \mathbb{R}^n$.


**Note to self:** Keep the model simple, should output the

## Citation

```
@inproceedings{Krasowski2022a,
	author = {Krasowski, Hanna and Althoff, Matthias},
	title = {CommonOcean: Composable Benchmarks for Motion Planning on Oceans},
	booktitle = {Proc. of the IEEE International Conference on Intelligent Transportation Systems},
	year = {2022},
}
```