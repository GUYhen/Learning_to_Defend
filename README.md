# Learning to Defend: Multi-Agent RL for Stackelberg Security Game

This repository contains the code for our paper accepted at **ICNC 2026**.

**Title:** Learning to Defend: A Multi-Agent Reinforcement Learning Framework for Stackelberg Security Game in Mobile Edge Computing

**Conference:** International Conference on Computing, Networking and Communications (ICNC 2026)

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{DingICNC2025,
  title={{Learning to Defend: A Multi-Agent Reinforcement Learning Framework for Stackelberg Security Game in Mobile Edge Computing}},
  author={Ding, Zihao and Huang, Jun and Qi, Junjian},
  booktitle={International Conference on Computing, Networking and Communications (ICNC)},
  year={2026},
  month={February},
  address={Honolulu, Hawaii, USA},
  organization={IEEE}
}
```

## Requirements

```
xuance==1.2.5
gymnasium==0.29.1
torch==2.1.0
tensorboard==2.14.0
numpy==1.24.3
scipy==1.11.1
pandas==2.0.3
networkx==3.1
matplotlib==3.7.2
seaborn==0.12.2
pyyaml==6.0.1
tqdm==4.66.1
gym==0.23.1
pettingzoo==1.24.3
argparse
pickle-mixin
```

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
Code/
├── train_xuance_ippo_enhanced.py
├── requirements.txt
├── environment/
│   ├── __init__.py
│   ├── network_topology.py
│   ├── stackelberg_env.py
│   └── xuance_env_wrapper.py
├── config/
│   ├── env_config.yaml
│   ├── ippo_stackelberg_config.yaml
│   └── test_config.yaml
└── utils/
    ├── __init__.py
    ├── logger.py
    └── visualizer.py
```

## Usage

### Training with IPPO

```bash
python train_xuance_ippo_enhanced.py --critical_nodes 10
```

## Parameters

- `--critical_nodes`: Number of critical nodes (default: 10, choices: [5, 10, 15])

## Outputs

Training results are saved in:

- `models/`: Trained model checkpoints
- `logs/`: Training logs and TensorBoard files
- `results/`: Training metrics and visualization plots

## Algorithms

This codebase implements three multi-agent reinforcement learning algorithms:

 **IPPO** (Independent Proximal Policy Optimization)


## Environment

The environment simulates a Stackelberg security game in Mobile Edge Computing:

- **Network:** node topology with configurable critical nodes
- **Agents:** Defender (leader) and Attacker (follower)
- **Objective:** Defender protects critical nodes while Attacker attempts to compromise them

## License

This code is provided for academic research purposes only.



