# Holdem-DCRM

Implementantion of the regret minimization method described in the paper "Deep Counterfactual Regret Minimization" (https://arxiv.org/abs/1811.00164) for NL holdem

## Usage
###Install dependencies:
```bash
pip install torch numpy pokerenv treys grpcio==1.30.0
```

### Train the strategy network (single machine)
```bash
client.py number_of_CFR_iterations k traversals_per_process n_process
```

### TODO:
* [ ] Implement evaluator module which evaluates performance against older versions
* [ ] Support distributed training and inference with multiple agent and learner machines
