# Holdem-DCRM

Implementantion of the regret minimization method described in the paper "Deep Counterfactual Regret Minimization" (https://arxiv.org/abs/1811.00164) for NL holdem


## TODO:
- Better evaluation (approximate exploitability: https://arxiv.org/abs/2004.09677) [ x ]
- Implement outcome sampling regret minimization [ ]
- Sync model parameters in case of multiple server hosts [ ]
- Implement single deep counterfactual regret minimization (https://arxiv.org/abs/1901.07621) [ ]


## Usage
### Configure the hosts in config.py
```python
# Host responsible for coordinating the algorithm execution, 
# it commands the slave workers and triggers network training
MASTER_HOST = 'localhost:50040'

# List of hosts which run traversals and evaluations
SLAVE_HOSTS = [
    'localhost:50041'
]

# The host which is responsible for global strategy reservoir sampling and training 
GLOBAL_STRATEGY_HOST = 'localhost:50050'

# A mapping from inference host to player, 
# specifying which host provides regret and strategy inference for which player(s)
ACTOR_HOST_PLAYER_MAP = {
    # To have own process for each player:
    'localhost:50051': [0],
    'localhost:50052': [1]
    # OR to have both on same process:
    # 'localhost:50051': [*range(2)]
}

# A mapping from regret host to player, 
# specifying which host provides regret reservoir sampling and training to which player(s)
REGRET_HOST_PLAYER_MAP = {
    # To have own process for each player:
    'localhost:50061': [0],
    'localhost:50062': [1]]
    # OR to have both on same process:
    # 'localhost:50061': [*range(2)]
}
```
### Start up the server(s)
```shell
./dcrm.py server -hosts hostname1:port1 hostname1:port2 hostname1:port3 ...
```
(note: hostname:port needs to be added to the config file lists)
### Start up the slave worker(s)
```shell
./dcrm.py slave -host hostname:port
```
(note: hostname:port needs to be added to the config file lists)
### Start up the master 
```shell
./dcrm.py master
```
(note: needs to be ran on the master host specified in the config file)
