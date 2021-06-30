import torch
from torch.nn.utils.rnn import pack_padded_sequence
from config import OBS_SHAPE, N_BET_BUCKETS, N_ACTIONS, RNN_HIDDENS


class RegretNetwork(torch.nn.Module):
    def __init__(self, device, bet_buckets=N_BET_BUCKETS):
        super().__init__()
        self.device = device
        self.bet_buckets = bet_buckets
        self.rnn = torch.nn.GRU(OBS_SHAPE, RNN_HIDDENS, 1, batch_first=True)
        self.action_regret = torch.nn.Linear(RNN_HIDDENS, N_ACTIONS)
        self.bet_regret = torch.nn.Linear(RNN_HIDDENS, self.bet_buckets)

    def forward(self, x, starts):
        packed = pack_padded_sequence(x, starts, batch_first=True, enforce_sorted=False)
        x, h = self.rnn(packed)
        x = h[-1]

        action_regret = self.action_regret(x)
        bet_regret = self.bet_regret(x)

        return action_regret, bet_regret


class StrategyNetwork(torch.nn.Module):
    def __init__(self, device, bet_buckets=N_BET_BUCKETS):
        super().__init__()
        self.device = device
        self.bet_buckets = bet_buckets
        self.rnn = torch.nn.GRU(OBS_SHAPE, RNN_HIDDENS, 1, batch_first=True)
        self.action_regret = torch.nn.Linear(RNN_HIDDENS, N_ACTIONS)
        self.bet_regret = torch.nn.Linear(RNN_HIDDENS, self.bet_buckets)

    def forward(self, x, starts):
        packed = pack_padded_sequence(x, starts, batch_first=True, enforce_sorted=False)
        x, h = self.rnn(packed)
        x = h[-1]

        action_regret = self.action_regret(x)
        bet_regret = self.bet_regret(x)

        return action_regret, bet_regret