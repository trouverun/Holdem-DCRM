import torch
from pokerenv.obs_indices import VALID_ACTIONS, VALID_BET_LOW, VALID_BET_HIGH, POT_SIZE
from torch.nn.utils.rnn import pack_padded_sequence
from config import OBS_SHAPE, N_BET_BUCKETS, N_ACTIONS, RNN_HIDDENS, BET_BUCKETS
from math import inf


class RegretNetwork(torch.nn.Module):
    def __init__(self, device, bet_buckets=N_BET_BUCKETS):
        super().__init__()
        self.device = device
        self.bet_buckets = bet_buckets
        self.rnn = torch.nn.GRU(OBS_SHAPE, RNN_HIDDENS, 1, batch_first=True)
        self.action_regret = torch.nn.Linear(RNN_HIDDENS, N_ACTIONS)
        self.bet_regret = torch.nn.Linear(RNN_HIDDENS, self.bet_buckets)

    def forward(self, x, starts):
        obses = x[torch.arange(0, x.shape[0]), starts - 1, :]
        invalid_actions = x[torch.arange(0, x.shape[0]), starts - 1, 3:7] == 0
        valid_bet_low = x[torch.arange(0, x.shape[0]), starts - 1, VALID_BET_LOW]
        valid_bet_low_r = valid_bet_low.unsqueeze(0).T.repeat((1, self.bet_buckets))
        valid_bet_high = x[torch.arange(0, x.shape[0]), starts - 1, VALID_BET_HIGH]
        valid_bet_high_r = valid_bet_high.unsqueeze(0).T.repeat((1, self.bet_buckets))
        valid_bet_mid = (valid_bet_low + valid_bet_high) / 2
        pot_sizes = x[torch.arange(0, x.shape[0]), starts - 1, POT_SIZE]

        packed = pack_padded_sequence(x, starts, batch_first=True, enforce_sorted=False)
        x, h = self.rnn(packed)
        x = h[-1]
        action_regret = self.action_regret(x)
        bet_regret = self.bet_regret(x)

        # Assign invalid actions and invalid bets 0 regret so that they don't contribute to the loss when training
        action_regret = torch.where(invalid_actions, torch.zeros_like(action_regret), action_regret)
        betsize_array = torch.tensor(BET_BUCKETS)
        betsize_array = betsize_array.repeat((x.shape[0], 1)).to(self.device)
        betsize_array = betsize_array * pot_sizes[:, None]
        betsize_array = torch.cat([valid_bet_low.unsqueeze(1), valid_bet_mid.unsqueeze(1), valid_bet_high.unsqueeze(1), betsize_array],
                                  dim=1)
        bet_regret = torch.where(((valid_bet_low_r > betsize_array).to(torch.bool) | (valid_bet_high_r < betsize_array).to(torch.bool)),
                                 torch.zeros_like(bet_regret), bet_regret)

        print(obses)
        print()
        print(action_regret)
        print()
        print(bet_regret)

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
        invalid_actions = x[torch.arange(0, x.shape[0]), starts - 1, VALID_ACTIONS] == 0
        valid_bet_low = x[torch.arange(0, x.shape[0]), starts - 1, VALID_BET_LOW]
        valid_bet_low_r = valid_bet_low.unsqueeze(0).T.repeat((1, self.bet_buckets))
        valid_bet_high = x[torch.arange(0, x.shape[0]), starts - 1, VALID_BET_HIGH]
        valid_bet_high_r = valid_bet_high.unsqueeze(0).T.repeat((1, self.bet_buckets))
        valid_bet_mid = (valid_bet_low + valid_bet_high) / 2
        pot_sizes = x[torch.arange(0, x.shape[0]), starts - 1, POT_SIZE]

        packed = pack_padded_sequence(x, starts, batch_first=True, enforce_sorted=False)
        x, h = self.rnn(packed)
        x = h[-1]
        action_logits = self.action_regret(x)
        bet_logits = self.bet_regret(x)

        # Assign invalid actions and invalid bets 0 regret so that they don't contribute to the loss when training
        action_logits = torch.where(invalid_actions, torch.zeros_like(action_logits), action_logits)
        denom_a = action_logits.abs().sum(dim=1).unsqueeze(1) + 1e-20
        action_logits = torch.where(invalid_actions, -inf*torch.ones_like(action_logits), action_logits.abs()/denom_a)

        betsize_array = torch.tensor(BET_BUCKETS) * pot_sizes
        betsize_array = betsize_array.repeat((x.shape[0], 1)).to(self.device)
        betsize_array = torch.cat([valid_bet_low.unsqueeze(1), valid_bet_mid.unsqueeze(1), valid_bet_high.unsqueeze(1), betsize_array],
                                  dim=1)
        bet_logits = torch.where(((valid_bet_low_r > betsize_array).to(torch.bool) | (valid_bet_high_r < betsize_array).to(torch.bool)),
                                 torch.zeros_like(bet_logits), bet_logits)
        all_infs = torch.all(bet_logits == -inf, dim=1).unsqueeze(0).T.repeat((1, self.bet_buckets))
        bet_logits = torch.where(all_infs, torch.zeros_like(bet_logits), bet_logits)

        return action_logits, bet_logits