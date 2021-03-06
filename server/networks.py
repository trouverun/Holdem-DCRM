import torch
import logging
from pokerenv.obs_indices import VALID_ACTIONS, VALID_BET_LOW, VALID_BET_HIGH, POT_SIZE
from torch.nn.utils.rnn import pack_padded_sequence
from torch.distributions import Categorical
from config import OBS_SHAPE, N_BET_BUCKETS, N_ACTIONS, BET_BUCKETS, DCRM_RNN_HIDDENS, MCTS_RNN_HIDDENS,  PPO_RNN_HIDDENS
from torch.nn.functional import log_softmax


ACTIONS_START = VALID_ACTIONS[0]
ACTIONS_END = VALID_ACTIONS[-1]
NULL_VALUE = -1e5


class RegretNetwork(torch.nn.Module):
    def __init__(self, device, n_hiddens=DCRM_RNN_HIDDENS, bet_buckets=N_BET_BUCKETS):
        super().__init__()
        self.device = device
        self.bet_buckets = bet_buckets
        self.rnn = torch.nn.GRU(OBS_SHAPE, n_hiddens, 1, batch_first=True)
        self.action_regret = torch.nn.Linear(n_hiddens, N_ACTIONS)
        self.bet_regret = torch.nn.Linear(n_hiddens, self.bet_buckets)

    def forward(self, x, starts):
        invalid_actions = x[torch.arange(0, x.shape[0]), starts - 1, ACTIONS_START:ACTIONS_END+1] == 0
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

        # Give 0 regret to all bet sizes that are outside of the valid range
        # TODO: default bets [bucket = 0,1,2] may still be 'valid'?
        action_regret = torch.where(invalid_actions, torch.zeros_like(action_regret), action_regret)
        betsize_array = torch.tensor(BET_BUCKETS)
        betsize_array = betsize_array.repeat((x.shape[0], 1)).to(self.device)
        betsize_array = betsize_array * pot_sizes[:, None]
        betsize_array = torch.cat([valid_bet_low.unsqueeze(1), valid_bet_mid.unsqueeze(1), valid_bet_high.unsqueeze(1), betsize_array],
                                  dim=1)
        bet_regret = torch.where(((valid_bet_low_r > betsize_array).to(torch.bool) | (valid_bet_high_r < betsize_array).to(torch.bool)),
                                 torch.zeros_like(bet_regret), bet_regret)
        # When bet action is invalid give all bet sizes 0 regret
        bet_invalid = (invalid_actions[:, 2] == 1).unsqueeze(1).repeat(1, N_BET_BUCKETS).to(self.device)
        bet_regret = torch.where(bet_invalid.to(torch.bool), torch.zeros_like(bet_regret), bet_regret)
        return action_regret, bet_regret


class PolicyNetwork(torch.nn.Module):
    def __init__(self, device, n_hiddens=DCRM_RNN_HIDDENS, bet_buckets=N_BET_BUCKETS):
        super().__init__()
        self.device = device
        self.bet_buckets = bet_buckets
        self.rnn = torch.nn.GRU(OBS_SHAPE, n_hiddens, 1, batch_first=True)
        self.action_logits = torch.nn.Linear(n_hiddens, N_ACTIONS)
        self.bet_logits = torch.nn.Linear(n_hiddens, self.bet_buckets)

    def forward(self, x, starts, return_dist=False):
        invalid_actions = x[torch.arange(0, x.shape[0]), starts - 1, ACTIONS_START:ACTIONS_END+1] == 0
        valid_bet_low = x[torch.arange(0, x.shape[0]), starts - 1, VALID_BET_LOW]
        valid_bet_low_r = valid_bet_low.unsqueeze(0).T.repeat((1, self.bet_buckets))
        valid_bet_high = x[torch.arange(0, x.shape[0]), starts - 1, VALID_BET_HIGH]
        valid_bet_high_r = valid_bet_high.unsqueeze(0).T.repeat((1, self.bet_buckets))
        valid_bet_mid = (valid_bet_low + valid_bet_high) / 2
        pot_sizes = x[torch.arange(0, x.shape[0]), starts - 1, POT_SIZE]

        packed = pack_padded_sequence(x, starts, batch_first=True, enforce_sorted=False)
        x, h = self.rnn(packed)
        x = h[-1]
        action_logits = self.action_logits(x)
        bet_logits = self.bet_logits(x)

        # Give NULL_VALUE logits to all bet sizes that are outside of the valid range
        action_logits = torch.where(invalid_actions, NULL_VALUE * torch.ones_like(action_logits), action_logits)
        all_infs = torch.all(action_logits == NULL_VALUE, dim=1).unsqueeze(0).T.repeat((1, N_ACTIONS))
        action_logits = torch.where(all_infs, torch.zeros_like(action_logits), action_logits)
        betsize_array = torch.tensor(BET_BUCKETS)
        betsize_array = betsize_array.repeat((x.shape[0], 1)).to(self.device)
        betsize_array = betsize_array * pot_sizes[:, None]
        betsize_array = torch.cat([valid_bet_low.unsqueeze(1), valid_bet_mid.unsqueeze(1), valid_bet_high.unsqueeze(1), betsize_array],
                                  dim=1)
        bet_logits = torch.where(((valid_bet_low_r > betsize_array).to(torch.bool) | (valid_bet_high_r < betsize_array).to(torch.bool)),
                                 NULL_VALUE * torch.ones_like(bet_logits), bet_logits)
        all_infs = torch.all(bet_logits == NULL_VALUE, dim=1).unsqueeze(0).T.repeat((1, self.bet_buckets))
        bet_logits = torch.where(all_infs, torch.zeros_like(bet_logits), bet_logits)
        # When bet action is invalid make all corresponding bet logits NULL_VALUE
        bet_invalid = (invalid_actions[:, 2] == 1).unsqueeze(1).repeat(1, N_BET_BUCKETS).to(self.device)
        bet_logits = torch.where(bet_invalid.to(torch.bool), NULL_VALUE * torch.ones_like(bet_logits), bet_logits)

        # action_dist = log_softmax(action_logits, dim=1)
        # bet_dist = log_softmax(bet_logits, dim=1)
        if return_dist:
            action_dist = Categorical(logits=action_logits)
            bet_dist = Categorical(logits=bet_logits)
        else:
            action_dist = log_softmax(action_logits, dim=1)
            bet_dist = log_softmax(bet_logits, dim=1)

        return action_dist, bet_dist


class ValueNetwork(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.rnn = torch.nn.GRU(OBS_SHAPE, PPO_RNN_HIDDENS, 1, batch_first=True)
        self.value = torch.nn.Linear(PPO_RNN_HIDDENS, 1)

    def forward(self, x, starts):
        # logging.info(x.shape)
        # logging.info(starts)
        packed = pack_padded_sequence(x, starts, batch_first=True, enforce_sorted=False)
        x, h = self.rnn(packed)
        x = h[-1]
        return self.value(x)


class EvaluatorNetwork(torch.nn.Module):
    def __init__(self, device, bet_buckets=N_BET_BUCKETS):
        super().__init__()
        self.device = device
        self.bet_buckets = bet_buckets
        self.rnn = torch.nn.GRU(OBS_SHAPE, MCTS_RNN_HIDDENS, 1, batch_first=True)
        self.value = torch.nn.Linear(MCTS_RNN_HIDDENS, 1)
        self.action_logits = torch.nn.Linear(MCTS_RNN_HIDDENS, N_ACTIONS)
        self.bet_logits = torch.nn.Linear(MCTS_RNN_HIDDENS, self.bet_buckets)

    def forward(self, x, starts):
        invalid_actions = x[torch.arange(0, x.shape[0]), starts - 1, ACTIONS_START:ACTIONS_END+1] == 0
        valid_bet_low = x[torch.arange(0, x.shape[0]), starts - 1, VALID_BET_LOW]
        valid_bet_low_r = valid_bet_low.unsqueeze(0).T.repeat((1, self.bet_buckets))
        valid_bet_high = x[torch.arange(0, x.shape[0]), starts - 1, VALID_BET_HIGH]
        valid_bet_high_r = valid_bet_high.unsqueeze(0).T.repeat((1, self.bet_buckets))
        valid_bet_mid = (valid_bet_low + valid_bet_high) / 2
        pot_sizes = x[torch.arange(0, x.shape[0]), starts - 1, POT_SIZE]

        packed = pack_padded_sequence(x, starts, batch_first=True, enforce_sorted=False)
        x, h = self.rnn(packed)
        x = h[-1]
        action_logits = self.action_logits(x)
        bet_logits = self.bet_logits(x)
        value = self.value(x)

        # Give NULL_VALUE logits to all bet sizes that are outside of the valid range
        action_logits = torch.where(invalid_actions, NULL_VALUE * torch.ones_like(action_logits), action_logits)
        all_infs = torch.all(action_logits == NULL_VALUE, dim=1).unsqueeze(0).T.repeat((1, N_ACTIONS))
        action_logits = torch.where(all_infs, torch.zeros_like(action_logits), action_logits)
        betsize_array = torch.tensor(BET_BUCKETS)
        betsize_array = betsize_array.repeat((x.shape[0], 1)).to(self.device)
        betsize_array = betsize_array * pot_sizes[:, None]
        betsize_array = torch.cat([valid_bet_low.unsqueeze(1), valid_bet_mid.unsqueeze(1), valid_bet_high.unsqueeze(1), betsize_array],
                                  dim=1)
        bet_logits = torch.where(((valid_bet_low_r > betsize_array).to(torch.bool) | (valid_bet_high_r < betsize_array).to(torch.bool)),
                                 NULL_VALUE * torch.ones_like(bet_logits), bet_logits)
        all_infs = torch.all(bet_logits == NULL_VALUE, dim=1).unsqueeze(0).T.repeat((1, self.bet_buckets))
        bet_logits = torch.where(all_infs, torch.zeros_like(bet_logits), bet_logits)
        # When bet action is invalid make all corresponding bet logits NULL_VALUE
        bet_invalid = (invalid_actions[:, 2] == 1).unsqueeze(1).repeat(1, N_BET_BUCKETS).to(self.device)
        bet_logits = torch.where(bet_invalid.to(torch.bool), NULL_VALUE * torch.ones_like(bet_logits), bet_logits)

        action_dist = log_softmax(action_logits, dim=1)
        bet_dist = log_softmax(bet_logits, dim=1)

        return action_dist, bet_dist, value