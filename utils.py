from pokergym.table import Table

def copyenv(env):
    new = Table(env.n_players, env.stack_low, env.stack_high)
    new.all_players = env.all_players.copy()