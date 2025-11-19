"""
Implementation of Connect Four using JAX and MCTX.
Last modified: 19.11.2025
"""

import os
import pickle
import time
import random
import glob
import re
from collections import deque
from typing import NamedTuple, Tuple
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk
import mctx
import optax
import numpy as np
import matplotlib.pyplot as plt


ROWS = 6
COLS = 7
NUM_ACTIONS = 7

BATCH_SIZE = 128  
SIMULATIONS = 50
LEARNING_RATE_START = 1e-3
LEARNING_RATE_END = 1e-5
BUFFER_CAPACITY = 50000
MIN_BUFFER_SIZE = 5000
TRAINING_STEPS_DEFAULT = 2000

MODEL_FILE = "connect4_final_model.pkl"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 10

class State(NamedTuple):
    """
    Define the board state.
    """
    board: jnp.ndarray  # (B, 6, 7)
    player: jnp.ndarray # (B,)  1.0/-1.0
    turn: jnp.ndarray   # (B,)
    won: jnp.ndarray    # (B,)


class ReplayBuffer:
    """
    The replay buffer stores games as (boards, policies, values)
    tuples in a deque.
    """
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, boards, policies, values):
        boards_np = np.array(boards)
        policies_np = np.array(policies)
        values_np = np.array(values)
        for i in range(boards.shape[0]):
            self.buffer.append((boards_np[i], policies_np[i], values_np[i]))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        b_list, p_list, v_list = zip(*batch)
        return (
            jnp.array(list(b_list)),
            jnp.array(list(p_list)),
            jnp.array(list(v_list))
        )

    def __len__(self):
        return len(self.buffer)


def get_legal_moves(board):
    return board[:, 0, :] == 0

@jax.jit
def check_immediate_wins(state: State):
    """
    Checks all 7 columns to see if placing a piece there results in an 
    immediate win, for changing the logits of MCTS.
    Args:
        state: The current state of the game.
    Returns:
        winning_moves: Boolean array (B, 7) where True means 'Action i wins'.
    """
    B = state.board.shape[0]


    flat_boards = jnp.repeat(state.board, NUM_ACTIONS, axis=0) # (B*7, 6, 7)
    flat_players = jnp.repeat(state.player, NUM_ACTIONS, axis=0)
    flat_turns = jnp.repeat(state.turn, NUM_ACTIONS, axis=0)
    flat_won = jnp.repeat(state.won, NUM_ACTIONS, axis=0)

    flat_state = State(flat_boards, flat_players, flat_turns, flat_won)

    # 2. Create actions [0, 1, ..., 6, 0, 1, ...]
    actions = jnp.tile(jnp.arange(NUM_ACTIONS), B) # (B*7,)

    # 3. Step the environment for all possibilities
    next_states = step_env(flat_state, actions)

    # 4. Reshape the 'won' array back to (B, 7)
    winning_moves = next_states.won.reshape(B, NUM_ACTIONS)

    # 5. Ensure we don't count moves in full columns as wins (illegal moves)
    legal = get_legal_moves(state.board)
    return winning_moves & legal

def check_win_batched(board):
    """
    Checks if any of the 7 columns results in a win.
    Args:
        board: The current board state.
    Returns:
        won: Boolean array (B,), True=win.
    """
    horizontal = jnp.ones((1, 4))
    vertical = jnp.ones((4, 1))
    diag_1 = jnp.eye(4)
    diag_2 = jnp.fliplr(diag_1)
    kernels = [horizontal, vertical, diag_1, diag_2]

    x = board[:, None, :, :]
    won = jnp.zeros(board.shape[0], dtype=jnp.bool_)
    for k in kernels:
        k = k[None, None, :, :]
        conv = jax.lax.conv(x, k, (1, 1), 'VALID')
        detected = jnp.max(jnp.abs(conv), axis=(1, 2, 3)) >= 4
        won = jnp.logical_or(won, detected)
    return won


def step_env(state: State, action: jnp.ndarray) -> State:
    """
    Steps the environment forward by one move.
    """
    col_mask = jax.nn.one_hot(action, COLS)
    col_values = jnp.sum(state.board * col_mask[:, None, :], axis=2)
    is_empty = (col_values == 0)
    row_idx = jnp.sum(is_empty, axis=1).astype(jnp.int32) - 1

    row_mask = jax.nn.one_hot(row_idx, ROWS)
    move_mask = row_mask[:, :, None] * col_mask[:, None, :]
    new_board = state.board + move_mask * state.player[:, None, None]
    just_won = check_win_batched(new_board)

    return State(
        board=new_board,
        player=-state.player,
        turn=state.turn + 1,
        won=just_won
    )


class AlphaZeroNet(hk.Module):
    """
    Updated Architecture: 128 Filters, 7 ResBlocks, BatchNorm.    
    """
    def __init__(self, num_actions, num_res_blocks=7, num_filters=128):
        super().__init__()
        self.num_actions = num_actions
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters

    def __call__(self, x, is_training):
        # Initial Conv
        x = hk.Conv2D(self.num_filters, kernel_shape=3, stride=1)(x)
        x = hk.BatchNorm(create_scale=True,
                         create_offset=True, decay_rate=0.99)(x, is_training=is_training)
        x = jax.nn.relu(x)

        # Residual Tower
        for _ in range(self.num_res_blocks):
            x = self.res_block(x, is_training)

        # Policy Head
        p = hk.Conv2D(2, kernel_shape=1)(x)
        p = hk.BatchNorm(create_scale=True,
                         create_offset=True, decay_rate=0.99)(p, is_training=is_training)
        p = jax.nn.relu(p)
        p = hk.Flatten()(p)
        logits = hk.Linear(self.num_actions)(p)

        # Value Head
        v = hk.Conv2D(1, kernel_shape=1)(x)
        v = hk.BatchNorm(create_scale=True,
                         create_offset=True, decay_rate=0.99)(v, is_training=is_training)
        v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(256)(v) 
        v = jax.nn.relu(v)
        value = hk.Linear(1)(v)
        value = jnp.tanh(value)

        return logits, jnp.squeeze(value, axis=-1)


    def res_block(self, x, is_training):
        residual = x
        out = hk.Conv2D(self.num_filters, kernel_shape=3, stride=1)(x)
        out = hk.BatchNorm(create_scale=True,
                           create_offset=True, decay_rate=0.99)(out, is_training=is_training)
        out = jax.nn.relu(out)

        out = hk.Conv2D(self.num_filters, kernel_shape=3, stride=1)(out)
        out = hk.BatchNorm(create_scale=True, 
                           create_offset=True, decay_rate=0.99)(out, is_training=is_training)
        return jax.nn.relu(out + residual)


def forward_fn(x, is_training=True):
    net = AlphaZeroNet(num_actions=NUM_ACTIONS, num_res_blocks=7, num_filters=128)
    return net(x, is_training)

network = hk.without_apply_rng(hk.transform_with_state(forward_fn))


@jax.jit
def make_input(state):
    canonical_board = state.board * state.player[:, None, None]
    my_pieces = (canonical_board == 1.0).astype(jnp.float32)
    opp_pieces = (canonical_board == -1.0).astype(jnp.float32)
    return jnp.stack([my_pieces, opp_pieces], axis=-1)


def root_fn(params, rng_key, embedding, network_state):
    """


    """
    current_state = embedding
    network_input = make_input(current_state)

    (logits, value), _ = network.apply(params, network_state, network_input, is_training=False)

    legal = get_legal_moves(current_state.board)

    winning_moves = check_immediate_wins(current_state)
    can_win = jnp.any(winning_moves, axis=-1, keepdims=True)

    dummy_opp_state = current_state._replace(player=-current_state.player)
    opp_winning_moves = check_immediate_wins(dummy_opp_state)
    must_block = jnp.any(opp_winning_moves, axis=-1, keepdims=True)

    base_logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)

    logits_with_blocks = jnp.where(
        must_block,
        jnp.where(opp_winning_moves, 100.0, jnp.finfo(jnp.float32).min), # Boost blocking moves
        base_logits
    )

    # Apply Win Logic (Overrides Blocks because Winning > Blocking)
    final_logits = jnp.where(
        can_win,
        jnp.where(winning_moves, 200.0, jnp.finfo(jnp.float32).min), # Boost winning moves
        logits_with_blocks
    )

    return mctx.RootFnOutput(prior_logits=final_logits, value=value, embedding=embedding)


def recurrent_fn(params, rng_key, action, embedding, network_state):
    current_state = embedding
    next_state = step_env(current_state, action)

    reward = jnp.where(next_state.won, 1.0, 0.0)
    discount = jnp.where(next_state.won, 0.0, 1.0)

    network_input = make_input(next_state)
    (logits, value), _ = network.apply(params, network_state, network_input, is_training=False)

    legal = get_legal_moves(next_state.board)
    winning_moves = check_immediate_wins(next_state)
    can_win = jnp.any(winning_moves, axis=-1, keepdims=True)

    logits = jnp.where(
        can_win,
        jnp.where(winning_moves, 100.0, jnp.finfo(jnp.float32).min),
        jnp.where(legal, logits, jnp.finfo(jnp.float32).min)
    )

    output = mctx.RecurrentFnOutput(
        reward=reward, discount=discount, prior_logits=logits, value=value
    )
    return output, next_state


@jax.jit
def run_mcts(params, network_state, rng_key, state):
    # Bind network_state to functions so MCTX can call them
    root_with_state = partial(root_fn, network_state=network_state)
    recurrent_with_state = partial(recurrent_fn, network_state=network_state)

    root_output = root_with_state(params, rng_key, state)

    return mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root_output,
        recurrent_fn=recurrent_with_state,
        num_simulations=SIMULATIONS,
        invalid_actions=~get_legal_moves(state.board),
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0
    )


def loss_fn(params, state, batch):
    boards, target_policy, target_value = batch

    my_pieces = (boards == 1.0).astype(jnp.float32)
    opp_pieces = (boards == -1.0).astype(jnp.float32)
    network_input = jnp.stack([my_pieces, opp_pieces], axis=-1)

    # Training mode: is_training=True (Updates BN stats)
    (logits, value), new_state = network.apply(params, state, network_input, is_training=True)

    log_probs = jax.nn.log_softmax(logits)
    policy_loss = -jnp.mean(jnp.sum(target_policy * log_probs, axis=-1))
    value_loss = jnp.mean((target_value - value) ** 2)

    total_loss = policy_loss + value_loss

    return total_loss, (new_state, policy_loss, value_loss)


@jax.jit
def train_step(params, network_state, opt_state, batch):
    grads, (new_network_state, p_loss, v_loss) = jax.grad(
        loss_fn, has_aux=True)(params, network_state, batch)

    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, new_network_state, new_opt_state, p_loss, v_loss


# Optimizer with Schedule
total_steps = 2500 # Estimate for decay
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0,
    peak_value=LEARNING_RATE_START,
    warmup_steps=50,
    decay_steps=total_steps,
    end_value=LEARNING_RATE_END
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=schedule)
)


def run_self_play_episode(params, network_state, key):
    state = State(
        board=jnp.zeros((BATCH_SIZE, ROWS, COLS)),
        player=jnp.ones((BATCH_SIZE,)),
        turn=jnp.zeros((BATCH_SIZE,), dtype=jnp.int32),
        won=jnp.zeros((BATCH_SIZE,), dtype=jnp.bool_)
    )

    history_boards = []
    history_policies = []
    history_players = []
    valid_mask = jnp.ones((BATCH_SIZE,), dtype=jnp.bool_)
    history_validity = []

    TEMP_THRESHOLD = 15

    for _ in range(ROWS * COLS):
        key, subkey = jax.random.split(key)
        # Pass network_state
        policy_output = run_mcts(params, network_state, subkey, state)

        action_logits = jnp.log(policy_output.action_weights + 1e-8)
        sampled_action = jax.random.categorical(subkey, action_logits)
        argmax_action = jnp.argmax(policy_output.action_weights, axis=-1)

        should_explore = state.turn < TEMP_THRESHOLD
        action = jnp.where(should_explore, sampled_action, argmax_action)

        canonical_board = state.board * state.player[:, None, None]
        history_boards.append(canonical_board)
        history_policies.append(policy_output.action_weights)
        history_players.append(state.player)
        history_validity.append(valid_mask)

        state = step_env(state, action)
        valid_mask = jnp.logical_and(valid_mask, ~state.won)

    p1_wins = check_win_batched(state.board * 1.0)
    p2_wins = check_win_batched(state.board * -1.0)
    outcomes = jnp.zeros((BATCH_SIZE,))
    outcomes = jnp.where(p1_wins, 1.0, outcomes)
    outcomes = jnp.where(p2_wins, -1.0, outcomes)

    all_boards = jnp.stack(history_boards)
    all_pis = jnp.stack(history_policies)
    all_masks = jnp.stack(history_validity)

    all_outcomes = jnp.broadcast_to(outcomes[None, :], all_boards.shape[:2])
    all_players = jnp.stack(history_players)
    all_vs = all_outcomes * all_players

    flat_boards = jnp.reshape(all_boards, (-1, ROWS, COLS))
    flat_pis = jnp.reshape(all_pis, (-1, NUM_ACTIONS))
    flat_vs = jnp.reshape(all_vs, (-1,))
    flat_masks = jnp.reshape(all_masks, (-1,))

    valid_indices = flat_masks > 0

    return (
        flat_boards[valid_indices],
        flat_pis[valid_indices],
        flat_vs[valid_indices]
    )


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_checkpoint(params, network_state, step):
    ensure_dir(CHECKPOINT_DIR)
    filename = os.path.join(CHECKPOINT_DIR, f"step_{step}.pkl")
    # Save both params and BN state
    data = {"params": params, "state": network_state}
    with open(filename, "wb") as f:
        pickle.dump(data, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(data, f)


def load_checkpoint(filename):
    print(f"Loading from {filename}...")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    # Handle legacy files vs new files
    if isinstance(data, dict) and "params" in data:
        return data["params"], data["state"]
    else:
        print("Warning: Legacy format detected. BN State lost.")
        return data, None


def select_checkpoint():
    files = glob.glob(f"{CHECKPOINT_DIR}/*.pkl") + glob.glob("*.pkl")
    files = sorted(list(set(files)))
    if not files: 
        return None, 0

    print("\n--- AVAILABLE MODELS ---")
    for i, f in enumerate(files): 
        print(f"{i+1}. {f}")
    print("0. Start Fresh")

    choice = input("Select: ")
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(files):
            filename = files[idx]
            step_match = re.search(r"step_(\d+)", filename)
            start_step = int(step_match.group(1)) if step_match else 0
            return filename, start_step
    return None, 0


def plot_history(p_hist, v_hist):
    if not p_hist: return
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(p_hist)
    plt.title("Policy Loss")
    plt.subplot(1, 2, 2)
    plt.plot(v_hist, color='orange')
    plt.title("Value Loss")
    plt.show()


def print_board_pretty(board):
    print("\n 0 1 2 3 4 5 6")
    print("---------------")
    for r in range(ROWS):
        row_str = "|"
        for c in range(COLS):
            v = board[r, c]
            row_str += ("X" if v==1 else "O" if v==-1 else " ") + "|"
        print(row_str)
    print("---------------\n")


def play_human_vs_ai(params, network_state, human_is_player_1=True):
    print(f"\n{'='*30}\n HUMAN vs AI\n{'='*30}")
    state = State(
        board=jnp.zeros((1, ROWS, COLS), dtype=jnp.float32),
        player=jnp.ones((1,), dtype=jnp.float32),
        turn=jnp.zeros((1,), dtype=jnp.int32),
        won=jnp.zeros((1,), dtype=jnp.bool_)
    )

    PLAY_SIMULATIONS = 400

    @jax.jit
    def play_mcts(k, s):
        root_with_state = partial(root_fn, network_state=network_state)
        recurrent_with_state = partial(recurrent_fn, network_state=network_state)
        
        root_output = root_with_state(params, k, s)
        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=k,
            root=root_output,
            recurrent_fn=recurrent_with_state,
            num_simulations=PLAY_SIMULATIONS,
            invalid_actions=~get_legal_moves(s.board),
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0
        )

    key = jax.random.PRNGKey(int(time.time()))

    while True:
        board_np = np.array(state.board[0])
        print_board_pretty(board_np)

        curr_p = state.player[0]
        is_human = (curr_p == 1) if human_is_player_1 else (curr_p == -1)

        if is_human:
            while True:
                try:
                    col = int(input(f"Your move ({'X' if human_is_player_1 else 'O'}): "))
                    if 0 <= col < COLS and board_np[0, col] == 0:
                        action = jnp.array([col])
                        break
                except:
                    pass

        else:
            print(f"AI Thinking ({PLAY_SIMULATIONS} sims)...")
            key, subkey = jax.random.split(key)
            out = play_mcts(subkey, state)
            col = np.argmax(out.action_weights[0])
            action = jnp.array([col])
            print(f"AI chose column: {col}")

        state = step_env(state, action)
        if state.won[0]:
            print_board_pretty(state.board[0])
            print(f"*** {'HUMAN' if is_human else 'AI'} WINS! ***")
            break
        if jnp.all(state.board != 0):
            print_board_pretty(state.board[0])
            print("DRAW")
            break


if __name__ == "__main__":
    print(f"Running on {jax.devices()[0]}")

    key = jax.random.PRNGKey(42)
    dummy = jnp.zeros((1, ROWS, COLS, 2)) 

    # Init params AND network_state
    params, network_state = network.init(key, dummy, is_training=True)
    opt_state = optimizer.init(params)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    filename, total_steps = select_checkpoint()
    if filename:
        try:
            params, loaded_state = load_checkpoint(filename)
            if loaded_state is not None:
                network_state = loaded_state
            opt_state = optimizer.init(params)
            print(f"Loaded '{filename}' (Resuming from step {total_steps})")
        except Exception as e:
            print(f"Load failed: {e}. Starting fresh.")
            # Re-init
            params, network_state = network.init(key, dummy, is_training=True)
            opt_state = optimizer.init(params)
            total_steps = 0
    else:
        print("Starting fresh model.")
        total_steps = 0

    p_hist, v_hist = [], []

    while True:
        print(f"\n--- MENU (Current Step: {total_steps}) ---")
        print("1. Train")
        print("2. Play AI")
        print("3. Plot Loss")
        print("4. Force Save & Exit")
        c = input("Choice: ")

        if c == "1":
            n = int(input(f"Steps to train (Default {TRAINING_STEPS_DEFAULT}): ") or TRAINING_STEPS_DEFAULT)
            print("Starting...")
            for i in range(n):
                key, subkey = jax.random.split(key)
                
                # Pass network_state to self-play
                b, p, v = run_self_play_episode(params, network_state, subkey)
                replay_buffer.add(b, p, v)

                if len(replay_buffer) > MIN_BUFFER_SIZE:
                    batch = replay_buffer.sample(BATCH_SIZE)

                    # Pass network_state to train, receive updated state
                    params, network_state, opt_state, pl, vl = train_step(params, network_state,
                                                                           opt_state, batch)

                    p_hist.append(float(pl))
                    v_hist.append(float(vl))

                    total_steps += 1
                    if total_steps % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(params, network_state, total_steps)
                        print(f"\nCheckpoint saved at step {total_steps}")
                    print(f"Step {i+1}/{n} (Total {total_steps}) | Buff: {len(
                        replay_buffer)} | P: {pl:.3f} V: {vl:.3f}", end="\r")
                else:
                    print(f"Filling Buffer: {len(replay_buffer)}/{MIN_BUFFER_SIZE}", end="\r")
            print("\nDone.")
            save_checkpoint(params, network_state, total_steps)

        elif c == "2": 
            play_human_vs_ai(params, network_state, input("Be Player 1 (X)? [y/n]: ").lower()=='y')

        elif c == "3": 
            plot_history(p_hist, v_hist)

        elif c == "4": 
            save_checkpoint(params, network_state, total_steps)
            break
