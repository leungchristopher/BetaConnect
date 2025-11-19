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

# Training hyperparameters
BATCH_SIZE = 32  # Batch = B
SIMULATIONS = 256
LEARNING_RATE = 1e-4
BUFFER_CAPACITY = 20000
MIN_BUFFER_SIZE = 2048
TRAINING_STEPS_DEFAULT = 2000

# Files & Directories
MODEL_FILE = "connect4_temperature_model.pkl"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_INTERVAL = 50

class State(NamedTuple):
    board: jnp.ndarray  # (B, 6, 7)
    player: jnp.ndarray # (B,)  1.0/-1.0
    turn: jnp.ndarray   # (B,)
    won: jnp.ndarray    # (B,)


class ReplayBuffer:
    """
    A replay buffer for storing and sampling self-play training data.
    Attributes:
        buffer (deque): (board, policy, value) tuples stored.
    Methods:
        add(boards, policies, values):
            Append a batch of training samples to the buffer.
            Args:
                boards: array-like
                policies: array-like
                values: array-like
        
        sample(batch_size):
            Randomly samples a batch of training data from the buffer.
            Args:
                batch_size (int): Number of samples to retrieve
            Returns:
                tuple: A tuple of (boards, policies, values) as JAX arrays
            Raises:
                ValueError: If batch_size is larger than the buffer size
        __len__():
            Returns the current number of samples in the buffer.
            Returns:
                int: Number of samples currently stored
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
    """
    A move is legal if at least one space in the column is empty.
    Args:
        board
    Returns:
        Boolean array of shape (B, 7) indicating legal moves
    """
    return board[:, 0, :] == 0


def check_win_batched(board):
    """
    Check for winning conditions in batched Connect Four boards using convolution.
    Args:
        board
    Returns:
        Boolean array of shape (B,) indicating if a board has been won.
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
    Apply action to the board state.
    Args:
        state: Current game state
        action: Column indices to drop pieces into (shape: (B,))
    Returns:
        New game state after applying the action
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
    Architecture:
    - Initial convolutional layer (64 filters, 3x3 kernel)
    - 3 residual blocks (each with two 3x3 convolutional layers)
    - Policy head: outputs logits for action probabilities
    - Value head: outputs a scalar value estimate in range [-1, 1]
    Args:
        x: Input tensor of shape (Batch, 6, 7, 2) representing the Connect Four board state.
           The last dimension typically encodes the current player's pieces and opponent's pieces.
    Returns:
        tuple: A pair of (logits, value) where:
            - logits: Action logits of shape (Batch, NUM_ACTIONS) for the policy distribution
            - value: Scalar value estimate of shape (Batch,) representing the expected outcome
                     from the current state, ranging from -1 (loss) to +1 (win)
    """
    def __call__(self, x):
        # Input x is now (Batch, 6, 7, 2)

        x = hk.Conv2D(64, kernel_shape=3)(x)
        x = jax.nn.relu(x)

        for _ in range(3):
            h = hk.Conv2D(64, kernel_shape=3)(x)
            h = jax.nn.relu(h)
            h = hk.Conv2D(64, kernel_shape=3)(h)
            x = jax.nn.relu(x + h)

        p = hk.Conv2D(2, kernel_shape=1)(x)
        p = hk.Flatten()(p)
        logits = hk.Linear(NUM_ACTIONS)(p)

        v = hk.Conv2D(1, kernel_shape=1)(x)
        v = hk.Flatten()(v)
        v = hk.Linear(64)(v)
        v = jax.nn.relu(v)
        value = hk.Linear(1)(v)
        value = jnp.tanh(value)

        return logits, jnp.squeeze(value, axis=-1)

def forward_fn(x): return AlphaZeroNet()(x)
network = hk.without_apply_rng(hk.transform(forward_fn))


@jax.jit
def make_input(state):
    """
    Converts the board into a 2-Channel Image:
    Channel 0: current player's pieces
    Channel 1: opponent's pieces.
    Args:
        state: Current game state
    Returns:
        Input tensor of shape (B, 6, 7, 2) for the NN.
    """
    canonical_board = state.board * state.player[:, None, None]

    my_pieces = (canonical_board == 1.0).astype(jnp.float32)
    opp_pieces = (canonical_board == -1.0).astype(jnp.float32)

    return jnp.stack([my_pieces, opp_pieces], axis=-1)


def root_fn(params, rng_key, embedding):
    """
    Initial inference function for MCTS.
    Args:
        params: Network parameters
        rng_key: JAX random key
        embedding: Current game state
    Returns:
        mctx.RootFnOutput containing prior logits, value, and embedding
    """
    current_state = embedding
    network_input = make_input(current_state)
    logits, value = network.apply(params, network_input)

    legal = get_legal_moves(current_state.board)
    logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)

    return mctx.RootFnOutput(prior_logits=logits, value=value, embedding=embedding)


def recurrent_fn(params, rng_key, action, embedding):
    """
    Recurrent inference function for MCTS.
    Args:
        params: Network parameters
        rng_key: JAX random key
        action: Action taken
        embedding: Current game state
    Returns:
        Tuple of (mctx.RecurrentFnOutput, next_state)
    """
    current_state = embedding
    next_state = step_env(current_state, action)

    reward = jnp.where(next_state.won, 1.0, 0.0)
    discount = jnp.where(next_state.won, 0.0, 1.0)

    network_input = make_input(next_state)
    logits, value = network.apply(params, network_input)

    legal = get_legal_moves(next_state.board)
    logits = jnp.where(legal, logits, jnp.finfo(jnp.float32).min)

    output = mctx.RecurrentFnOutput(
        reward=reward, discount=discount, prior_logits=logits, value=value
    )
    return output, next_state


@jax.jit
def run_mcts(params, rng_key, state):
    """
    MCTS function using the mctx gumbel_muzero_policy.
    Args:
        params: Network parameters
        rng_key: JAX random key
        state: Current game state
    Returns:
        mctx.PolicyOutput containing action weights and visit counts
    """
    root_output = root_fn(params, rng_key, state)
    return mctx.gumbel_muzero_policy(
        params=params,
        rng_key=rng_key,
        root=root_output,
        recurrent_fn=recurrent_fn,
        num_simulations=SIMULATIONS,
        invalid_actions=~get_legal_moves(state.board),
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0
    )


def loss_fn(params, batch):
    """
    Compute combined policy and value loss.
    Args:
        params: Network parameters
        batch: Tuple of (boards, target_policy, target_value)
    Returns:
        Total loss and individual losses as a tuple
    """
    boards, target_policy, target_value = batch

    # Convert boards to 2-channel format.
    my_pieces = (boards == 1.0).astype(jnp.float32)
    opp_pieces = (boards == -1.0).astype(jnp.float32)
    network_input = jnp.stack([my_pieces, opp_pieces], axis=-1)

    logits, value = network.apply(params, network_input)

    log_probs = jax.nn.log_softmax(logits)
    policy_loss = -jnp.mean(jnp.sum(target_policy * log_probs, axis=-1))
    value_loss = jnp.mean((target_value - value) ** 2)
    return policy_loss + value_loss, (policy_loss, value_loss)


@jax.jit
def train_step(params, opt_state, batch):
    """
    Perform a single training step.
    Args:
        params: Network parameters
        opt_state: Optimizer state
        batch: Tuple of (boards, target_policy, target_value)
    Returns:
        Updated parameters, optimizer state, policy loss, value loss"""
    grads, (p_loss, v_loss) = jax.grad(loss_fn, has_aux=True)(params, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, p_loss, v_loss

optimizer = optax.adam(LEARNING_RATE)

def run_self_play_episode(params, key):
    """
    Run a single self-play episode with temperature annealing to fully explore game space.
    TODO: think more about annealing schedule.
    Args:
        params: Network parameters
        key: JAX random key
    Returns:
        Tuple of (boards, policies, values) for training
    """
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

    TEMP_THRESHOLD = 6  # To be adjusted

    for _ in range(ROWS * COLS):
        key, subkey = jax.random.split(key)
        policy_output = run_mcts(params, subkey, state)

        # Exploration exploitation trade-off implementation
        action_logits = jnp.log(policy_output.action_weights + 1e-8)
        sampled_action = jax.random.categorical(subkey, action_logits)
        argmax_action = jnp.argmax(policy_output.action_weights, axis=-1)

        # Annealing mask
        should_explore = state.turn < TEMP_THRESHOLD
        action = jnp.where(should_explore, sampled_action, argmax_action)

        # --- Store History ---
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
    """Make checkpoint directory"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_checkpoint(params, step):
    """Save checkpoints."""
    ensure_dir(CHECKPOINT_DIR)
    filename = os.path.join(CHECKPOINT_DIR, f"step_{step}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(params, f)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(params, f)

def load_params_from_file(filename):
    """Load parameters."""
    print(f"Loading from {filename}...")
    with open(filename, "rb") as f:
        return pickle.load(f)

def select_checkpoint():
    """Pick checkpoints from IO."""
    files = glob.glob(f"{CHECKPOINT_DIR}/*.pkl") + glob.glob("*.pkl")
    files = sorted(list(set(files)))
    if not files: return None, 0

    print("\n--- AVAILABLE MODELS ---")
    for i, f in enumerate(files): print(f"{i+1}. {f}")
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
    """Plot the training history."""
    if not p_hist: return
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.plot(p_hist); plt.title("Policy Loss")
    plt.subplot(1, 2, 2); plt.plot(v_hist, color='orange'); plt.title("Value Loss")
    plt.show()

def print_board_pretty(board):
    """Printing utility for gameplay."""
    print("\n 0 1 2 3 4 5 6")
    print("---------------")
    for r in range(ROWS):
        row_str = "|"
        for c in range(COLS):
            v = board[r, c]
            row_str += ("X" if v==1 else "O" if v==-1 else " ") + "|"
        print(row_str)
    print("---------------\n")

def play_human_vs_ai(params, human_is_player_1=True):
    """
    Play a game against the trained AI.
    Args:
        params: Network parameters
        human_is_player_1: True -> Human is player 1 (X).
    Returns:
        None
    """
    print(f"\n{'='*30}\n HUMAN vs AI\n{'='*30}")
    state = State(
        board=jnp.zeros((1, ROWS, COLS), dtype=jnp.float32),
        player=jnp.ones((1,), dtype=jnp.float32),
        turn=jnp.zeros((1,), dtype=jnp.int32),
        won=jnp.zeros((1,), dtype=jnp.bool_)
    )

    PLAY_SIMULATIONS = 256

    @jax.jit
    def play_mcts(k, s):
        root_output = root_fn(params, k, s)
        return mctx.gumbel_muzero_policy(
            params=params,
            rng_key=k,
            root=root_output,
            recurrent_fn=recurrent_fn,
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
                        action = jnp.array([col]); break
                except: pass
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
            print(f"*** {'HUMAN' if is_human else 'AI'} WINS! ***"); break
        if jnp.all(state.board != 0):
            print_board_pretty(state.board[0]); print("DRAW"); break


if __name__ == "__main__":
    print(f"Running on {jax.devices()[0]}")

    key = jax.random.PRNGKey(42)
    dummy = jnp.zeros((1, ROWS, COLS, 2)) 
    params = network.init(key, dummy)
    opt_state = optimizer.init(params)
    replay_buffer = ReplayBuffer(BUFFER_CAPACITY)

    filename, total_steps = select_checkpoint()
    if filename:
        try:
            params = load_params_from_file(filename)
            opt_state = optimizer.init(params)
            print(f"Loaded '{filename}' (Resuming from step {total_steps})")
        except Exception as e:
            print(f"Load failed: {e}. Starting fresh.")
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
                b, p, v = run_self_play_episode(params, subkey)
                replay_buffer.add(b, p, v)

                if len(replay_buffer) > MIN_BUFFER_SIZE:
                    batch = replay_buffer.sample(BATCH_SIZE)
                    params, opt_state, pl, vl = train_step(params, opt_state, batch)
                    p_hist.append(float(pl)); v_hist.append(float(vl))
                    total_steps += 1
                    if total_steps % CHECKPOINT_INTERVAL == 0:
                        save_checkpoint(params, total_steps)
                        print(f"\nCheckpoint saved at step {total_steps}")
                    print(f"Step {i+1}/{n} (Total {total_steps}) | Buff: {len(replay_buffer)} | P: {pl:.3f} V: {vl:.3f}", end="\r")
                else:
                    print(f"Filling Buffer: {len(replay_buffer)}/{MIN_BUFFER_SIZE}", end="\r")
            print("\nDone.")
            save_checkpoint(params, total_steps)
        elif c == "2": play_human_vs_ai(params, input("Be Player 1 (X)? [y/n]: ").lower()=='y')
        elif c == "3": plot_history(p_hist, v_hist)
        elif c == "4": save_checkpoint(params, total_steps); break
