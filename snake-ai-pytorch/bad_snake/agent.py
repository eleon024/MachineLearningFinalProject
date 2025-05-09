import os
import random
import numpy as np
import torch
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot
from checkpoint_utils import save_checkpoint, load_checkpoint

MAX_MEMORY = 100_000   # Maximum memory size for experience replay
BATCH_SIZE = 64      # Mini-batch size for training from memory
LR = 0.001             # Learning rate for the Q-network

class Agent:
    """
    Agent that interacts with the Snake game using a DQN approach. 
    It stores experiences, decides actions using an epsilon-greedy strategy, and trains a neural network to approximate Q-values.
    """
    def __init__(self, game: SnakeGameAI):
        self.n_games = 0
        self.epsilon = 0   # exploration rate (will decay with more games played)
        self.gamma   = 0.9 # discount rate for future rewards
        self.memory  = deque(maxlen=MAX_MEMORY)  # replay memory

        # Determine the state representation size based on game grid
        # The state is a grid of size (rows x cols) with 6 channels:
        # 0 = snake body, 1 = good food, 2 = poison, 3 = static obstacle, 4 = moving obstacle, 5 = snake head.
        self.C = 6 + 4 
        self.H = game.rows
        self.W = game.cols
        # how big is our flattened state?
        self.state_size = self.C * self.H * self.W   # e.g. 10 * rows * cols

        # choose a hidden layer size (you can tune this)
        hidden_size = 256

        # we have 3 possible actions: [left, straight, right]
        output_size = 3  

        self.model   = Linear_QNet(
            input_size=self.state_size,
            hidden_size=hidden_size,
            output_size=output_size
        )

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)



    def get_state(self, game: SnakeGameAI) -> np.ndarray:
        # now 10 channels: 
        # 0=body, 1=good food, 2=poison, 3=static obs, 4=moving obs, 5=head,
        # 6=dir_right, 7=dir_left, 8=dir_up, 9=dir_down
        grid = np.zeros((game.rows, game.cols, self.C), dtype=int)

        # --- existing code to fill channels 0–5 ---
        for pt in game.snake[1:]:
            gx, gy = pt.x // BLOCK_SIZE, pt.y // BLOCK_SIZE
            grid[gy, gx, 0] = 1

        for item in game.items:
            gx, gy = item.pos.x // BLOCK_SIZE, item.pos.y // BLOCK_SIZE
            grid[gy, gx, 1 if item.kind=='good' else 2] = 1

        for pt in game.obstacles:
            gx, gy = pt.x // BLOCK_SIZE, pt.y // BLOCK_SIZE
            grid[gy, gx, 3] = 1

        for obs in game.moving_obstacles:
            gx, gy = obs.pos.x // BLOCK_SIZE, obs.pos.y // BLOCK_SIZE
            grid[gy, gx, 4] = 1

        hx, hy = game.head.x // BLOCK_SIZE, game.head.y // BLOCK_SIZE
        grid[hy, hx, 5] = 1

        # --- new: one-hot direction planes ---
        if game.direction == Direction.RIGHT:
            grid[:, :, 6] = 1
        elif game.direction == Direction.LEFT:
            grid[:, :, 7] = 1
        elif game.direction == Direction.UP:
            grid[:, :, 8] = 1
        elif game.direction == Direction.DOWN:
            grid[:, :, 9] = 1

        return grid.flatten()


    def remember(self, state, action, reward, next_state, done):
        """Store a single experience (transition) in replay memory."""
        self.memory.append((state, action, reward, next_state, done))
        # If MAX_MEMORY is reached, older experiences are automatically discarded (deque behavior).

    def train_long_memory(self):
        """
        Train the Q-network on a batch of past experiences (experience replay).
        If there are enough samples, sample a random batch; otherwise use all available experiences.
        """
        if len(self.memory) < 500:
            return
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of BATCH_SIZE random experiences
        else:
            mini_sample = list(self.memory)
        # Transpose the batch (list of tuples) into tuple of lists for states, actions, etc.
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Train the Q-network on a single step (state transition). 
        This is done during the game to quickly adjust to recent changes.
        """
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state, saliency: bool = False):
        """
        Decide on an action given the current state using an epsilon-greedy strategy.
        Returns a one-hot encoded action (as a list of length 3) where:
        index 0 = turn left, 1 = go straight, 2 = turn right (relative to current direction).
        If saliency=True, computes gradients to indicate importance of input channels (for debugging).
        """
        # Decrease exploration rate as games progress (simple linear decay)
        final_move = [0, 0, 0]
        eps_start, eps_end, eps_decay = 1.0, 0.05, 200
        self.epsilon = 80 - self.n_games
        if random.random() < self.epsilon:
            move_idx = random.randint(0,2)
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            state0 = state0.unsqueeze(0)            # -> [1, state_size]
            # or equivalently: state0 = state0.view(1, -1)
            q_values = self.model(state0)

            # Choose the action with the maximum Q-value
            _, move_idx = q_values.max(dim=1)
            move_idx = move_idx.item()
            # Optionally compute saliency (sensitivity of Q-max to input) if requested
            if saliency:
                q_max = q_values.max()  # maximum Q-value for this state
                # Compute gradients of q_max w.r.t. the input state
                state0.requires_grad_(True)
                q_max.backward(retain_graph=True)
                grads = state0.grad.data.abs()
                # Compute average absolute gradient per channel as a simple saliency measure
                channel_importance = grads.view(self.C, -1).mean(dim=1)
                print(f"Saliency (avg |grad| per channel): {channel_importance.tolist()}")

        # Set the chosen move in one-hot format
        final_move[move_idx] = 1
        return final_move

# The training loop can be executed when running this file directly.
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    # Initialize game with desired configuration (grid size and number of objects)
    game = SnakeGameAI(w=480, h=480, num_good=1, num_poison=0, num_obstacles=0, num_moving=0)
    agent = Agent(game)

    # Optionally load existing model checkpoint
    if load_checkpoint(agent):
        print(f"Resumed training at game {agent.n_games}, with epsilon = {agent.epsilon:.2f}")

    # Main training loop
    while True:
        # 1. Get current state
        state_old = agent.get_state(game)

        # 2. Get action based on current state
        final_move = agent.get_action(state_old)  # one-hot encoded action

        # 3. Perform the action in the game and get the reward and new state
        reward, done, score = game.play_step(final_move)
        # Get new state if game continues, otherwise use a zero-state if game is over
        if not done:
            state_new = agent.get_state(game)
        else:
            state_new = np.zeros(agent.state_size, dtype=int)
        
        # # # 4. Train the agent with the transition (short-term learning)
        if len(agent.memory) < BATCH_SIZE:
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
        # Remember the transition for long-term replay
        agent.remember(state_old, final_move, reward, state_new, done)

        # 5. If the game is over, prepare for the next round
        if done:
            # Reset game and tally results
            game.reset()
            agent.n_games += 1  # increment game count
            # Train long-term memory (experience replay) after each game
            agent.train_long_memory()
            # Update record high score and save model if improved
            if score > record:
                record = score
                agent.model.save()
            # Save periodic checkpoints of the model and agent state every 25 games
            if agent.n_games % 25 == 0:
                save_checkpoint(agent)
                print(f"Checkpoint saved at game {agent.n_games}")
            # Print game results
            avg_last50 = np.mean(plot_scores[-50:])
            print(f"Game {agent.n_games}, score={score}, avg50={avg_last50:.2f}, ε={agent.epsilon:.2f}")

            # Update score plot
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == "__main__":
    train()
