import os, torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot
from checkpoint_utils import save_checkpoint, load_checkpoint


MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

 def __init__(self,game:SnakeGameAI):
        self.n_games  = 0
        self.epsilon  = 0
        self.gamma    = 0.9
        self.memory   = deque(maxlen=MAX_MEMORY)

        # compute flattened grid size (rows * cols * channels)
        C = 6  # e.g. [body, good food, poison, static obs, moving obs, head]
        self.state_size = game.rows * game.cols * C

        # now instantiate with the correct input size
        self.model   = Linear_QNet(self.state_size, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    # def get_state(self, game):
    #     head = game.snake[0]
    #     point_l = Point(head.x - 20, head.y)
    #     point_r = Point(head.x + 20, head.y)
    #     point_u = Point(head.x, head.y - 20)
    #     point_d = Point(head.x, head.y + 20)
        
    #     dir_l = game.direction == Direction.LEFT
    #     dir_r = game.direction == Direction.RIGHT
    #     dir_u = game.direction == Direction.UP
    #     dir_d = game.direction == Direction.DOWN

    #     state = [
    #         # Danger straight
    #         (dir_r and game.is_collision(point_r)) or 
    #         (dir_l and game.is_collision(point_l)) or 
    #         (dir_u and game.is_collision(point_u)) or 
    #         (dir_d and game.is_collision(point_d)),

    #         # Danger right
    #         (dir_u and game.is_collision(point_r)) or 
    #         (dir_d and game.is_collision(point_l)) or 
    #         (dir_l and game.is_collision(point_u)) or 
    #         (dir_r and game.is_collision(point_d)),

    #         # Danger left
    #         (dir_d and game.is_collision(point_r)) or 
    #         (dir_u and game.is_collision(point_l)) or 
    #         (dir_r and game.is_collision(point_u)) or 
    #         (dir_l and game.is_collision(point_d)),
            
    #         # Move direction
    #         dir_l,
    #         dir_r,
    #         dir_u,
    #         dir_d,
            
    #         # Food location 
    #         game.food.x < game.head.x,  # food left
    #         game.food.x > game.head.x,  # food right
    #         game.food.y < game.head.y,  # food up
    #         game.food.y > game.head.y  # food down
    #         ]

    #     return np.array(state, dtype=int)

 def get_state(self, game: SnakeGameAI):
     # channels: 0=body,1=good,2=poison,3=static obs,4=moving obs,6=head
     grid = np.zeros((game.rows, game.cols, 6), dtype=int)
    
    
     # snake body
     for pt in game.snake[1:]:
         gx, gy = pt.x//BLOCK_SIZE, pt.y//BLOCK_SIZE
         grid[gy, gx, 0] = 1
     
     # items
     for item in game.items:
         gx, gy = item.pos.x//BLOCK_SIZE, item.pos.y//BLOCK_SIZE
         grid[gy, gx, 1 if item.kind=='good' else 2] = 1
     
     
     # static obstacles
     for pt in game.obstacles:
         gx, gy = pt.x//BLOCK_SIZE, pt.y//BLOCK_SIZE
         grid[gy, gx, 3] = 1
     # moving obstacles
     for obs in game.moving_obstacles:
         gx, gy = obs.pos.x//BLOCK_SIZE, obs.pos.y//BLOCK_SIZE
         grid[gy, gx, 4] = 1
     
     
     # head
     hx, hy = game.head.x//BLOCK_SIZE, game.head.y//BLOCK_SIZE
     grid[hy, hx, 5] = 1
     return grid.flatten()



 def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

 def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

 def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

 def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    speeds_per_game = []
    survival_per_game = []
    total_score = 0
    record = 0
    game = SnakeGameAI(w=640, h=640, num_good=3, num_poison=1, num_obstacles=10, num_moving=4)
    agent = Agent(game)

    if load_checkpoint(agent):
        print(f"Resumed at game {agent.n_games}, ε={agent.epsilon:.2f}")

        
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        if not done:
            state_new = agent.get_state(game)
        else:
            state_new = np.zeros(agent.state_size,dtype=int)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            
            if game.speeds:
                avg_speed = sum(game.speeds)/len(game.speeds)
            else:
                avg_speed = float('nan')
            speeds_per_game.append(avg_speed)


            
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            save_checkpoint(agent)
            print(f"Checkpoint saved at game {agent.n_games}")
            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)



if __name__ == '__main__':
    train()