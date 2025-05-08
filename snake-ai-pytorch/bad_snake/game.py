import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

class Food:
    def __init__(self,pos,kind='good'):
        self.pos = pos
        self.kind = kind 

class MovingObstacle:
    def __init__(self, pos, direction):
        self.pos = pos
        self.dir = direction
    
    def step(self, cols,rows):
        new = Point(self.pos.x + self.dir[0]*BLOCK_SIZE,
                    self.pos.y + self.dir[1]*BLOCK_SIZE)
        #Get them to bounce off walls
        if not (0<=new.x < cols*BLOCK_SIZE and 0<= new.y<rows*BLOCK_SIZE):
            self.dir = Point(-self.dir[0], -self.dir[1])
            new = Point(self.pos.x + self.dir[0]*BLOCK_SIZE,
                        self.pos.y + self.dir[1]*BLOCK_SIZE)
        self.pos = new


pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 40000000

class SnakeGameAI:

    def __init__(self, w=640, h=480,
                num_good=5, num_poison=1,
                num_obstacles=2,num_moving=2, render=True):
        self.w = w
        self.h = h
        self.cols = w // BLOCK_SIZE
        self.rows = h // BLOCK_SIZE
        self.num_good = num_good
        self.num_poison = num_poison
        self.num_obstacles = num_obstacles
        self.num_moving = num_moving
        self.render = render
        
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()


    def reset(self):
        # init game state
        self.last_food_frame = 0
        from collections import deque
        self.last_positions = deque(maxlen=6)

        self.direction = Direction.RIGHT

        self.head = Point(self.w//2, self.h//2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.visited = set()
        self.visited.add(self.head)
        self._place_obstacles()
        self._place_items()
        self.prev_food_dist = float('inf')
        self.frame_iteration = 0
        self.speeds=[]


    def _random_pos(self):
       x = random.randint(0, self.cols-1) * BLOCK_SIZE
       y = random.randint(0, self.rows-1) * BLOCK_SIZE
       return Point(x, y)

    def _place_items(self):
       self.items = []
       for _ in range(self.num_good):
           self.items.append(Food(self._random_pos(), 'good'))
       for _ in range(self.num_poison):
           self.items.append(Food(self._random_pos(), 'poison'))
       self.prev_food_dist = min(
            abs(self.head.x - f.pos.x) + abs(self.head.y - f.pos.y)
            for f in self.items if f.kind=='good'
        )

    def _place_obstacles(self):
       # static obstacles
       self.obstacles = [self._random_pos() for _ in range(self.num_obstacles)]
       # moving obstacles
       dirs = [Point(1,0), Point(-1,0), Point(0,1), Point(0,-1)]
       self.moving_obstacles = [
           MovingObstacle(self._random_pos(), random.choice(dirs))
           for _ in range(self.num_moving)
       ]


    def play_step(self, action):


        # 0. move moving obstacles
        for obs in self.moving_obstacles:
            obs.step(self.cols, self.rows)

        self.frame_iteration += 1

        reward = 0
        game_over = False
        grow = False

        reward  = -0.02
    
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)

        self.last_positions.append(self.head)
        if len(self.last_positions)==6 and len(set(self.last_positions))<3:
            reward = -7  # stuck in a tiny loop of ≤2 spots

        d = min(abs(self.head.x - f.pos.x) + abs(self.head.y - f.pos.y)
                for f in self.items if f.kind=='good')
        reward = 0.5 if d < self.prev_food_dist else -0.2
        self.prev_food_dist = d

        if self.head in self.visited:
            reward = -2
        else:
            self.visited.add(self.head)
        
        # 3. check if game over
       # 3a. food / poison collisions
        for item in self.items:
           if self.head == item.pos:
               if item.kind == 'good':
                    steps_to_food = self.frame_iteration - self.last_food_frame
                    self.speeds.append(steps_to_food)
                    self.last_food_frame = self.frame_iteration
                    reward = 10
                    self.score += 1
                    grow = True
               else:
                   reward = -10
                   game_over = True
               self.items.remove(item)
               self._place_items()
               break



       # 3b. obstacle collisions
        if self.head in self.obstacles or any(obs.pos == self.head for obs in self.moving_obstacles):
           reward = -5
           game_over = True
           return reward, game_over, self.score

        if self.is_collision() or self.frame_iteration > 50*len(self.snake):
            game_over = True
            reward = -5
            return reward, game_over, self.score

        if not grow:
            self.snake.pop()
        
     
        self._update_ui()
        self.clock.tick(SPEED)

        if game_over:
            self.survival_time = self.frame_iteration
            # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

       # draw items
        for item in self.items:
           color = (0,255,0) if item.kind=='good' else (128,0,128)
           pygame.draw.rect(self.display, color,
                            pygame.Rect(item.pos.x, item.pos.y, BLOCK_SIZE, BLOCK_SIZE))
       # draw obstacles
        for x,y in self.obstacles:
           pygame.draw.rect(self.display, (100,100,100),
                            pygame.Rect(x, y, BLOCK_SIZE, BLOCK_SIZE))
        for obs in self.moving_obstacles:
           pygame.draw.rect(self.display, (255,165,0),
                            pygame.Rect(obs.pos.x, obs.pos.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)