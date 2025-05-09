import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import time

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


# font = pygame.font.SysFont('arial', 25)



class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (0, 200, 0)
YELLOW = (255,255,0)

BLOCK_SIZE = 20
SPEED = 40000000

class MovingObstacle:
    """A single obstacle that hops one block per frame and bounces off walls."""
    def __init__(self, pos: Point):
        self.pos = pos
        # pick a random cardinal direction as a (dx, dy) tuple
        self.dir = random.choice([(1,0),(-1,0),(0,1),(0,-1)])

    def step(self, w: int, h: int):
        # compute the new position
        new = Point(self.pos.x + self.dir[0]*BLOCK_SIZE,
                    self.pos.y + self.dir[1]*BLOCK_SIZE)
        # if it would leave the screen, reverse direction
        if not (0 <= new.x < w and 0 <= new.y < h):
            self.dir = (-self.dir[0], -self.dir[1])
            new = Point(self.pos.x + self.dir[0]*BLOCK_SIZE,
                        self.pos.y + self.dir[1]*BLOCK_SIZE)
        self.pos = new
 


class SnakeGameAI:

    def __init__(self, w=640, h=480, num_poison=3, num_obstacles=3):
        self.w = w
        self.h = h
        self.num_poison = num_poison  # holds the num of poison apples
        self.num_obstacles = num_obstacles

        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self):

        self.last_food_frame = 0 # - R
        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.poison = []  # initializes poison points as an empty list
        self.moving_obstacles = []  # initializes poison points as an empty list
        self.food = None
        self._place_food()
        for poison in range(self.num_poison):   # places each poison apple down
            self._place_poison()
        for x in range(self.num_obstacles):   # places each poison apple down
            self._place_moving_obstacle()
        self.frame_iteration = 0

        self.speeds = [] # - R
        self.start_time = time.time() # starts timer - R
        self.elapsed_time = 0 # initialize timer - R

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()
        for point in self.poison:  # makes sure food doesn't spawn on poison
            if self.food == point:
                self._place_food()

    def _place_poison(self):
        # gets random location on map
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.poison.append(Point(x,y))  # adds point to collection of poison points
        if self.poison[:-1] in self.snake:  # makes sure that it doesn't spawn on snake
            self.poison.pop(-1)  # removes point if it does
            self._place_poison()  # reruns function
        if self.poison[:-1] == self.food:  # makes sure it doesn't spawn on food
            self.poison.pop(-1)
            self._place_poison()
        for point in self.poison:  # makes sure poison doesn't spawn on other poison
            if self.food == point:
                self.poison.pop(-1)
                self._place_food()

    def _place_moving_obstacle(self):
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        pt = Point(x,y)
        # avoid snake, food, poison, other moving obstacles
        if (pt in self.snake or pt == self.food 
            or pt in self.poison 
            or any(obs.pos == pt for obs in self.moving_obstacles)):
            return self._place_moving_obstacle()
        self.moving_obstacles.append(MovingObstacle(pt))


    def play_step(self, action):


                # 0) **move the obstacles first**  
        for obs in self.moving_obstacles:
            obs.step(self.w, self.h)

        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score, self.elapsed_time

        # 4. place new food or just move
        if self.head == self.food:
            steps_to_food = self.frame_iteration - self.last_food_frame
            self.speeds.append(steps_to_food)
            self.last_food_frame = self.frame_iteration

            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score

        # - R edits
        self.survival_time = self.frame_iteration
        self.elapsed_time = time.time() - self.start_time

        return reward, game_over, self.score, self.elapsed_time

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        # hits poison
        for point in self.poison:
            if pt == point:
                return True
        if any(pt == obs.pos for obs in self.moving_obstacles):
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        # draws snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))

        # draws food
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # draws poison
        for point in self.poison:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))

        for obs in self.moving_obstacles:
                pygame.draw.rect(self.display, YELLOW,
                                pygame.Rect(obs.pos.x, obs.pos.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # right turn r -> d -> l -> u
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # left turn r -> u -> l -> d

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