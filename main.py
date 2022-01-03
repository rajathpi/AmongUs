import pygame
import neat
import sys
import os
import math
import random
from pygame.locals import *

pygame.init()
clock = pygame.time.Clock()

# Options
WINDOW_SIZE = (1000, 500) # (Width, Height)
DRAW_LINES = False # (Draw lines between the character and blocks to see what the AI sees)

screen = pygame.display.set_mode(WINDOW_SIZE)
display = pygame.Surface(WINDOW_SIZE)

GROUND_LEVEL = WINDOW_SIZE[1]/2 + 75
ground_rect = pygame.Rect(0, GROUND_LEVEL, WINDOW_SIZE[0], WINDOW_SIZE[1])

character_img = pygame.image.load('data/red.png').convert_alpha()
blocks_img = pygame.image.load('data/block.png').convert_alpha()
font = pygame.font.Font('data/roboto.ttf', 25)

generation = 0

class Character():
    def __init__(self, x, y, width, height, img): #img must be a pygame surface object
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.img = pygame.transform.scale(img, (width, height))
        self.rect = pygame.Rect(x, y, width, height)
        self.vertical_momentum = 0
        self.onGround = False
        self.last_closest_pipe = block[0] # Setting the closest blocks to the leftmost blocks by default

    def update(self):
        self.x, self.y = self.rect.x, self.rect.y # Updating position atributes
        self.movement()

    def draw(self):
        display.blit(self.img, (self.x, self.y))

    def jump(self):
        if self.onGround:
            self.vertical_momentum = -11

    def movement(self):
        self.rect.y += self.vertical_momentum

        if self.rect.colliderect(ground_rect):
            self.onGround = True
        else:
            self.onGround = False

        if self.onGround:
            self.rect.bottom = ground_rect.top + 1 # Adding 1 so that the character continues to collide with the rect, instead of shaking up and down
            # Prevent from falling through the ground
            self.vertical_momentum = 0
        else:
            # Add gravity
            self.vertical_momentum += 0.5

        # Cap gravity
        if self.vertical_momentum >= 40:
            self.vertical_momentum = 40

class Blocks():
    def __init__(self, x, y, width, height, img, scroll_speed = 7):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.scroll_speed = scroll_speed
        self.img = pygame.transform.scale(img, (width, height))
        self.rect = pygame.Rect(x, y, width, height)

    def update(self):
        self.x -= self.scroll_speed # Moves blocks to the left
        self.rect.x, self.rect.y = self.x, self.y # Update position atributes

    def draw(self):
        display.blit(self.img, (self.x, self.y))

def get_distance(first_pos, second_pos):
    # Distance formula
    dx = first_pos[0] - second_pos[0]
    dy = first_pos[1] - second_pos[1]
    return math.sqrt(dx**2 + dy**2)

def remove_character(index):
    # 'Kills' the character and its corresponding genome and nn
    characters.pop(index)
    ge.pop(index)
    nets.pop(index)

def draw():
    display.fill('white')

    pygame.draw.line(display, (0, 0, 0), (0, GROUND_LEVEL), (WINDOW_SIZE[0], GROUND_LEVEL), 3)

    for character in characters:
        character.draw()
        if DRAW_LINES:
            pygame.draw.line(
                display, 
                (50, 200, 75), 
                (character.rect.right, character.rect.centery), 
                character.closest_pipe.rect.midtop,
                2
            )
    for blocks in block:
        blocks.draw()

    alive_text = font.render(f'Number alive: {len(characters)}', 1, 'red')
    generation_text = font.render(f'Generation: {generation}', 1, 'red')
    display.blit(alive_text, (10, WINDOW_SIZE[1] - 50))
    display.blit(generation_text, (10, WINDOW_SIZE[1] - 100))

    screen.blit(display, (0, 0))

    pygame.display.update()

def main(genomes, config):
    global block, characters, nets, ge, generation

    block = [Blocks(WINDOW_SIZE[0] + 100, GROUND_LEVEL - 86, 50, 86, blocks_img)]

    characters = []
    nets = []
    ge = []

    scroll_speed = 7
    generation += 1

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        characters.append(Character(100, GROUND_LEVEL-90, 80, 85, character_img))
        g.fitness = 0
        ge.append(g)

    run = True
    while run:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        # Break if all the characters die
        if len(characters) <= 0:
            break

        # Adding new blocks
        if len(block) <= 1:
            if block[0].x < random.randint(300, WINDOW_SIZE[0] - 200) + scroll_speed:
                block.append(Blocks(WINDOW_SIZE[0] + 100, GROUND_LEVEL - 86, 50, 86, blocks_img, scroll_speed))

        for blocks in block:
            blocks.update()
            if blocks.x < -100:
                block.remove(blocks)
            for i, character in enumerate(characters):
                if character.rect.colliderect(blocks.rect):
                    ge[i].fitness -= 3
                    remove_character(i)

        for i, character in enumerate(characters):
            character.update()
            # Check if the character passed a blocks
            # Getting the closest blocks by finding the leftmost blocks that is to the right of the character
            character.closest_pipe = [blocks for blocks in block if blocks.rect.x > character.x - character.width][0]
            # Checking if the character passed a blocks by comparing it to the closest blocks in the last frame and seeing if there is a change
            if character.closest_pipe != character.last_closest_pipe:
                ge[i].fitness += 1
                for blocks in block:
                    # Increace speed everytime a blocks is passed
                    blocks.scroll_speed += 0.05
                    scroll_speed += 0.05 
            character.last_closest_pipe = character.closest_pipe

            # Giving all dinsoaurs a little fitness for staying alive
            ge[i].fitness += 0.05

            output = nets[i].activate(
                (
                    character.y,
                    get_distance((character.x, character.y), character.closest_pipe.rect.midtop)
                )
            )

            if output[0] > 0.5:
                character.jump()

        draw()

# Setup the NEAT nn
def run(config_path):
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    winner = p.run(main, 50)

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)
