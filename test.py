import pygame
import random
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake Game")
