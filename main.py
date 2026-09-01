import random
import time
import pygame
from tetris import Tetris, Action



BACKGROUND_COLOR = (0, 0, 0)
CELL_COLOR = (0, 255, 0)
WIDTH, HEIGHT = 800, 750
CELL_WIDTH, CELL_HEIGHT = 30, 30
MATRIX_WIDTH, MATRIX_HEIGHT = 300, 600
X, Y = 100, 50



def draw_cell(x, y):
    pygame.draw.rect(windows, CELL_COLOR, (x, y, CELL_WIDTH, CELL_HEIGHT), 3)
    pygame.draw.circle(windows, CELL_COLOR, (x + CELL_WIDTH // 2, y + CELL_HEIGHT // 2), 3)


def draw_empty(x, y):
    pygame.draw.circle(windows, CELL_COLOR, (x, y), 3)
    pygame.draw.circle(windows, CELL_COLOR, (x + CELL_WIDTH, y), 3)
    pygame.draw.circle(windows, CELL_COLOR, (x, y + CELL_HEIGHT), 3)
    pygame.draw.circle(windows, CELL_COLOR, (x + CELL_WIDTH, y + CELL_HEIGHT), 3)


def draw_matrix(matrix):
    pygame.draw.rect(windows, CELL_COLOR, (X - 8, Y - 8, MATRIX_WIDTH + 16 , MATRIX_HEIGHT + 16), 2, border_radius=4)
    for y in range(20):
        for x in range(10):
            if matrix[y][x] == 0:
                draw_empty(X + x * CELL_WIDTH, Y + y * CELL_HEIGHT)
            elif matrix[y][x] == 1:
                draw_cell(X + x * CELL_WIDTH, Y + y * CELL_HEIGHT)
                

def draw_tetrimino(tetrimino_matrix, tetrimino_x, tetrimino_y):
    for x in range(4):
        for y in range(4):
            if tetrimino_matrix[y][x] == 1:
                draw_cell(X + (tetrimino_x + x) * CELL_WIDTH, Y + (tetrimino_y + y) * CELL_HEIGHT)
    

# 1. Initialser Pygame
pygame.init()

# 2. Setup de la fenêtre
windows = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TETRIS")

# Setup de l'env
env = Tetris()

# Créer une horloge pour contrôler les FPS
clock = pygame.time.Clock()

# Boucle principal du jeu
running = True

# Start Gravity Timer
start = pygame.time.get_ticks()

while running:
    # Gestions des événements (fermeture de la fenêtre, touches, etc.)
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            # Les Entré utilisateur et ou Agent
            if event.key == pygame.K_RIGHT:
                action = Action.RIGHT
            elif event.key == pygame.K_LEFT:
                action = Action.LEFT
            elif event.key == pygame.K_DOWN:
                action = Action.DOWN
            elif event.key == pygame.K_UP:
                action = Action.ROTATE

    # Mise à jour de la matrice
    # Si ya une action soumis
    if action:
        env.set_state(action=action)

    # Verifier si la gaviter doit s'activer
    if pygame.time.get_ticks() - start >= 300:
            env.set_state(Action.DOWN)
            start = pygame.time.get_ticks()

    # Mise à jour de l'affichage
    current_tetrimino_matrix = env.current_tetrimino.value[env.current_rotation]
    current_tetrimino_x = env.current_x
    current_tetrimino_y = env.current_y
    windows.fill(BACKGROUND_COLOR)
    draw_matrix(env.matrix)
    draw_tetrimino(current_tetrimino_matrix, current_tetrimino_x, current_tetrimino_y)

    # Rafraîchir la fenêtre
    pygame.display.flip()

    # Bloquer à 30 FPS
    clock.tick(30)

# Quitter pygame
pygame.quit()