import pygame
from tetris import Tetris, Action
from CONSTANTS import WIDTH, HEIGHT, BACKGROUND_COLOR
from game_ui_render import *



BEST_SCORE_PATH = "./best_score.txt"

       
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

# Le Score
score = 0
speed = 1

# Lecture du meilleur score
best_score = 0
try:
    with open(BEST_SCORE_PATH, 'r') as file:
        best_score = int(file.readline())
except (FileNotFoundError, ValueError):
    pass

# Le Joueur
player = "Human"

# Start Gravity Timer
start = pygame.time.get_ticks()

while running:
    # Gestions des événements (fermeture de la fenêtre, touches, etc.)
    reward = 0
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if not env.game_over:
                # Les Entré utilisateur et ou Agent
                if event.key == pygame.K_RIGHT:
                    action = Action.RIGHT
                elif event.key == pygame.K_LEFT:
                    action = Action.LEFT
                elif event.key == pygame.K_DOWN:
                    action = Action.SOFTDROP
                elif event.key == pygame.K_UP:
                    action = Action.HARDDROP
                elif event.key == pygame.K_SPACE:
                    action = Action.ROTATE
            else:
                if event.key == pygame.K_RETURN:
                    env = Tetris()
                    if score > best_score:
                        best_score = score
                        with open(BEST_SCORE_PATH, "w") as file:
                            file.write(str(best_score))
                    score = 0
                    speed = 1
                    start = pygame.time.get_ticks()

    windows.fill(BACKGROUND_COLOR)
    if not env.game_over:
        # Mise à jour de la vitesse selon l'evolution du score
        new_speed = 1 + score // 500
        speed = new_speed if new_speed <= MAX_SPEED else MAX_SPEED

        # Mise à jour de la matrice
        # Si ya une action soumis
        
        # Action du Joueur
        if action:
            reward = env.set_state(action=action)
            if reward > 0 :
                score += 2 * (reward + speed)

        # Verifier si la gaviter doit s'activer
        # Action de la Grapvité, une action automatiser
        if pygame.time.get_ticks() - start >= 500 - 10 * (speed - 1):
                gravity_reward = env.set_state(Action.SOFTDROP)
                if gravity_reward > 0:
                    score += 2 * (gravity_reward + speed)
                start = pygame.time.get_ticks()

        # Mise à jour de l'affichage
        current_tetrimino_matrix = env.current_tetrimino.value[env.current_rotation]
        current_tetrimino_x = env.current_x
        current_tetrimino_y = env.current_y
        next_tetrimino_matrix = env.next_tetrimino.value[0]
        draw_game_screen(windows, score, best_score, speed, player)
        draw_matrix(windows, env.matrix)
        draw_next_tetrimino(windows, next_tetrimino_matrix)
        draw_tetrimino(windows, current_tetrimino_matrix, current_tetrimino_x, current_tetrimino_y)
    else:
        draw_game_over_screen(windows, score, best_score, speed, player)

    # Rafraîchir la fenêtre
    pygame.display.flip()

    # Bloquer à 30 FPS
    clock.tick(30)

# Quitter pygame
pygame.quit()