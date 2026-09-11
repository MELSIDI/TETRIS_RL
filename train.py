import pygame
from tetris.tetris import Tetris, Action
from tetris.CONSTANTS import WIDTH, HEIGHT, BACKGROUND_COLOR
from tetris.game_ui_render import *
import agent
from torch.utils.tensorboard import SummaryWriter
import datetime


# Path le scpre
BEST_SCORE_PATH = "./best_agent_score.txt"

# Path pour les parametres de l'agent

WEIGHT_PATH = "./agent_weight.pth"

       
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
player = "Agent"
agent_ = agent.Agent(WEIGHT_PATH)
agent_.policy_net.train()   # On active le mode Training

# Game Max_iteration
MAX_ITERATION = 1000

# le board ummary
date_run = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f"runs/tetris_training_{date_run}")
loss_history = []

# Start Gravity Timer
start = pygame.time.get_ticks()
iteration = 0

# Steps 
step = 0

while running and iteration < MAX_ITERATION:
    # Gestions des événements (fermeture de la fenêtre, touches, etc.)
    reward = 0
    gravity_reward = 0
    shaped_reward = 0
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if  env.game_over:
        # Les Entré utilisateur et ou Agent
        env = Tetris()
        iteration += 1
        agent_.exploite_more()
        if len(loss_history) > 0:
            loss_mean = sum(loss_history) / len(loss_history)
            writer.add_scalar("Progression/Mean-Loss", loss_mean, iteration)
        writer.add_scalar("Progression/Score", score, iteration)
        if score > best_score:
            best_score = score
            with open(BEST_SCORE_PATH, "w") as file:
                file.write(str(best_score))
        score = 0
        speed = 1
        loss_history = []
        agent_.update_target_network()
        start = pygame.time.get_ticks()

    windows.fill(BACKGROUND_COLOR)
    # Mise à jour de la vitesse selon l'evolution du score
    new_speed = 1 + score // 500
    speed = new_speed if new_speed <= MAX_SPEED else MAX_SPEED

    # Mise à jour de la matrice
    # Si ya une action soumis
        
    # Action du Joueur
    _, state = env.get_state()
    action = agent_.act(state=state, train_mode=True)
    done = env.game_over
    if action:
        reward, shaped_reward = env.set_state(action=action)
        done = env.game_over
        if reward > 0 :
            score += 10 * (reward + speed)

    total_reward = reward + shaped_reward
    _, next_state = env.get_state()
    agent_.remenber(state, action, total_reward, next_state, done)
    state = next_state
    loss = agent_.replay()
    if loss is not None:
        loss_history.append(loss)
        step +=1
        writer.add_scalar("Progression/Loss", loss, step)

    # Verifier si la gaviter doit s'activer
    # Action de la Grapvité, une action automatiser
    if pygame.time.get_ticks() - start >= 500 - 10 * (speed - 1):
        gravity_reward, shaped_reward = env.set_state(Action.SOFTDROP)
        done = env.game_over
        if gravity_reward > 0:
            score += 10 * (gravity_reward + speed)
        start = pygame.time.get_ticks()

        _, next_state = env.get_state()
        total_reward = gravity_reward + shaped_reward
        agent_.remenber(state, None, total_reward, next_state, done)
        loss = agent_.replay()
        if loss is not None:
            loss_history.append(loss)
            step +=1
            writer.add_scalar("Progression/Loss", loss, step)

    # Mise à jour de l'affichage
    current_tetrimino_matrix = env.current_tetrimino.value[env.current_rotation]
    current_tetrimino_x = env.current_x
    current_tetrimino_y = env.current_y
    next_tetrimino_matrix = env.next_tetrimino.value[0]
    draw_game_screen(windows, score, best_score, speed, player)
    draw_matrix(windows, env.matrix)
    draw_next_tetrimino(windows, next_tetrimino_matrix)
    draw_tetrimino(windows, current_tetrimino_matrix, current_tetrimino_x, current_tetrimino_y)
    

    # Rafraîchir la fenêtre
    pygame.display.flip()

    # Bloquer à 30 FPS
    clock.tick(30)

# Quitter pygame
pygame.quit()

with open(BEST_SCORE_PATH, "w") as file:
    file.write(str(best_score))

agent_.save(WEIGHT_PATH)