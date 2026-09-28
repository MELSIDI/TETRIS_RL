import pygame
from collections import deque
from tetris.tetris import Tetris, Action
from tetris.CONSTANTS import WIDTH, HEIGHT, BACKGROUND_COLOR, MAX_SPEED
from tetris.game_ui_render import *
import agent
from agent import build_input
from torch.utils.tensorboard import SummaryWriter
import datetime

BEST_SCORE_PATH = "./best_agent_score.txt"
WEIGHT_PATH = "./agent_weight.pth"

MAX_ITERATION = 5000
BASE_GRAVITY_MS = 500
GRAVITY_STEP_MS = 10


pygame.init()
windows = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TETRIS")
clock = pygame.time.Clock()

env = Tetris()
running = True
render_mode = False
score = 0
speed = 1

best_score = 0
try:
    with open(BEST_SCORE_PATH, "r") as file:
        best_score = int(file.readline())
except (FileNotFoundError, ValueError):
    pass

agent_ = agent.Agent(WEIGHT_PATH)
agent_.policy_net.train()

date_run = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f"runs/tetris_training_{date_run}")
loss_history = []
iteration = 0
step = 0

# ------------------------------------------------------------
# État du Macro Agent
# ------------------------------------------------------------
current_plan = None
current_path = deque()

# Features du plateau AVANT le placement de la pièce en cours
# (= features de l'afterstate précédent : c'est la chaîne TD(0)).
current_board_input = build_input(env.matrix, env.next_tetrimino)

# Récompense cumulée jusqu'au verrouillage de la pièce.
decision_reward = 0.0


def reset_macro_state():
    global current_plan, current_path, decision_reward

    current_plan = None
    current_path.clear()
    decision_reward = 0.0


def create_macro_plan():
    """
    Calcule les afterstates atteignables depuis la position réelle
    actuelle, puis demande à l'Agent Macro d'en choisir un.

    Le chemin obtenu est un plan temporaire : la gravité peut
    l'invalider et provoquer une nouvelle planification (le plateau
    "avant" ne change pas pour autant, seul le chemin est recalculé).
    """
    global current_plan, current_path, decision_reward

    final_states = env.find_final_states()
    next_tetrimino = env.next_tetrimino

    if not final_states:
        current_plan = None
        current_path.clear()
        return False

    current_plan = agent_.act(final_states=final_states, next_tetrimino=next_tetrimino, train_mode=True)
    current_path = deque(current_plan["path"])
    decision_reward = 0.0

    return True


def finish_macro_transition(done):
    """
    Enregistre une transition RL uniquement lorsque la pièce est
    réellement verrouillée. Utilise l'état RÉEL du plateau après
    verrouillage (env.matrix) plutôt que la prédiction du plan, pour
    rester correct même si clear_lines() a fait tomber des amas
    flottants différemment de ce qui était prévu.
    """
    global current_board_input


    next_tetrimino_after = None if done else env.next_tetrimino
    next_board_input = build_input(env.matrix, next_tetrimino_after) if not done else build_input(env.matrix, env.current_tetrimino)

    agent_.remember(
        state_features=current_board_input,
        next_state_features=next_board_input,
        reward=decision_reward,
        done=done
    )

    current_board_input = next_board_input


def apply_action(action):
    """
    Applique une action sur l'env et détecte de façon fiable si elle a
    verrouillé la pièce, via le compteur env.pieces_placed (PAS via
    reward > 0 : la plupart des locks ne complètent aucune ligne et
    auraient un reward de 0, ce qui faisait rater la détection).

    Nécessite d'avoir ajouté `self.pieces_placed = 0` dans
    Tetris.__init__ et `self.pieces_placed += 1` dans le bloc de
    verrouillage de set_state().
    """
    pieces_before = env.pieces_placed
    reward, shaped_reward = env.set_state(action)
    locked = env.pieces_placed != pieces_before
    done = env.game_over
    return reward, shaped_reward, locked, done


def train_step():
    global step

    loss = agent_.replay()

    if loss is not None:
        loss_history.append(loss)
        step += 1
        writer.add_scalar("Progression/Loss", loss, step)

    return loss


gravity_start = pygame.time.get_ticks()


while running and iteration < MAX_ITERATION:

    # --------------------------------------------------------
    # Événements Pygame
    # --------------------------------------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_v:
                render_mode = not render_mode

    # --------------------------------------------------------
    # Nouvelle partie
    # --------------------------------------------------------
    if env.game_over:
        env = Tetris()
        iteration += 1

        agent_.exploite_more()

        if loss_history:
            writer.add_scalar(
                "Progression/Mean-Loss",
                sum(loss_history) / len(loss_history),
                iteration
            )

        writer.add_scalar("Progression/Score", score, iteration)

        if score > best_score:
            best_score = score
            with open(BEST_SCORE_PATH, "w") as file:
                file.write(str(best_score))

        score = 0
        speed = 1
        loss_history = []

        gravity_start = pygame.time.get_ticks()
        reset_macro_state()
        current_board_input = build_input(env.matrix, env.next_tetrimino)

    # --------------------------------------------------------
    # Vitesse / intervalle de gravité
    # --------------------------------------------------------
    speed = min(1 + score // 500, MAX_SPEED)
    gravity_interval = max(
        50,
        BASE_GRAVITY_MS - GRAVITY_STEP_MS * (speed - 1)
    )

    # --------------------------------------------------------
    # PLANIFICATION MACRO
    # --------------------------------------------------------
    # Si le chemin est vide, soit le plan vient de se terminer,
    # soit il vient d'être invalidé par la gravité.
    if not current_path and not env.game_over:
        create_macro_plan()

    # --------------------------------------------------------
    # EXÉCUTION D'UNE ACTION DU PLAN
    # --------------------------------------------------------
    if current_path and not env.game_over:
        action = current_path.popleft()

        reward, shaped_reward, locked, done = apply_action(action)
        decision_reward += reward + shaped_reward

        if reward > 0:
            score += 10 * (reward + speed)

        # Le plan se termine normalement par un HARDDROP, qui verrouille
        # toujours la pièce -> locked sera True ici.
        if locked:
            finish_macro_transition(done)
            reset_macro_state()

            gravity_start = pygame.time.get_ticks()
            train_step()

    # --------------------------------------------------------
    # GRAVITÉ EXTERNE
    # --------------------------------------------------------
    now = pygame.time.get_ticks()

    if now - gravity_start >= gravity_interval and not env.game_over:
        gravity_reward, gravity_shaped_reward, locked, done = apply_action(
            Action.SOFTDROP
        )

        decision_reward += gravity_reward + gravity_shaped_reward

        if gravity_reward > 0:
            score += 10 * (gravity_reward + speed)

        gravity_start = now

        # SOFTDROP a verrouillé la pièce (détecté via pieces_placed, PAS
        # via gravity_reward > 0 : la plupart des locks ne complètent
        # aucune ligne et auraient été manqués par cette ancienne
        # condition).
        #
        # Dans set_state(), un SOFTDROP qui rencontre le sol :
        #   - fige la pièce,
        #   - clear les lignes,
        #   - charge la prochaine pièce,
        #   - réinitialise sa position.
        #
        # Donc on termine la transition macro.
        if locked:
            finish_macro_transition(done)
            reset_macro_state()
            train_step()

        else:
            # La pièce a été déplacée par la gravité mais reste active.
            # Le plan calculé avant la chute n'est plus notre plan de
            # référence (le "state avant" ne change pas, seul le chemin
            # doit être recalculé) : on invalide le plan et on
            # recalculera depuis la nouvelle position réelle.
            current_plan = None
            current_path.clear()

    # --------------------------------------------------------
    # AFFICHAGE
    # --------------------------------------------------------
    if render_mode:
        windows.fill(BACKGROUND_COLOR)

        current_tetrimino_matrix = (
            env.current_tetrimino.value[env.current_rotation]
        )

        draw_game_screen(
            windows, score, best_score, speed, "Agent"
        )
        draw_matrix(windows, env.matrix)
        draw_next_tetrimino(
            windows, env.next_tetrimino.value[0]
        )
        draw_tetrimino(
            windows,
            current_tetrimino_matrix,
            env.current_x,
            env.current_y
        )

        pygame.display.flip()
        clock.tick(30)
    else:
        clock.tick(120)


pygame.quit()

with open(BEST_SCORE_PATH, "w") as file:
    file.write(str(best_score))

agent_.save(WEIGHT_PATH)
writer.close()