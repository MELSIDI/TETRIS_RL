import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from tetris.tetrimino import Tetrimino


# Si GPU disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TETRIMINO_LIST = list(Tetrimino)
NUM_TETRIMINO = len(TETRIMINO_LIST)


def _tetrimino_to_onehot(tetrimino):
    vec = [0.0] * NUM_TETRIMINO
    vec[TETRIMINO_LIST.index(tetrimino)] = 1.0
    return vec


def extract_features(grid):
    """
    Calcule les 4 features classiques utilisées pour évaluer un afterstate
    Tetris : lignes complètes, trous, bumpiness (irrégularité des hauteurs
    entre colonnes adjacentes), hauteur agrégée.

    grid : grille 20x10 (list of lists de 0/1), un afterstate déjà posé
           (ou n'importe quel état de plateau figé).

    Retourne une liste de 4 floats.
    """
    height = len(grid)
    width = len(grid[0])

    # --- Hauteurs de colonnes ---
    heights = []
    for col in range(width):
        col_height = 0
        for row in range(height):
            if grid[row][col] == 1:
                col_height = height - row
                break
        heights.append(col_height)

    aggregate_height = sum(heights)

    # --- Bumpiness ---
    bumpiness = sum(abs(heights[i] - heights[i + 1]) for i in range(width - 1))

    # --- Trous (case vide sous au moins un bloc plein dans la même colonne) ---
    holes = 0
    for col in range(width):
        block_found = False
        for row in range(height):
            if grid[row][col] == 1:
                block_found = True
            elif block_found and grid[row][col] == 0:
                holes += 1

    # --- Lignes complètes ---
    lines_cleared = sum(1 for row in grid if all(cell == 1 for cell in row))

    return [float(lines_cleared), float(holes), float(bumpiness), float(aggregate_height)]


def build_input(grid, next_tetrimino):
    """
        Concatène les 4 features du board + le one-hot de la pièce suivante.
    """
    return extract_features(grid) + _tetrimino_to_onehot(next_tetrimino)


class ValueNetwork(nn.Module):
    """
    Petit MLP qui évalue un afterstate à partir de ses 4 features
    (lines_cleared, holes, bumpiness, aggregate_height).
    Sort une seule valeur scalaire V(features).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(4 + NUM_TETRIMINO, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, features):
        x = torch.relu(self.fc1(features))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class Agent:
    """
    Agent qui choisit, pour chaque pièce, un état final (afterstate) parmi
    ceux générés par env.find_final_states(), évalué via ses 4 features.

    Apprentissage en TD(0) sur la chaîne des afterstates réellement
    choisis (pas de max recalculé pendant replay(), pas de target_net) :

        V(état_avant) <- reward + gamma * V(état_choisi_suivant)
    """

    def __init__(self, weight_path):
        # Hyperparamètres
        self.gamma = 0.99            # Importance du futur
        self.epsilon = 1.0           # Taux d'exploration, 100% au début
        self.epsilon_min = 1e-3      # Taux minimal d'exploration
        self.epsilon_decay = 0.997   # Taux de réduction de l'exploration
        self.batch_size = 512        # Nombre de souvenirs étudiés par boucle

        # Mémoire
        self.memory = deque(maxlen=30000)

        # Initialisation du réseau (un seul, plus de target_net)
        self.policy_net = self.load(weight_path)
        self.policy_net.to(device=device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def act(self, final_states, next_tetrimino, train_mode=True):
        """
        Choisit un état final parmi les candidats.

        final_states : liste retournée par env.find_final_states() pour
                        la pièce actuelle.

        Retourne le dict candidat choisi (contient "state" et "path").
        """
        if train_mode and random.random() <= self.epsilon:
            return random.choice(final_states)

        with torch.no_grad():
            feats = torch.FloatTensor(
                [build_input(final_state["state"], next_tetrimino) for final_state in final_states]
            ).to(device=device)
            values = self.policy_net(feats).squeeze(1)

        best_index = torch.argmax(values).item()
        return final_states[best_index]

    def remember(self, state_features, next_state_features, reward, done):
        """
        Enregistre une transition dans la mémoire.

        state_features      : features du plateau AVANT le placement de
                               cette pièce (= features de l'afterstate
                               précédent dans la chaîne).
        next_state_features : features du plateau APRÈS verrouillage de
                               cette pièce (l'afterstate réellement atteint,
                               pas un candidat recalculé).
        reward               : récompense obtenue en posant la pièce.
        done                 : True si la partie est terminée.
        """
        self.memory.append((state_features, next_state_features, reward, done))

    def replay(self):
        """Routine d'entraînement sur un mini-batch de la mémoire (TD(0))."""
        if len(self.memory) < self.batch_size:
            return  # pas assez de souvenirs pour s'entrainer

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.FloatTensor(
            [s for s, _, _, _ in minibatch]
        ).to(device=device)
        next_states = torch.FloatTensor(
            [ns for _, ns, _, _ in minibatch]
        ).to(device=device)
        rewards = torch.FloatTensor(
            [r for _, _, r, _ in minibatch]
        ).to(device=device)
        dones = torch.FloatTensor(
            [float(d) for _, _, _, d in minibatch]
        ).to(device=device)

        # Valeur actuelle prédite pour chaque état "avant"
        V_states = self.policy_net(states).squeeze(1)

        # Cible : reward + gamma * V(état choisi réellement atteint), sans max
        with torch.no_grad():
            V_next = self.policy_net(next_states).squeeze(1)
            targets = rewards + self.gamma * V_next * (1 - dones)

        loss = self.criterion(V_states, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def exploite_more(self):
        # Déclin de la curiosité
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename="tetris_model.pth"):
        """Sauvegarde des poids du réseau."""
        torch.save(self.policy_net.state_dict(), filename)

    def load(self, weight_path):
        """Charge les poids dans le réseau."""
        policy = ValueNetwork()
        try:
            policy.load_state_dict(torch.load(weight_path))
            policy.eval()
        except Exception:
            pass
        return policy