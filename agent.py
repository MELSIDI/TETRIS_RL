import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from tetris.tetris import Action


# Si  GPU Disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Couches de Convolution
        # Entrée : 1 canal (noir et blanc), sortie : 16 filtres, fenêtre de 3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Sortie : 32 filtres
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)

        # 
        # Entrer : 32 * (4*10 pour la piece actuelle et la prochaine + 1*10 ligne  de separation
        # + 20*10 la matrice de tetrice) = 32 * 25*10 = 8000 
        self.fc1 = nn.Linear(8000, 256)
        self.fc2 = nn.Linear(256, 64)
        # Sortie : 6 valeurs (les Q-Values pour les 6 Actions(y compris rien !))
        self.out = nn.Linear(64, 6)

    def forward(self, x):
        # Convolution
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Applatissiment
        x = x.view(x.size(0), -1)

        # Passages dans le Danse
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)



class Agent:
    def __init__(self, weight_path):
        # Hyperparamètre
        self.gamma = 0.99           # Importance du futur
        self.epsilon = 1.0          # Taux d'exploration ici 100ù au début
        self.epsilon_min = 0.01     # Taux minimal d'exploration
        self.epsilon_decay = 0.995  # Taux de réduction de l'exploration
        self.batch_size = 32        # Nombre de souvenirs étudiés par boucle

        # Mémoire
        self.memory = deque(maxlen=10000)

        # Intialisation des Réseaux
        self.policy_net, self.target_net = self.load(weight_path)
        self.policy_net.to(device=device)
        self.target_net.to(device=device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.SmoothL1Loss() # MSELoss() ou SmoothL1Loss()

        # Les action légales
        self.actions = [None, Action.RIGHT, Action.LEFT, Action.ROTATE, Action.SOFTDROP, Action.HARDDROP]



    def remenber(self, state, action, reward, next_state, done):
        """Enregistre une transition dans la mémoire"""
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state, train_mode=True):
        """Prend une action selon la statégie Epsilo,-Greedy"""
        # Chois aléatoire (Exploration)
        if train_mode and random.random() <= self.epsilon:
            return random.choice(self.actions)

        # Chois intelligznt (Exploitation)
        # Tenseur (1, 1, 25, 10)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(device=device)

        with torch.no_grad(): # Prediction
            q_values = self.policy_net(state_tensor)

        action_index = torch.argmax(q_values).item()
        return self.actions[action_index]


    def replay(self):
        """La routine d'entrainement"""
        if len(self.memory) < self.batch_size:
            return # pas assez de souvenirs pour s'netrainer

        minibatch = random.sample(self.memory, self.batch_size)

        # Extraction des tenseurs
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for memory in minibatch:
            states.append([memory[0]])
            actions.append(self.actions.index(memory[1]))
            rewards.append(memory[2])
            next_states.append([memory[3]])
            dones.append(memory[4])
        states_tensor = torch.FloatTensor(states).to(device=device)
        actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device=device)
        rewards_tensor = torch.FloatTensor(rewards).flatten().to(device=device)
        next_states_tensor = torch.FloatTensor(next_states).to(device=device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool).to(device=device)

        # Calcul des prédiction Actuelles (Q-Values)
        Q_values = self.policy_net(states_tensor)
        Q_state_actions = torch.gather(Q_values,
                                      dim=1, index=actions_tensor)
        
       # Calcul de la meilleur Q_value futur possible
        Q_next_values = self.target_net(next_states_tensor)
        Q_next_state_actions = torch.where(dones_tensor, 0.0,
                                           Q_next_values.max(dim=1)[0])
        
        # L'equation de Bellman (La Target)
        target = rewards_tensor + self.gamma * Q_next_state_actions

        # Retropropagation (Learning)
        loss = self.criterion(Q_state_actions, target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


    def exploite_more(self):
        # Declin de la curiosité
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


    def save(self, filename="tetris_model.pth"):
        """Sauvegarde des poids du réseau principal"""
        torch.save(self.policy_net.state_dict(), filename)


    def load(self, weight_path):
        """charger les pois dans le réseaux"""
        policy = DQN()
        target = DQN()

        try:
            policy.load_state_dict(torch.load(weight_path))
            policy.eval()
        except:
            pass

        target.load_state_dict(policy.state_dict())
        target.eval()   # Le réseau cible ne s'entraine Jamais

        return policy, target


        

        

