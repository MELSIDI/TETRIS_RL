# 🟩 TETRIS_RL : Environnement d'Apprentissage par Renforcement

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-2.6.1-00CC66?style=for-the-badge&logo=pygame&logoColor=white)
![Reinforcement Learning](https://img.shields.io/badge/Reinforcement%20Learning-En%20cours-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Statut-Environnement%20Pr%C3%AAt-blue?style=for-the-badge)

Bienvenue dans le dépôt du projet **TETRIS_RL** !

Ce projet est un environnement Tetris personnalisé et robuste, développé de zéro en Python et Pygame. Conçu principalement comme un terrain d'essai pour les agents d'**Apprentissage par Renforcement (RL)**, il intègre une physique avancée (comme la gravité par amas basée sur un algorithme DFS) tout en conservant une interface rétro entièrement jouable pour les joueurs humains.

---

## 📖 Table des Matières

- [Aperçu](#-aperçu)
- [Mécaniques Principales & Physique](#️-mécaniques-principales--physique)
- [Intégration RL & Flux de Données](#-intégration-rl--flux-de-données)
- [Tableau de Bord Pygame](#-tableau-de-bord-pygame)
- [Contrôles](#️-contrôles)
- [Structure du Dépôt](#-structure-du-dépôt)
- [Démarrage Rapide](#-démarrage-rapide)
- [Feuille de Route du Projet](#-feuille-de-route-du-projet)
- [Technologies Utilisées](#️-technologies-utilisées)

---

## 🎯 Aperçu

L'objectif de ce projet est de créer une passerelle parfaite entre un jeu d'arcade classique et un environnement d'entraînement d'IA moderne. Le moteur sépare strictement l'état physique du jeu de son rendu visuel, permettant aux agents de traiter les données en quelques millisecondes, ou aux humains de jouer en temps réel à 30 FPS.

- **Moteur de Jeu** — Un système de grille 10×20 gérant les collisions complexes, les *Wall Kicks* et l'effacement des lignes.
- **Physique Personnalisée** — Implémentation des *Sticky Blocks* (gravité par amas) où les blocs flottants déconnectés tombent de manière réaliste.
- **Prêt pour l'IA** — Des méthodes comme `get_full_state()` exposent les matrices brutes directement aux réseaux de neurones.
- **Interface Rétro** — Un visualiseur Pygame totalement découplé imitant l'esthétique des anciens terminaux cathodiques.

---

## 🏗️ Mécaniques Principales & Physique

L'environnement (`tetris.py`) agit comme la source de vérité absolue, gérant toutes les transformations mathématiques et les vérifications des limites spatiales.

| Fonctionnalité | Description | Détail de l'implémentation |
|---|---|---|
| **Matrice de la Grille** | Dimensions standard de Tetris | 10 colonnes par 20 lignes (`MATRIX_WIDTH`, `MATRIX_HEIGHT`) |
| **Wall Kicks** | Collisions intelligentes lors de la rotation | Teste 8 décalages X/Y successifs (dont deux diagonales) pour faire rebondir les pièces contre les murs ou le sol si la rotation échoue |
| **Gravité par Amas** | Les blocs non soutenus tombent de manière fluide | Utilise l'algorithme **DFS (Depth-First Search)** pour regrouper les cellules adjacentes (orthogonales et diagonales) et faire chuter les amas non ancrés au sol |
| **Chutes Unifiées** | Mécanique de descente standardisée | Le Soft Drop (gravité/accélération du joueur) et le Hard Drop (téléportation instantanée) utilisent la même logique de collision sous-jacente |
| **Vitesse Dynamique** | Difficulté progressive | Le délai de chute diminue automatiquement en fonction du score du joueur, avec une limite fixée à `MAX_SPEED` |

---

## 🔄 Intégration RL & Flux de Données

L'architecture est explicitement conçue pour entraîner des *Value Networks* (réseaux de valeur) et des agents. L'état du jeu est mis à jour via la méthode `set_state(action)`, qui renvoie la récompense immédiate pour l'agent.

**Système de Récompense (Reward) :**

| Lignes Effacées | Récompense |
|---|---|
| 1 ligne | **3 points** |
| 2 lignes | **8 points** |
| 3 lignes | **16 points** |
| 4 lignes (Tetris) | **50 points** |

Le **score affiché** dans le jeu prend en plus en compte la vitesse courante (`score += 2 * (reward + speed)`), ce qui valorise les lignes effacées à un niveau de vitesse plus élevé — une incitation supplémentaire, utile pour un futur agent, à survivre plus longtemps tout en restant efficace.

**Observation de l'État par l'Agent :**
La fonction `get_full_state()` génère en toute sécurité une vue combinée de la grille figée et de la pièce active en train de chuter, fournissant la matrice exacte requise pour la couche d'entrée d'une IA, et ce **sans altérer l'état réel du jeu**.

---

## 📊 Tableau de Bord Pygame

Le rendu visuel est intégralement pris en charge par `game_ui_render.py` à l'aide de la bibliothèque Pygame.

**Éléments Clés de l'Interface :**

- **Esthétique Terminal** : des cellules vert vif (`(0, 255, 0)`) dessinées sur un fond noir absolu (`(0, 0, 0)`).
- **Pièce Suivante** : une zone d'affichage dédiée montrant le Tetrimino à venir.
- **Statistiques en Direct** : suivi en temps réel du score actuel (`SCORE`) et de la vitesse (`SPEED`).
- **Record Persistant** : le meilleur score (`BEST SCORE`) est lu et sauvegardé de manière persistante dans un fichier texte local (`best_score.txt`).
- **État Game Over** : un écran superposé dédié permettant de relancer instantanément la partie en appuyant sur la touche `ENTRÉE`.

---

## ⌨️ Contrôles

| Touche | Action |
|---|---|
| `←` | Déplacer la pièce à gauche |
| `→` | Déplacer la pièce à droite |
| `↓` | Chute douce (Soft Drop) |
| `↑` | Chute instantanée (Hard Drop) |
| `Espace` | Faire pivoter la pièce |
| `Entrée` | Relancer la partie après un Game Over |

---

## 📂 Structure du Dépôt

```text
tetris_rl/
│
├── trl_env/                   # Environnement virtuel ignoré par Git
├── __pycache__/                # Fichiers Python compilés ignorés par Git
├── best_score.txt              # Stockage persistant du record historique (meilleur score)
├── CONSTANTS.py                 # Paramètres globaux : dimensions des matrices, couleurs, vitesse maximale
├── game_ui_render.py            # Fonctions de rendu Pygame pour les cellules, la grille et l'interface utilisateur
├── main.py                      # Boucle principale (30 FPS), gestion des événements et de la gravité dynamique
├── requirement.txt              # Dépendances du projet (pygame==2.6.1)
├── tetrimino.py                 # Enumération définissant les matrices 4x4 de rotation pour I, O, L, J, S, Z, T
└── tetris.py                    # Moteur principal : collisions, récompenses, gestion de l'état et gravité DFS
```

---

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.12+
- pip

### Installation

```bash
git clone https://github.com/MELSIDI/TETRIS_RL.git
cd TETRIS_RL
pip install -r requirement.txt
```

### Lancer le jeu

```bash
python main.py
```

Utilisez les flèches directionnelles pour déplacer la pièce, `Espace` pour la faire pivoter, et essayez de battre le meilleur score enregistré ! 🏆

---

## 📋 Feuille de Route du Projet

### Phase 1 : Moteur & Mode Humain (Terminé)

- Construire un moteur physique robuste basé sur une matrice 10×20.
- Implémenter les rotations Wall Kicks et la gravité DFS par amas (Sticky Blocks).
- Développer l'interface visuelle Pygame et la boucle de Game Over/Redémarrage.

### Phase 2 : Simulation de l'Agent Macro (Prochaines Étapes)

- Implémenter un algorithme **BFS (Breadth-First Search)** au sein de l'environnement.
- Générer tous les états de grille finaux valides et les chemins d'action possibles pour n'importe quelle pièce entrante.
- Permettre au moteur d'exécuter des séquences d'actions automatiques générées par le chercheur de chemin BFS.

### Phase 3 : Apprentissage par Renforcement (Prévu)

- Connecter un réseau de neurones **PyTorch** (Value Network) pour évaluer les différents états du plateau.
- Comparer la stabilité de l'apprentissage entre un agent **« Macro »** (qui sélectionne directement les états finaux générés par le BFS) et un agent **« Micro »** (qui choisit les pressions de touches de manière individuelle, action par action).

---

## 🛠️ Technologies Utilisées

- **Langage :** Python 3.12+
- **Moteur Graphique :** Pygame (`pygame==2.6.1`)
- **Algorithmes :** DFS (gravité par amas — implémenté), BFS (recherche de chemin — prévu)
- **Apprentissage Automatique (prévu) :** PyTorch (Value Network)
- **Persistance des Données :** fichier texte local (`best_score.txt`)