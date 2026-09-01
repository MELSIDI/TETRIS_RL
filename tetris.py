import random
from tetrimino import Tetrimino, MATRIX_LEN
from enum import Enum, auto


class Action(Enum):
    LEFT = auto()
    RIGHT = auto()
    DOWN = auto()
    ROTATE = auto()



class Tetris:
    def __init__(self):
        # La grille principal du Jeu (20x10)
        self.matrix = [[0 for _ in range(10)] for _ in range(20)]

        # Les Tetriminos, l'actuel et le prochain
        self.current_tetrimino = random.choice(list(Tetrimino))
        self.next_tetrimino = random.choice(list(Tetrimino))

        # Rotation actuelle
        self.current_rotation = 0

        # Initialisation de la position du tetrimino
        self.current_x , self.current_y = self.init_tetrimino_position()

       

        # Fin de partie
        self.game_over = False


    def get_tetimino_matrix(self):
        # Avoir la matrice actuelle du tetrimo et correspondant à la bonne rotation
        return self.current_tetrimino.value[self.current_rotation]


    def init_tetrimino_position(self):
        """
        Initialiser la bonne position pour le tetrimo qui apparait pour eviter 
        des moitier de tetrimo qui apparaisse, il faut que le tetrimo apparraisse
        d'un seul coup.
        Elle retour x et y cette fonction
        """
        x = 3
        y = 0
        for row in self.current_tetrimino.value[self.current_rotation]:
            if all(cell == 0 for cell in row):
                y -= 1
            else:
                break
        return x, y
        



    def check_collision(self, offset_x=0, offset_y=0, rotation=None):
        """
        Vérifie si la pièce toucherait un obstacle avec ce déplacement ou cette rotation.
        Retourne True si collision, False si le mouvement est valide.
        offset_x : decalage à tester suivant l'axe x
        offset_y : decalage à tester suivant l'axe y
        rotation : rotation à tester
        """
        if rotation is None:
            rotation = self.current_rotation
            
        piece_matrix = self.current_tetrimino.value[rotation]
        
        for y in range(MATRIX_LEN):
            for x in range(MATRIX_LEN):
                if piece_matrix[y][x] == 1: # Si la case de la pièce est pleine
                    board_x = self.current_x + x + offset_x
                    board_y = self.current_y + y + offset_y
                    
                    # 1. Vérification des murs (Gauche, Droite, Bas)
                    if board_x < 0 or board_x >= 10 or board_y >= 20:
                        return True
                        
                    # 2. Vérification des blocs figés sur la grille
                    if board_y >= 0 and self.matrix[board_y][board_x] == 1:
                        return True
                        
        return False

    def get_full_state(self):
        """
        Génère une vue complète (Grille figée + Pièce active) SANS modifier la vraie grille.
        C'est ce que seras envoyer au réseau de neurones (Value Network).
        """
        # On copie la grille figée
        state = [row[:] for row in self.matrix] 
        
        # On superpose la pièce active
        piece_matrix = self.get_tetrimino_matrix()
        for y in range(MATRIX_LEN):
            for x in range(MATRIX_LEN):
                if piece_matrix[y][x] == 1:
                    board_y = self.current_y + y
                    board_x = self.current_x + x
                    if board_y >= 0: # Pour éviter les erreurs si la pièce spawn tout en haut
                        state[board_y][board_x] = 1
                        
        return state


    def set_state(self, action):
        """
        Met à jour l'état du jeux(l'environnement du jeux) suivant l'action action

        """
        # Les decalage qui pourraient être provoquer par une action 
        # suivant l'axe des x, des y ou celui des rotation


        offset_x = 0 
        offset_y = 0

        # reward
        reward = 0

        # Rotation
        if action == Action.ROTATE:
            self.try_rotation() # Rotation et Wall Kicks gerer automatiquement
            return reward

        # Translation
        if action == Action.RIGHT:
            offset_x = 1
        elif action == Action.LEFT:
            offset_x = -1
        elif action == Action.DOWN:
            offset_y = 1

        # L'état de la colision
        collision = self.check_collision(offset_x, offset_y)

        if collision:
            if action == Action.DOWN:
                # VERROUILLAGE : La pièce ne peut plus descendre, on la fige.
                tetrimino_matrix = self.get_tetimino_matrix()
                for y in range(MATRIX_LEN):
                    for x in range(MATRIX_LEN):
                        if tetrimino_matrix[y][x] == 1:
                            self.matrix[self.current_y + y][self.current_x + x] = 1

                # On elimine les lignes 
                reward = self.clear_lines()

                # RESPONSABILITE : On charge la piece suivante
                self.current_tetrimino = self.next_tetrimino
                self.next_tetrimino = random.choice(list(Tetrimino))

                # RENITIALISATION : On replace la nouvelle pièce 
                self.current_rotation = 0
                self.current_x, self.current_y = self.init_tetrimino_position()

                # GAME OVER: Si la nouvelle pièce entre en collision dès son apparition
                if self.check_collision():
                    self.game_over = True

            else:
                # Si l'action est un autre move on annule l'action (on fait rien)
                # la piece continue de tomber normalement ensuite.
                pass

        else: # Sinon on le fait bouger, car y'a pas d'obstacle
            self.current_x += offset_x
            self.current_y += offset_y

        return reward


    def clear_lines(self):
        """
        Vérifie et supprime les lignes complètes.
        Retourne le reward obtenue pour l'agent RL
        """
        lines_cleared = 0
        new_matrix = []

        # On filtre la grille en gardant que les lignes incomplètes
        for row in self.matrix:
            if all(cell == 1 for cell in row):
                lines_cleared += 1
            else:
                new_matrix.append(row)

        # On rajoute les lignes vides tout en haut pour compenser celles détruites
        for _ in range(lines_cleared):
            new_matrix.insert(0, [0 for _ in range(10)])
        
        # On met à jour la grille
        self.matrix = new_matrix

        # Calcul de la récompense type Tetris pour encourager les combos
        reward = 0
        if lines_cleared == 1:
            reward = 3
        elif lines_cleared == 2:
            reward = 8
        elif lines_cleared == 3:
            reward = 16
        elif lines_cleared == 4: # Tetris
            reward = 50

        return reward


    def try_rotation(self):
        """
        Tente de tourner la pièce. Si ça bloque, essaie des Wall Kicks.
        Retourne True si la rotation a reussi, False si elle est impossible.
        """
        # on recupère tous les état de rotation disponible
        rotation_number = len(self.current_tetrimino.value)

        # on prend la potentiel futur rotation
        new_rotation = (self.current_rotation + 1) % rotation_number

        # Liste des décalages à tester dans l'ordre (dx, dy)
        # Note : dans cette matrice, y augmente vers le bas, donc -1 en y = déplacement vers le haut
        kick_tests = [
            (0, 0),    # 1. Test normal (sur place)
            (-1, 0),   # 2. Rebond 1 case à gauche (mur droite)
            (1, 0),    # 3. Rebond 1 case à droite (mur gauche)
            (0, -1),   # 4. Rebond 1 case vers le haut (collé au sol)
            (-2, 0),   # 5. Rebond 2 cases à gauche (vital pour la longue barre I)
            (2, 0),    # 6. Rebond 2 cases à droite
            (-1, -1),  # 7. Diagonale haut-gauche
            (1, -1)    # 8. Diagonale haut-droite
        ]

        for dx, dy in kick_tests:
            if not self.check_collision(offset_x=dx, offset_y=dy, rotation=new_rotation):
                # Le Wall Kick a fonctionné ! On applique la position et la rotation.
                self.current_x += dx
                self.current_y += dy
                self.current_rotation = new_rotation
                return True

        # Si on arrive ici, aucun kick n'a marché, la pièce reste bloquée
        return False

