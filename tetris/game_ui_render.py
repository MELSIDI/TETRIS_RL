import pygame
from tetris.CONSTANTS import *



def draw_cell(windows, x, y):
    pygame.draw.rect(windows, CELL_COLOR, (x, y, CELL_WIDTH, CELL_HEIGHT), 2)
    #pygame.draw.circle(windows, CELL_COLOR, (x + CELL_WIDTH // 2, y + CELL_HEIGHT // 2), 2)
    pygame.draw.rect(windows, CELL_COLOR, (x + 5, y + 5, CELL_WIDTH - 10, CELL_HEIGHT - 10))


def draw_empty(windows, x, y):
    pygame.draw.circle(windows, CELL_COLOR, (x, y), 2)
    pygame.draw.circle(windows, CELL_COLOR, (x + CELL_WIDTH, y), 2)
    pygame.draw.circle(windows, CELL_COLOR, (x, y + CELL_HEIGHT), 2)
    pygame.draw.circle(windows, CELL_COLOR, (x + CELL_WIDTH, y + CELL_HEIGHT), 2)


def draw_matrix(windows, matrix):
    for y in range(MATRIX_HEIGHT):
        for x in range(MATRIX_WIDTH):
            if matrix[y][x] == 0:
                draw_empty(windows, X + x * CELL_WIDTH, Y + y * CELL_HEIGHT)
            elif matrix[y][x] == 1:
                draw_cell(windows, X + x * CELL_WIDTH, Y + y * CELL_HEIGHT)
                

def draw_tetrimino(windows, tetrimino_matrix, tetrimino_x, tetrimino_y):
    for x in range(TETRIMINO_LEN):
        for y in range(TETRIMINO_LEN):
            if tetrimino_matrix[y][x] == 1:
                draw_cell(windows, X + (tetrimino_x + x) * CELL_WIDTH, Y + (tetrimino_y + y) * CELL_HEIGHT)


def draw_next_tetrimino(windows, next_tetrimino_matrix):
    y_start = 0
    for row in next_tetrimino_matrix:
        if all(cell == 0 for cell in row):
            y_start += 1
        else:
            break
    if all(cell == 1 for cell in next_tetrimino_matrix[y_start]):
        y_start = 0
        
    for x in range(TETRIMINO_LEN):
        for y in range(y_start, y_start + 2):
            if next_tetrimino_matrix[y][x] == 1:
                draw_cell(windows, X + GRAPHIC_MATRIX_WIDTH + 45 + x * CELL_WIDTH, Y + 60 + (y - y_start)  * CELL_HEIGHT)


def draw_game_over_screen(windows, score, best_score, speed, player):
    police_1 = pygame.font.Font(None, 100)
    police_2 = pygame.font.Font(None, 50)
    police_3 = pygame.font.Font(None, 30)

    windows.blit(police_1.render("GAME OVER", True, CELL_COLOR), (20, 240))
    windows.blit(police_2.render(f"PLAYER: {player}", True, CELL_COLOR), (50, 300))
    windows.blit(police_2.render(f"SPEED: {speed}", True, CELL_COLOR), (50, 340))
    windows.blit(police_2.render(f"SCORE: {score}", True, CELL_COLOR), (50, 380))
    windows.blit(police_2.render(f"BEST SCORE: {best_score}", True, CELL_COLOR), (50, 420))
    windows.blit(police_3.render(f"TO RESTART THE GAME PRESS ENTER", True, CELL_COLOR), (50, 460))




def draw_game_screen(windows, score, best_score, speed, player):
    police_1 = pygame.font.Font(None, 100)
    police_2 = pygame.font.Font(None, 50)

    # Rect pour Encadrer tous le desing graphique de Tetris
    pygame.draw.rect(windows, CELL_COLOR, 
                     (X - 13, 2, 515, 705), 
                     2, border_radius=4)

    # Rect Pour encadrer le Title
    pygame.draw.rect(windows, CELL_COLOR,
                        (X - 8, 7, 505, 73),
                        2, border_radius=4)
    # Le rendu du Title
    title = police_1.render("TETRIS", True, CELL_COLOR)
    windows.blit(title, (125, 12))

    # Rect pour encadrer la matrice du Jeux
    pygame.draw.rect(windows, CELL_COLOR, 
                         (X - 8, Y - 8, GRAPHIC_MATRIX_WIDTH + 16 , GRAPHIC_MATRIX_HEIGHT + 16), 
                         2, border_radius=4)
    
    # Rect pour Encadrer les Metadonnées sur la partie 
    pygame.draw.rect(windows, CELL_COLOR,
                             (X + GRAPHIC_MATRIX_WIDTH + 12, Y - 8, 185 , GRAPHIC_MATRIX_HEIGHT + 16),
                             2, border_radius=4)
    
    # Rect pour encadrer le next_tetrinimo
    pygame.draw.rect(windows, CELL_COLOR,
                         (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 50 , 
                          160 , 80), 
                          width=1, border_radius=4)
    # Next rendu 
    next_tetrimino_title = police_2.render("NEXT:", True, CELL_COLOR)
    windows.blit(next_tetrimino_title, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 10))

    # Ligne de séparation
    pygame.draw.line(windows, CELL_COLOR, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 160), (X + GRAPHIC_MATRIX_WIDTH + 185 , Y + 160))


    # Score
    score_title = police_2.render("SCORE:", True, CELL_COLOR)
    windows.blit(score_title, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 180))
    current_score = police_2.render(f"{score}", True, CELL_COLOR)
    windows.blit(current_score, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 220))

    # Ligne de séparation
    pygame.draw.line(windows, CELL_COLOR, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 270), (X + GRAPHIC_MATRIX_WIDTH + 185, Y + 270))

    # BEST Score
    best_score_title = police_2.render("BEST:", True, CELL_COLOR)
    windows.blit(best_score_title, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 290))
    best_score_render = police_2.render(f"{best_score}", True, CELL_COLOR)
    windows.blit(best_score_render, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 330))

    # Ligne de séparation
    pygame.draw.line(windows, CELL_COLOR, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 380), (X + GRAPHIC_MATRIX_WIDTH + 185, Y + 380))

    # speed
    speed_title = police_2.render("SPEED:", True, CELL_COLOR)
    windows.blit(speed_title, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 400))
    speed_render = police_2.render(f"{speed}", True, CELL_COLOR)
    windows.blit(speed_render, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 440))

    # Ligne de séparation
    pygame.draw.line(windows, CELL_COLOR, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 490), (X + GRAPHIC_MATRIX_WIDTH + 185, Y + 490))

    # speed
    player_title = police_2.render("PLAYER:", True, CELL_COLOR)
    windows.blit(player_title, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 510))
    player_render = police_2.render(player, True, CELL_COLOR)
    windows.blit(player_render, (X + GRAPHIC_MATRIX_WIDTH + 25, Y + 550))

    
    
    