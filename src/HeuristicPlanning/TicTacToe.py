import numpy as np
import pygame

class TicTacToe:
    def __init__(self):
        self.agents = ["You", "AI-agent"]
        self.obs = np.zeros(shape=(3,3,2), dtype=int)
        self.action_mask = np.ones(shape = 9)
        self.whoseTurn = 0
        self.winner = None

        # Fenstergröße und Farben definieren
        self.window_size = 600
        self.grid_color = (200, 200, 200)
        self.bg_color = (128, 185, 39)
        self.cross_color = (255, 0, 0)
        self.circle_color = (0, 0, 255)
        self.line_width = 5
        self.cross_width = 10
        self.circle_width = 10
        self.circle_radius = 60

        # Pygame initialisieren
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Tic Tac Toe")

    def __str__(self):
        result = []

        # Header for the game
        hor_line = "+---+---+---+"
        result.append(hor_line)

        # go through obs array
        for i in range(3):
            row = "|"
            for j in range(3):
                if self.obs[i, j, 0] == 1:
                    row += " X |"
                elif self.obs[i, j, 1] == 1:
                    row += " O |"
                else:
                    row += "   |"  # empty cell
            result.append(row)  # add row
            result.append(hor_line)  # add separator

        # return result as string
        return "\n".join(result)

    def step(self, action):

        # adjust env representation and action masking
        self.obs[action % 3,action // 3, self.whoseTurn] = 1
        self.action_mask[action] = 0

        # move is finished
        self.whoseTurn = self.changePlayer()

        self.render()

        # check whether game is over
        done = self.isWon() or self.isDraw()

        return self.obs, done, {'Winner': self.winner, 'isDraw': self.isDraw(), 'isWon': self.isWon()}

    def render(self):
        # Fenster mit Hintergrundfarbe füllen
        self.screen.fill(self.bg_color)

        # Raster zeichnen
        cell_size = self.window_size // 3
        for i in range(1, 3):  # Zwei Linien horizontal und vertikal
            pygame.draw.line(self.screen, self.grid_color, (0, i * cell_size), (self.window_size, i * cell_size),
                             self.line_width)
            pygame.draw.line(self.screen, self.grid_color, (i * cell_size, 0), (i * cell_size, self.window_size),
                             self.line_width)

        # Figuren auf dem Spielfeld zeichnen
        for x in range(3):  # Spalten (0, 3, 6 ...)
            for y in range(3):  # Reihen (0, 1, 2 ...)
                # Umrechung der Position für das vertikale Layout
                row = y
                col = x
                index = col * 3 + row  # 0, 1, 2, 3, 4, 5, 6, 7, 8

                # Position in der Anzeige
                center_x = col * cell_size + cell_size // 2
                center_y = row * cell_size + cell_size // 2

                # Kreuz zeichnen, wenn obs[index, 0] == 1
                if self.obs[index % 3, index // 3, 0] == 1:
                    pygame.draw.line(self.screen, self.cross_color,
                                     (center_x - self.circle_radius, center_y - self.circle_radius),
                                     (center_x + self.circle_radius, center_y + self.circle_radius),
                                     self.cross_width)
                    pygame.draw.line(self.screen, self.cross_color,
                                     (center_x + self.circle_radius, center_y - self.circle_radius),
                                     (center_x - self.circle_radius, center_y + self.circle_radius),
                                     self.cross_width)

                # Kreis zeichnen, wenn obs[index, 1] == 1
                if self.obs[index % 3, index // 3, 1] == 1:
                    pygame.draw.circle(self.screen, self.circle_color,
                                       (center_x, center_y), self.circle_radius, self.circle_width)

        # Fenster aktualisieren
        pygame.display.flip()

    def reset(self):
        self.obs = np.zeros(shape=(3,3,2), dtype=int)
        self.action_mask = np.ones(shape = 9)
        self.whoseTurn = 0
        self.winner = None

        self.render()

        return self.obs

    def changePlayer(self):
        return (self.whoseTurn + 1) % 2

    def get_user_action(self):
        """
        wait for a mouse click and provide the index of the clicked cell.
        """
        cell_size = self.window_size // 3

        # event loop
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None  # close window

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Linksklick
                    # mouse position
                    mouse_x, mouse_y = event.pos

                    # calculation of column and row of mouse
                    col = mouse_x // cell_size  # 0, 1, 2
                    row = mouse_y // cell_size  # 0, 1, 2

                    # Calculate row and col with index 0-8 according to the layout
                    # 0 | 3 | 6
                    # 1 | 4 | 7
                    # 2 | 5 | 8
                    index = col * 3 + row

                    # check whether it's a legal action
                    if self.action_mask[index] == 1:
                        return index
                    else:
                        continue

    def isWon(self):
        for i in range(3):
            for j in range(2):
                if (sum(self.obs[:, i, j]) == 3) or sum(self.obs[i, :, j]) == 3 or \
                    sum(np.diagonal(self.obs[:, :, j])) == 3 or \
                        sum(np.diagonal(self.obs[range(3), ::-1, j])) == 3:
                    self.winner = self.agents[j]
                    return True
        return False

    def isDraw(self):
        return not (self.obs[:, :, 0] + self.obs[:, :, 1] == 0).any() and not self.isWon()