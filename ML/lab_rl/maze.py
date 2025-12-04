import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Maze:
    DEFAULT_BOTTOM_WALLS = [
        "1c", "1d", "1i", "1j", "2d", "2g", "4c", "4d", "4e", "5e",
        "6e", "6f", "6g", "6h", "8c", "8d", "9b", "9c", "9d", "9i", "8g"
    ]
    
    DEFAULT_RIGHT_WALLS = [
        "2b", "3b", "4b", "6b", "7b", "8b", "2d", "9d", "5e",
        "3f", "4f", "4i", "3g", "7h", "8h", "9h", "8i", "9i", "9e", "8e"
    ]    
    def __init__(self, width=10, height=10, bottom_walls=None, right_walls=None):
        self.width = width
        self.height = height
        self.size = width * height
        
        # Старт и финиш фиксированы
        self.start_pos = 0   # S0
        self.goal_pos = 99   # S99
        
        # Награды
        self.goal_reward = 20
        self.step_penalty = -0.1
        
        # Создаем стены
        bottom = bottom_walls or self.DEFAULT_BOTTOM_WALLS
        right = right_walls or self.DEFAULT_RIGHT_WALLS
        self.walls = self._create_walls_from_description(bottom, right)

    def _pos_to_state(self, row, col_char):
        # перевод буквы в число
        col_mapping = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4,
                      'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9}
        col = col_mapping[col_char.lower()]
        return (row - 1) * self.width + col

    def _create_walls_from_description(self, bottom_walls, right_walls):
        walls = set()
        
        # горизонтальные стены
        for pos in bottom_walls:
            row = int(pos[:-1])
            col_char = pos[-1]
            state = self._pos_to_state(row, col_char)
            if state + self.width < self.size:
                walls.add((state, state + self.width))
                walls.add((state + self.width, state))
        
        # вертикальные стены
        for pos in right_walls:
            row = int(pos[:-1])
            col_char = pos[-1]
            state = self._pos_to_state(row, col_char)
            if (state + 1) % self.width != 0:
                walls.add((state, state + 1))
                walls.add((state + 1, state))
        
        return walls

    def is_valid_move(self, from_state, to_state):
        # проверка хода
        from_x = from_state % self.width
        from_y = from_state // self.width
        to_x = to_state % self.width
        to_y = to_state // self.width
        
        # границы
        if to_x < 0 or to_x >= self.width or to_y < 0 or to_y >= self.height:
            return False
        
        # только соседи
        if abs(from_x - to_x) + abs(from_y - to_y) != 1:
            return False
        
        # стена
        if (from_state, to_state) in self.walls:
            return False
        
        return True

    def get_reward(self, state):
        if state == self.goal_pos:
            return self.goal_reward
        else:
            return self.step_penalty

    def get_possible_actions(self, state):
        actions = []
        # проверяем 4 направления
        if self.is_valid_move(state, state - self.width):
            actions.append(0)  # вверх
        if self.is_valid_move(state, state + 1):
            actions.append(1)  # вправо
        if self.is_valid_move(state, state + self.width):
            actions.append(2)  # вниз
        if self.is_valid_move(state, state - 1):
            actions.append(3)  # влево
        
        return actions

    def visualize(self, title="Лабиринт"):
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # сетка
        for i in range(self.width + 1):
            ax.plot([i, i], [0, self.height], 'k-', linewidth=1, alpha=0.3)
            ax.plot([0, self.width], [i, i], 'k-', linewidth=1, alpha=0.3)
        
        # стены
        for from_state, to_state in self.walls:
            from_x = from_state % self.width
            from_y = from_state // self.width
            to_x = to_state % self.width
            to_y = to_state // self.width
            
            from_y_plot = self.height - 1 - from_y
            to_y_plot = self.height - 1 - to_y
            
            if abs(from_state - to_state) == 1:  # вертикальная
                x = max(from_x, to_x)
                y_bottom = min(from_y_plot, to_y_plot)
                ax.plot([x, x], [y_bottom, y_bottom + 1], 'r-', linewidth=6)
            else:  # горизонтальная
                y = max(from_y_plot, to_y_plot)
                x_left = min(from_x, to_x)
                ax.plot([x_left, x_left + 1], [y, y], 'r-', linewidth=6)
        
        # старт и цель
        start_x = self.start_pos % self.width + 0.5
        start_y = (self.height - 1 - self.start_pos // self.width) + 0.5
        goal_x = self.goal_pos % self.width + 0.5
        goal_y = (self.height - 1 - self.goal_pos // self.width) + 0.5
        
        ax.add_patch(patches.Circle((start_x, start_y), 0.4, color='green', alpha=0.8))
        ax.add_patch(patches.Circle((goal_x, goal_y), 0.4, color='yellow', alpha=0.8))
        ax.text(start_x, start_y, 'S0', ha='center', va='center', fontsize=9, fontweight='bold')
        ax.text(goal_x, goal_y, 'S99', ha='center', va='center', fontsize=9, fontweight='bold')
        
        # номера состояний
        for state in range(self.size):
            x = state % self.width + 0.5
            y = (self.height - 1 - state // self.width) + 0.5
            ax.text(x, y, f'S{state}', ha='center', va='center', fontsize=6, alpha=0.7)
        
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    maze = Maze()
    maze.visualize()
    print(f"Лабиринт {maze.width}x{maze.height}")
    print(f"Старт: S{maze.start_pos}, Цель: S{maze.goal_pos}")