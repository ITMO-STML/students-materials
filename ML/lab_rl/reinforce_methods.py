import numpy as np
from maze import Maze


class Ablation1_BasicManyCycles:
    # первая попытка - много циклов
    
    def __init__(self, maze):
        self.maze = maze
        self.eta = 0.01
        self.gamma = 0.9
        
        self.goal_reward = 100
        self.step_penalty = -0.1
        
        # для анализа
        self.distances = self._compute_distances()

    def _compute_distances(self):
        # расстояние от каждой клетки до цели
        distances = {}
        goal_x = self.maze.goal_pos % self.maze.width
        goal_y = self.maze.goal_pos // self.maze.width
        
        for state in range(self.maze.size):
            x = state % self.maze.width
            y = state // self.maze.width
            distances[state] = abs(goal_x - x) + abs(goal_y - y)
        
        return distances

    def softmax(self, theta):
        pi = np.zeros_like(theta)
        exp_theta = np.exp(theta)
        
        for i in range(theta.shape[0]):
            row_sum = np.nansum(exp_theta[i, :])
            if row_sum > 0:
                pi[i, :] = exp_theta[i, :] / row_sum
        
        return np.nan_to_num(pi)

    def get_action(self, pi, s):
        probs = pi[s, :]
        probs = probs / probs.sum()
        action = np.random.choice(4, p=probs)
        
        # следующее состояние
        if action == 0:
            s_next = s - self.maze.width
        elif action == 1:
            s_next = s + 1
        elif action == 2:
            s_next = s + self.maze.width
        else:
            s_next = s - 1
        
        if not self.maze.is_valid_move(s, s_next):
            s_next = s
        
        return [action, s_next]

    def generate_episode(self, pi, max_steps=200):
        s = self.maze.start_pos
        history = [[s, np.nan]]
        states = [s]
        
        for _ in range(max_steps):
            action, next_s = self.get_action(pi, s)
            history[-1][1] = action
            history.append([next_s, np.nan])
            states.append(next_s)
            
            if next_s == self.maze.goal_pos:
                break
            
            s = next_s
        
        return history, states

    def calculate_returns(self, history):
        T = len(history) - 1
        returns = []
        
        for t in range(T):
            G = 0
            discount = 1.0
            
            for k in range(t, T):
                state = history[k+1][0]
                if state == self.maze.goal_pos:
                    reward = self.goal_reward
                else:
                    reward = self.step_penalty
                G += discount * reward
                discount *= self.gamma
            
            returns.append(G)
        
        return returns

    def update_theta(self, theta, pi, history):
        T = len(history) - 1
        returns = self.calculate_returns(history)
        delta = np.zeros_like(theta)
        
        for t in range(T):
            state = history[t][0]
            action = history[t][1]
            
            if not np.isnan(action):
                grad = 1 - pi[state, action]
                delta[state, action] = grad * returns[t]
        
        new_theta = theta + self.eta * delta
        new_theta[np.isnan(theta)] = np.nan
        
        return new_theta

    def train(self, max_episodes=100):
        print("\nБазовая версия")
        
        # инициализация
        theta = np.full((self.maze.size, 4), np.nan)
        for state in range(self.maze.size):
            for action in self.maze.get_possible_actions(state):
                theta[state, action] = 1.0
        
        pi = self.softmax(theta)
        
        lengths = []
        goals = []
        
        for episode in range(max_episodes):
            history, states = self.generate_episode(pi)
            length = len(history) - 1
            goal_reached = states[-1] == self.maze.goal_pos
            
            lengths.append(length)
            goals.append(goal_reached)
            
            if episode % 20 == 0:
                min_dist = min(self.distances[s] for s in states)
                status = "ЦЕЛЬ" if goal_reached else ""
                print(f"Эпизод {episode:3d}: длина={length:3d}, "
                      f"мин.расст={min_dist:2d} {status}")
            
            theta = self.update_theta(theta, pi, history)
            pi = self.softmax(theta)
        
        success_rate = sum(goals) / max_episodes
        print(f"\nИтоги:")
        print(f"Успех: {sum(goals)}/{max_episodes} ({success_rate:.1%})")
        print(f"Средняя длина: {np.mean(lengths):.1f}")
        
        return {
            'lengths': lengths,
            'goals': goals,
            'success_rate': success_rate
        }


class Ablation2_AggressivePenalties:
    # вторая попытка - большие штрафы
    
    def __init__(self, maze):
        self.maze = maze
        self.eta = 0.01
        self.gamma = 0.9
        
        self.goal_reward = 10
        self.step_penalty = -0.2
        self.repeat_penalty = -5.0  
        self.cycle_penalty = -10.0  
        
        self.distances = self._compute_distances()

    def _compute_distances(self):
        distances = {}
        goal_x = self.maze.goal_pos % self.maze.width
        goal_y = self.maze.goal_pos // self.maze.width
        
        for state in range(self.maze.size):
            x = state % self.maze.width
            y = state // self.maze.width
            distances[state] = abs(goal_x - x) + abs(goal_y - y)
        
        return distances

    def softmax(self, theta):
        pi = np.zeros_like(theta)
        exp_theta = np.exp(theta)
        
        for i in range(theta.shape[0]):
            row_sum = np.nansum(exp_theta[i, :])
            if row_sum > 0:
                pi[i, :] = exp_theta[i, :] / row_sum
        
        return np.nan_to_num(pi)

    def get_action(self, pi, s):
        probs = pi[s, :]
        probs = probs / probs.sum()
        action = np.random.choice(4, p=probs)
        
        if action == 0:
            s_next = s - self.maze.width
        elif action == 1:
            s_next = s + 1
        elif action == 2:
            s_next = s + self.maze.width
        else:
            s_next = s - 1
        
        if not self.maze.is_valid_move(s, s_next):
            s_next = s
        
        return [action, s_next]

    def generate_episode(self, pi, max_steps=200):
        s = self.maze.start_pos
        history = [[s, np.nan]]
        states = [s]
        
        same_count = 0
        prev_state = s
        
        for _ in range(max_steps):
            action, next_s = self.get_action(pi, s)
            history[-1][1] = action
            history.append([next_s, np.nan])
            states.append(next_s)
            
            # проверка застревания
            if next_s == prev_state:
                same_count += 1
                if same_count > 10:
                    break
            else:
                same_count = 0
                prev_state = next_s
            
            if next_s == self.maze.goal_pos:
                break
            
            s = next_s
        
        return history, states

    def calculate_returns(self, states, history):
        T = len(history) - 1
        returns = []
        
        # считаем посещения для штрафов
        visits = {}
        for state in states:
            visits[state] = visits.get(state, 0) + 1
        
        for t in range(T):
            G = 0
            discount = 1.0
            
            for k in range(t, T):
                state = history[k+1][0]
                
                # базовая награда
                if state == self.maze.goal_pos:
                    reward = self.goal_reward
                else:
                    reward = self.step_penalty
                
                # штрафы
                count = visits[state]
                if count > 1:
                    reward += self.repeat_penalty * (count - 1)
                
                # штраф за цикл
                if k > 0 and state == history[k][0]:
                    reward += self.cycle_penalty
                
                # штраф за стояние
                if k > 1 and state == states[k-1] == states[k-2]:
                    reward += -2.0
                
                G += discount * reward
                discount *= self.gamma
            
            returns.append(G)
        
        return returns

    def update_theta(self, theta, pi, history, states):
        T = len(history) - 1
        returns = self.calculate_returns(states, history)
        delta = np.zeros_like(theta)
        
        for t in range(T):
            state = history[t][0]
            action = history[t][1]
            
            if not np.isnan(action):
                grad = 1 - pi[state, action]
                delta[state, action] = grad * returns[t]
        
        new_theta = theta + self.eta * delta
        new_theta[np.isnan(theta)] = np.nan
        
        return new_theta

    def train(self, max_episodes=100):
        print("\nВерсия с большими штрафами")
        
        theta = np.full((self.maze.size, 4), np.nan)
        for state in range(self.maze.size):
            for action in self.maze.get_possible_actions(state):
                theta[state, action] = 1.0
        
        pi = self.softmax(theta)
        
        lengths = []
        goals = []
        frozen = 0
        
        for episode in range(max_episodes):
            history, states = self.generate_episode(pi)
            length = len(history) - 1
            goal_reached = states[-1] == self.maze.goal_pos
            
            lengths.append(length)
            goals.append(goal_reached)
            
            if length < 10:
                frozen += 1
            
            if episode % 20 == 0:
                min_dist = min(self.distances[s] for s in states)
                status = "ЗАМЕРЗ" if length < 10 else "ЦЕЛЬ" if goal_reached else ""
                print(f"Эпизод {episode:3d}: длина={length:3d}, "
                      f"мин.расст={min_dist:2d} {status}")
            
            theta = self.update_theta(theta, pi, history, states)
            pi = self.softmax(theta)
        
        success_rate = sum(goals) / max_episodes
        print(f"\nИтоги:")
        print(f"Успех: {sum(goals)}/{max_episodes} ({success_rate:.1%})")
        print(f"Замерзаний: {frozen}/{max_episodes}")
        print(f"Средняя длина: {np.mean(lengths):.1f}")
        
        return {
            'lengths': lengths,
            'goals': goals,
            'frozen': frozen,
            'success_rate': success_rate
        }


class StableREINFORCE:
    # стабильная версия
    
    def __init__(self, maze):
        self.maze = maze
        self.eta = 0.1
        self.gamma = 0.95
        
        # сбалансированные награды
        self.goal_reward = 1000
        self.step_penalty = -0.01
        self.progress_bonus = 0.5
        
        # память успешных путей
        self.success_paths = []
        self.best_path = None
        
        self.distances = self._compute_distances()

    def _compute_distances(self):
        distances = {}
        goal_x = self.maze.goal_pos % self.maze.width
        goal_y = self.maze.goal_pos // self.maze.width
        
        for state in range(self.maze.size):
            x = state % self.maze.width
            y = state // self.maze.width
            distances[state] = abs(goal_x - x) + abs(goal_y - y)
        
        return distances

    def softmax(self, theta):
        pi = np.zeros_like(theta)
        
        for i in range(theta.shape[0]):
            possible = self.maze.get_possible_actions(i)
            if not possible:
                continue
            
            # безопасный softmax
            row = np.full(4, -1e6)
            for action in possible:
                if not np.isnan(theta[i, action]):
                    row[action] = theta[i, action]
                else:
                    row[action] = 0.0
            
            max_val = np.max(row)
            exp_row = np.exp(row - max_val)
            sum_exp = np.sum(exp_row)
            
            if sum_exp > 0:
                pi[i, :] = exp_row / sum_exp
            
            # минимальные вероятности для исследования
            min_prob = 0.05
            for action in possible:
                if pi[i, action] < min_prob:
                    pi[i, action] = min_prob
            
            # пере-нормализация
            pi_sum = np.sum(pi[i, :])
            if pi_sum > 0:
                pi[i, :] = pi[i, :] / pi_sum
        
        return pi

    def get_action(self, pi, s, epsilon=0.2):
        possible = self.maze.get_possible_actions(s)
        if not possible:
            return [0, s]
        
        probs = pi[s, :].copy()
        probs_sum = np.sum(probs)
        if probs_sum <= 0:
            for action in possible:
                probs[action] = 1.0
            probs = probs / np.sum(probs)
        else:
            probs = probs / probs_sum
        
        # уменьшаем exploration если есть хороший путь
        if self.best_path and len(self.success_paths) > 0:
            epsilon = max(0.05, epsilon * 0.5)
        
        if np.random.random() < epsilon:
            # взвешенный случайный выбор
            weights = np.zeros(4)
            for action in possible:
                if action == 0:
                    next_s = s - self.maze.width
                elif action == 1:
                    next_s = s + 1
                elif action == 2:
                    next_s = s + self.maze.width
                else:
                    next_s = s - 1
                
                current_dist = self.distances[s]
                next_dist = self.distances.get(next_s, current_dist)
                
                if next_dist < current_dist:
                    weights[action] = 3.0
                elif next_dist == current_dist:
                    weights[action] = 1.0
                else:
                    weights[action] = 0.3
            
            valid_weights = [weights[a] for a in possible]
            total = sum(valid_weights)
            if total > 0:
                prob_weights = [weights[a]/total for a in possible]
                action = np.random.choice(possible, p=prob_weights)
            else:
                action = np.random.choice(possible)
        else:
            # жадный выбор
            valid_probs = [probs[a] for a in possible]
            prob_sum = sum(valid_probs)
            if prob_sum > 0:
                valid_probs = [p/prob_sum for p in valid_probs]
                action = np.random.choice(possible, p=valid_probs)
            else:
                action = np.random.choice(possible)
        
        # следующее состояние
        if action == 0:
            s_next = s - self.maze.width
        elif action == 1:
            s_next = s + 1
        elif action == 2:
            s_next = s + self.maze.width
        else:
            s_next = s - 1
        
        if not self.maze.is_valid_move(s, s_next):
            s_next = s
        
        return [action, s_next]

    def generate_episode(self, pi, max_steps=300, epsilon=0.2):
        s = self.maze.start_pos
        history = [[s, np.nan]]
        states = [s]
        
        for step in range(max_steps):
            action, next_s = self.get_action(pi, s, epsilon)
            history[-1][1] = action
            history.append([next_s, np.nan])
            states.append(next_s)
            
            if next_s == self.maze.goal_pos:
                break
            
            s = next_s
        
        return history, states

    def calculate_returns(self, states, history):
        T = len(history) - 1
        returns = []
        goal_reached = states[-1] == self.maze.goal_pos
        
        for t in range(T):
            G = 0
            discount = 1.0
            
            for k in range(t, T):
                state = history[k+1][0]
                
                # базовая награда
                if state == self.maze.goal_pos:
                    reward = self.goal_reward
                else:
                    reward = self.step_penalty
                
                # бонус за прогресс
                if k > 0:
                    prev_state = history[k][0]
                    if self.distances[state] < self.distances[prev_state]:
                        reward += self.progress_bonus
                
                G += discount * reward
                discount *= self.gamma
            
            returns.append(G)
        
        return returns

    def update_theta(self, theta, pi, history, states):
        T = len(history) - 1
        returns = self.calculate_returns(states, history)
        delta = np.zeros_like(theta)
        
        goal_reached = states[-1] == self.maze.goal_pos
        multiplier = 1.0
        
        if goal_reached:
            multiplier = 1.5
            self.success_paths.append(states.copy())
            if len(self.success_paths) > 5:
                self.success_paths.pop(0)
        
        for t in range(T):
            state = history[t][0]
            action = history[t][1]
            
            if not np.isnan(action):
                grad = 1 - pi[state, action]
                update = multiplier * grad * returns[t]
                # ограничение градиента
                update = np.clip(update, -10, 10)
                delta[state, action] = update
        
        new_theta = theta + self.eta * delta
        new_theta[np.isnan(theta)] = np.nan
        
        return new_theta

    def reinforce_success_paths(self, theta):
        # дополнительное обучение на успешных путях
        if not self.success_paths:
            return theta
        
        for path in self.success_paths[-3:]:
            history = []
            for i, state in enumerate(path[:-1]):
                next_state = path[i+1]
                # определяем действие
                if next_state == state - self.maze.width:
                    action = 0
                elif next_state == state + 1:
                    action = 1
                elif next_state == state + self.maze.width:
                    action = 2
                else:
                    action = 3
                history.append([state, action])
            history.append([path[-1], np.nan])
            
            temp_pi = self.softmax(theta)
            theta = self.update_theta(theta, temp_pi, history, path)
        
        return theta

    def train(self, max_episodes=400):
        print("\nСтабильная версия")
        
        # инициализация с умеренным руководством
        theta = np.full((self.maze.size, 4), np.nan)
        for state in range(self.maze.size):
            possible = self.maze.get_possible_actions(state)
            for action in possible:
                if action == 0:
                    next_s = state - self.maze.width
                elif action == 1:
                    next_s = state + 1
                elif action == 2:
                    next_s = state + self.maze.width
                else:
                    next_s = state - 1
                
                current_dist = self.distances[state]
                next_dist = self.distances[next_s]
                
                if next_dist < current_dist:
                    weight = 3.0
                elif next_dist == current_dist:
                    weight = 1.5
                else:
                    weight = 0.5
                
                theta[state, action] = weight
        
        pi = self.softmax(theta)
        
        lengths = []
        min_dists = []
        best_length = float('inf')
        best_path = None
        
        for episode in range(max_episodes):
            epsilon = max(0.05, 0.3 * (1 - episode / max_episodes))
            history, states = self.generate_episode(pi, epsilon=epsilon)
            
            length = len(history) - 1
            min_dist = min(self.distances[s] for s in states)
            goal_reached = states[-1] == self.maze.goal_pos
            
            lengths.append(length)
            min_dists.append(min_dist)
            
            if goal_reached and length < best_length:
                best_length = length
                best_path = states.copy()
                self.best_path = best_path.copy()
            
            # основное обновление
            theta = self.update_theta(theta, pi, history, states)
            
            # дополнительное обучение на успешных путях
            if len(self.success_paths) > 0 and episode % 10 == 0:
                theta = self.reinforce_success_paths(theta)
            
            pi = self.softmax(theta)
            
            if episode % 25 == 0:
                status = "ЦЕЛЬ" if goal_reached else "БЛИЗКО" if min_dist <= 2 else ""
                print(f"Эпизод {episode:3d}: длина={length:3d}, "
                      f"мин.расст={min_dist:2d} {status}")
            
            # ранняя остановка
            if best_length <= 30 and episode > 100:
                break
        
        # финальное закрепление
        if self.success_paths:
            theta = self.reinforce_success_paths(theta)
            pi = self.softmax(theta)
        
        success_rate = len(self.success_paths) / max_episodes if max_episodes > 0 else 0
        
        print(f"\nИтоги:")
        if best_path:
            print(f"Найден путь длиной {best_length} шагов")
        else:
            print(f"Лучшее расстояние: {min(min_dists)}")
        print(f"Успешных эпизодов: {len(self.success_paths)}")
        
        return {
            'lengths': lengths,
            'min_dists': min_dists,
            'best_length': best_length,
            'best_path': best_path,
            'success_rate': success_rate,
            'pi': pi
        }