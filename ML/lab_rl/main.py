import numpy as np
import matplotlib.pyplot as plt
import os
import time
from datetime import datetime
from maze import Maze
from reinforce_methods import Ablation1_BasicManyCycles, Ablation2_AggressivePenalties, StableREINFORCE


def setup_output_dir():
    """Создание папки для результатов"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def save_figure(fig, filename, output_dir):
    """Сохранение графика"""
    path = os.path.join(output_dir, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Сохранен: {path}")


def log_results(output_dir, results, method_names):
    """Сохранение текстовых результатов"""
    log_file = os.path.join(output_dir, "results.txt")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА\n")
        f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for i, (result, name) in enumerate(zip(results, method_names), 1):
            f.write(f"МЕТОД {i}: {name}\n")
            f.write("-"*40 + "\n")
            f.write(f"Успешность: {result.get('success_rate', 0):.1%}\n")
            f.write(f"Средняя длина: {np.mean(result.get('lengths', [0])):.1f}\n")
            
            if 'best_length' in result and result['best_length'] is not None:
                f.write(f"Лучший путь: {result['best_length']} шагов\n")
            
            if 'frozen' in result:
                f.write(f"Замерзаний: {result['frozen']}\n")
            
            if 'min_dists' in result and len(result['min_dists']) > 0:
                f.write(f"Лучшее расстояние: {min(result['min_dists'])}\n")
            
            f.write("\n")
        
        # Сравнение
        f.write("="*60 + "\n")
        f.write("СРАВНЕНИЕ МЕТОДОВ\n")
        f.write("="*60 + "\n")
        
        success_rates = [r.get('success_rate', 0) for r in results]
        avg_lengths = [np.mean(r.get('lengths', [0])) for r in results]
        
        for i, (name, success, avg_len) in enumerate(zip(method_names, success_rates, avg_lengths), 1):
            f.write(f"{i}. {name}: успешность={success:.1%}, сред.длина={avg_len:.1f}\n")
        
        # Определение лучшего метода
        if len(success_rates) > 0:
            best_idx = np.argmax(success_rates)
            f.write(f"\nЛУЧШИЙ МЕТОД: {method_names[best_idx]}\n")


def compare_methods(maze, episodes=200, output_dir="results"):
    """Сравнение трех методов с сохранением результатов"""
    
    print("\n" + "="*60)
    print("СРАВНЕНИЕ МЕТОДОВ REINFORCE")
    print("="*60)
    
    # Метод 1: базовая версия
    print("\nЗапуск метода 1...")
    method1 = Ablation1_BasicManyCycles(maze)
    result1 = method1.train(max_episodes=episodes)
    
    # Метод 2: с большими штрафами
    print("\nЗапуск метода 2...")
    method2 = Ablation2_AggressivePenalties(maze)
    result2 = method2.train(max_episodes=episodes)
    
    # Метод 3: стабильная версия
    print("\nЗапуск метода 3...")
    method3 = StableREINFORCE(maze)
    result3 = method3.train(max_episodes=episodes)
    
    # Сохранение графиков сравнения
    plot_comparison(result1, result2, result3, output_dir)
    
    return result1, result2, result3


def plot_comparison(res1, res2, res3, output_dir):
    """Графики сравнения с сохранением"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Сравнение методов REINFORCE', fontsize=16)
    
    # График 1: длины эпизодов
    axes[0, 0].plot(res1['lengths'], label='Базовая', alpha=0.7)
    axes[0, 0].plot(res2['lengths'], label='Штрафы', alpha=0.7)
    axes[0, 0].plot(res3['lengths'], label='Стабильная', alpha=0.7)
    axes[0, 0].set_title('Длины эпизодов')
    axes[0, 0].set_xlabel('Эпизод')
    axes[0, 0].set_ylabel('Длина')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # График 2: минимальные расстояния
    if 'min_dists' in res1:
        axes[0, 1].plot(res1['min_dists'], label='Базовая', alpha=0.7)
    if 'min_dists' in res2:
        axes[0, 1].plot(res2['min_dists'], label='Штрафы', alpha=0.7)
    axes[0, 1].plot(res3['min_dists'], label='Стабильная', alpha=0.7)
    axes[0, 1].set_title('Минимальные расстояния до цели')
    axes[0, 1].set_xlabel('Эпизод')
    axes[0, 1].set_ylabel('Расстояние')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # График 3: успешность
    success1 = np.cumsum(res1['goals']) / (np.arange(len(res1['goals'])) + 1)
    success2 = np.cumsum(res2['goals']) / (np.arange(len(res2['goals'])) + 1)
    if 'best_path' in res3 and res3['best_path'] is not None:
        success3 = np.ones_like(res3['lengths']) * res3['success_rate']
    else:
        success3 = np.zeros_like(res3['lengths'])
    
    axes[1, 0].plot(success1, label='Базовая', alpha=0.7)
    axes[1, 0].plot(success2, label='Штрафы', alpha=0.7)
    axes[1, 0].plot(success3, label='Стабильная', alpha=0.7)
    axes[1, 0].set_title('Успешность (скользящее среднее)')
    axes[1, 0].set_xlabel('Эпизод')
    axes[1, 0].set_ylabel('Успешность')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # График 4: сравнение итогов
    methods = ['Базовая', 'Штрафы', 'Стабильная']
    success_rates = [res1['success_rate'], res2['success_rate'], res3['success_rate']]
    avg_lengths = [np.mean(res1['lengths']), np.mean(res2['lengths']), np.mean(res3['lengths'])]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, success_rates, width, label='Успешность')
    axes[1, 1].bar(x + width/2, avg_lengths, width, label='Средняя длина')
    axes[1, 1].set_title('Итоговые метрики')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(methods)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_figure(fig, "comparison.png", output_dir)
    plt.close(fig)  # Закрываем чтобы не показывать
    
    # Сохраняем отдельные графики для каждого метода
    save_individual_plots(res1, res2, res3, output_dir)


def save_individual_plots(res1, res2, res3, output_dir):
    """Сохранение отдельных графиков для каждого метода"""
    
    # Метод 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Метод 1: Базовая версия', fontsize=14)
    
    ax1.plot(res1['lengths'])
    ax1.set_title('Длины эпизодов')
    ax1.set_xlabel('Эпизод')
    ax1.set_ylabel('Длина')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(np.cumsum(res1['goals']) / (np.arange(len(res1['goals'])) + 1))
    ax2.set_title('Успешность')
    ax2.set_xlabel('Эпизод')
    ax2.set_ylabel('Успешность')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "method1_basic.png", output_dir)
    plt.close(fig)
    
    # Метод 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Метод 2: Агрессивные штрафы', fontsize=14)
    
    ax1.plot(res2['lengths'])
    ax1.set_title('Длины эпизодов')
    ax1.set_xlabel('Эпизод')
    ax1.set_ylabel('Длина')
    ax1.grid(True, alpha=0.3)
    
    if 'min_dists' in res2:
        ax2.plot(res2['min_dists'])
        ax2.set_title('Минимальные расстояния')
        ax2.set_xlabel('Эпизод')
        ax2.set_ylabel('Расстояние')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "method2_penalties.png", output_dir)
    plt.close(fig)
    
    # Метод 3
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Метод 3: Стабильная версия', fontsize=14)
    
    ax1.plot(res3['lengths'])
    ax1.set_title('Длины эпизодов')
    ax1.set_xlabel('Эпизод')
    ax1.set_ylabel('Длина')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(res3['min_dists'])
    ax2.set_title('Минимальные расстояния')
    ax2.set_xlabel('Эпизод')
    ax2.set_ylabel('Расстояние')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, "method3_stable.png", output_dir)
    plt.close(fig)


def show_best_path(maze, result, output_dir):
    """Показать и сохранить лучший путь"""
    if result.get('best_path') is None:
        print("Путь не найден")
        return
    
    path = result['best_path']
    path_length = len(path) - 1
    
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # сетка
    for i in range(maze.width + 1):
        ax.plot([i, i], [0, maze.height], 'k-', linewidth=1, alpha=0.3)
        ax.plot([0, maze.width], [i, i], 'k-', linewidth=1, alpha=0.3)
    
    # стены
    for from_state, to_state in maze.walls:
        from_x = from_state % maze.width
        from_y = from_state // maze.width
        to_x = to_state % maze.width
        to_y = to_state // maze.width
        
        from_y_plot = maze.height - 1 - from_y
        to_y_plot = maze.height - 1 - to_y
        
        if abs(from_state - to_state) == 1:
            x = max(from_x, to_x)
            y_bottom = min(from_y_plot, to_y_plot)
            ax.plot([x, x], [y_bottom, y_bottom + 1], 'r-', linewidth=6, alpha=0.7)
        else:
            y = max(from_y_plot, to_y_plot)
            x_left = min(from_x, to_x)
            ax.plot([x_left, x_left + 1], [y, y], 'r-', linewidth=6, alpha=0.7)
    
    # путь
    path_x = [s % maze.width + 0.5 for s in path]
    path_y = [(maze.height - 1 - s // maze.width) + 0.5 for s in path]
    
    ax.plot(path_x, path_y, 'b-', linewidth=3, label=f'Путь ({path_length} шагов)')
    ax.scatter(path_x, path_y, c='blue', s=30, alpha=0.6)
    
    # старт и цель
    start_x = maze.start_pos % maze.width + 0.5
    start_y = (maze.height - 1 - maze.start_pos // maze.width) + 0.5
    goal_x = maze.goal_pos % maze.width + 0.5
    goal_y = (maze.height - 1 - maze.goal_pos // maze.width) + 0.5
    
    ax.add_patch(plt.Circle((start_x, start_y), 0.3, color='green', alpha=0.9))
    ax.add_patch(plt.Circle((goal_x, goal_y), 0.3, color='red', alpha=0.9))
    ax.text(start_x, start_y, 'Старт', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    ax.text(goal_x, goal_y, 'Цель', ha='center', va='center', color='white', fontweight='bold', fontsize=10)
    
    ax.set_xlim(0, maze.width)
    ax.set_ylim(0, maze.height)
    ax.set_aspect('equal')
    ax.set_title(f'Лучший путь: {path_length} шагов', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    save_figure(fig, "best_path.png", output_dir)
    plt.close(fig)


def save_maze_plot(maze, output_dir):
    """Сохранение изображения лабиринта"""
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # сетка
    for i in range(maze.width + 1):
        ax.plot([i, i], [0, maze.height], 'k-', linewidth=1, alpha=0.3)
        ax.plot([0, maze.width], [i, i], 'k-', linewidth=1, alpha=0.3)
    
    # стены
    for from_state, to_state in maze.walls:
        from_x = from_state % maze.width
        from_y = from_state // maze.width
        to_x = to_state % maze.width
        to_y = to_state // maze.width
        
        from_y_plot = maze.height - 1 - from_y
        to_y_plot = maze.height - 1 - to_y
        
        if abs(from_state - to_state) == 1:
            x = max(from_x, to_x)
            y_bottom = min(from_y_plot, to_y_plot)
            ax.plot([x, x], [y_bottom, y_bottom + 1], 'r-', linewidth=6)
        else:
            y = max(from_y_plot, to_y_plot)
            x_left = min(from_x, to_x)
            ax.plot([x_left, x_left + 1], [y, y], 'r-', linewidth=6)
    
    # старт и цель
    start_x = maze.start_pos % maze.width + 0.5
    start_y = (maze.height - 1 - maze.start_pos // maze.width) + 0.5
    goal_x = maze.goal_pos % maze.width + 0.5
    goal_y = (maze.height - 1 - maze.goal_pos // maze.width) + 0.5
    
    ax.add_patch(plt.Circle((start_x, start_y), 0.4, color='green', alpha=0.8))
    ax.add_patch(plt.Circle((goal_x, goal_y), 0.4, color='yellow', alpha=0.8))
    ax.text(start_x, start_y, 'S0', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(goal_x, goal_y, 'S99', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # номера состояний
    for state in range(maze.size):
        x = state % maze.width + 0.5
        y = (maze.height - 1 - state // maze.width) + 0.5
        ax.text(x, y, f'S{state}', ha='center', va='center', fontsize=6, alpha=0.7)
    
    ax.set_xlim(0, maze.width)
    ax.set_ylim(0, maze.height)
    ax.set_aspect('equal')
    ax.set_title('Лабиринт для обучения', fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    save_figure(fig, "maze.png", output_dir)
    plt.close(fig)


def main():
    # Создание папки для результатов
    output_dir = setup_output_dir()
    print(f"\nРезультаты будут сохранены в: {output_dir}")
    
    # создаем лабиринт
    maze = Maze()
    
    # Сохраняем изображение лабиринта
    save_maze_plot(maze, output_dir)
    
    # сравниваем методы
    results = compare_methods(maze, episodes=200, output_dir=output_dir)
    
    # показываем лучший путь
    print("\nСохранение лучшего пути...")
    show_best_path(maze, results[2], output_dir)
    
    # сохраняем текстовые результаты
    print("\nСохранение текстовых результатов...")
    method_names = ["Базовая версия", "Агрессивные штрафы", "Стабильная версия"]
    log_results(output_dir, results, method_names)
    
    # итоговый вывод

    print(f"Все результаты сохранены в папке: {output_dir}")
    print("\nСозданные файлы:")
    print("- maze.png                - изображение лабиринта")
    print("- comparison.png          - сравнение методов")
    print("- method1_basic.png       - результаты метода 1")
    print("- method2_penalties.png   - результаты метода 2")
    print("- method3_stable.png      - результаты метода 3")
    print("- best_path.png           - лучший найденный путь")
    print("- results.txt             - текстовые результаты")
    print("\nГотово!")


if __name__ == "__main__":
    main()