import numpy as np
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

def generate_data(n_samples=300, n_features=2, n_classes=2, 
                  random_state=42, class_sep=1.0):
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        n_informative=n_features,
        n_clusters_per_class=1,
        n_classes=n_classes,
        random_state=random_state,
        class_sep=class_sep,
        flip_y=0.05
    )
    
    return X, y

def plot_data(X, y, title="Generated data"):
    
    plt.figure(figsize=(8, 6))
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'Class {class_label}',
                   alpha=0.6)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt

if __name__ == "__main__":
    X, y = generate_data()
    plot_data(X, y)
    plt.show()
