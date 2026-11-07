import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import argparse

from data_generation import generate_data, plot_data
from model import SimpleGaussianNB

def plot_decision_boundary(X, y, model, title="Decision Boundaries"):
    
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    
    colors = ['red', 'blue', 'green', 'purple']
    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'Class {class_label}',
                   alpha=0.8, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for i, class_label in enumerate(np.unique(y)):
        mask = y == class_label
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=colors[i % len(colors)], 
                   label=f'Class {class_label}',
                   alpha=0.8, edgecolors='black', linewidth=0.5)
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title("Original Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plt

def main():
    parser = argparse.ArgumentParser(description='Naive Bayes on synthetic data')
    parser.add_argument('--samples', type=int, default=300, help='Number of samples')
    parser.add_argument('--sep', type=float, default=1.0, help='Class separability')
    parser.add_argument('--test-size', type=float, default=0.3, help='Test set size')
    parser.add_argument('--save-plot', action='store_true', help='Save plot to file')
    
    args = parser.parse_args()
    
    X, y = generate_data(
        n_samples=args.samples,
        class_sep=args.sep,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    model = SimpleGaussianNB()
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"Generated data: {len(X)} samples, {len(np.unique(y))} classes")
    print(f"Training set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    print(f"Training accuracy: {train_acc:.1%}")
    print(f"Test accuracy: {test_acc:.1%}")
    
    plot = plot_decision_boundary(X, y, model, 
                                  f"Naive Bayes (separability={args.sep})")
    
    if args.save_plot:
        filename = f'naive_bayes_result_{args.samples}samples.png'
        plot.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {filename}")
if __name__ == "__main__":
    main()
