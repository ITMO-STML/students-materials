import numpy as np

class SimpleGaussianNB:
    
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.class_means = {}
        self.class_vars = {}
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_features = X.shape
        
        for c in self.classes:
            X_c = X[y == c]
            self.class_priors[c] = len(X_c) / n_samples
            self.class_means[c] = np.mean(X_c, axis=0)
            self.class_vars[c] = np.var(X_c, axis=0) + 1e-9
            
        return self
    
    def _gaussian_pdf(self, x, mean, var):
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_proba = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes):
            log_prior = np.log(self.class_priors[c])
            log_likelihood = np.sum(
                np.log(self._gaussian_pdf(X, self.class_means[c], self.class_vars[c])),
                axis=1
            )
            log_proba[:, i] = log_prior + log_likelihood
        
        log_proba = log_proba - np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba = proba / np.sum(proba, axis=1, keepdims=True)
        
        return proba
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
