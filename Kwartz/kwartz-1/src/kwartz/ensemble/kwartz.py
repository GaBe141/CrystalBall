from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import numpy as np

class Kwartz(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models
        self.weights = None

    def fit(self, X, y):
        # Train each model and calculate accuracy
        accuracies = []
        for model in self.models:
            model.fit(X, y)
            predictions = model.predict(X)
            accuracy = accuracy_score(y, predictions)
            accuracies.append(accuracy)

        # Normalize accuracies to get weights
        self.weights = np.array(accuracies) / np.sum(accuracies)

    def predict(self, X):
        # Get predictions from each model
        predictions = np.array([model.predict(X) for model in self.models])
        
        # Weighted voting
        weighted_predictions = np.tensordot(self.weights, predictions, axes=(0, 0))
        return np.argmax(weighted_predictions, axis=0)

    def predict_proba(self, X):
        # Get probabilities from each model
        probabilities = np.array([model.predict_proba(X) for model in self.models])
        
        # Weighted probabilities
        weighted_probabilities = np.tensordot(self.weights, probabilities, axes=(0, 0))
        return weighted_probabilities / np.sum(weighted_probabilities, axis=1, keepdims=True)