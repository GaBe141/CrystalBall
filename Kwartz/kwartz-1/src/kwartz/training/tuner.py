from typing import List, Dict, Any
from src.kwartz.models.model_a import ModelA
from src.kwartz.models.model_b import ModelB
from src.kwartz.models.model_c import ModelC

class Kwartz:
    def __init__(self, models: List[Any], weights: List[float]):
        self.models = models
        self.weights = weights

    def predict(self, X: Any) -> Any:
        weighted_predictions = sum(weight * model.predict(X) for weight, model in zip(self.weights, self.models))
        return weighted_predictions

    def fit(self, X: Any, y: Any) -> None:
        for model in self.models:
            model.fit(X, y)

    def evaluate(self, X: Any, y: Any) -> Dict[str, float]:
        evaluations = {}
        for model in self.models:
            evaluations[model.__class__.__name__] = model.evaluate(X, y)
        return evaluations

    @classmethod
    def create_with_best_models(cls, X: Any, y: Any, model_classes: List[Any]) -> 'Kwartz':
        models = [model_class() for model_class in model_classes]
        for model in models:
            model.fit(X, y)
        
        accuracies = [model.evaluate(X, y)['accuracy'] for model in models]
        weights = [accuracy / sum(accuracies) for accuracy in accuracies]
        
        return cls(models, weights)