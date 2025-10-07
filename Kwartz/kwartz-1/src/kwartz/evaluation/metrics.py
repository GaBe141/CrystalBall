from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metrics:
    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def calculate_precision(y_true, y_pred):
        return precision_score(y_true, y_pred, average='weighted')

    @staticmethod
    def calculate_recall(y_true, y_pred):
        return recall_score(y_true, y_pred, average='weighted')

    @staticmethod
    def calculate_f1(y_true, y_pred):
        return f1_score(y_true, y_pred, average='weighted')

    @staticmethod
    def evaluate_models(models, X, y):
        results = {}
        for model_name, model in models.items():
            y_pred = model.predict(X)
            results[model_name] = {
                'accuracy': Metrics.calculate_accuracy(y, y_pred),
                'precision': Metrics.calculate_precision(y, y_pred),
                'recall': Metrics.calculate_recall(y, y_pred),
                'f1_score': Metrics.calculate_f1(y, y_pred)
            }
        return results