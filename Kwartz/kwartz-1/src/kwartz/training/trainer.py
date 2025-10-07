class Trainer:
    def __init__(self, models, data_loader, evaluator):
        self.models = models
        self.data_loader = data_loader
        self.evaluator = evaluator

    def train(self, epochs):
        for model in self.models:
            for epoch in range(epochs):
                data = self.data_loader.load_data()
                model.train(data)
                accuracy = self.evaluator.evaluate(model, data)
                model.set_accuracy(accuracy)

    def get_best_model(self):
        best_model = max(self.models, key=lambda m: m.get_accuracy())
        return best_model

    def predict(self, input_data):
        best_model = self.get_best_model()
        return best_model.predict(input_data)