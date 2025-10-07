import unittest
from src.kwartz.ensemble.kwartz import Kwartz
from src.kwartz.models.model_a import ModelA
from src.kwartz.models.model_b import ModelB
from src.kwartz.models.model_c import ModelC

class TestKwartz(unittest.TestCase):

    def setUp(self):
        self.model_a = ModelA()
        self.model_b = ModelB()
        self.model_c = ModelC()
        self.kwartz_model = Kwartz(models=[self.model_a, self.model_b, self.model_c])

    def test_model_initialization(self):
        self.assertIsNotNone(self.kwartz_model)
        self.assertEqual(len(self.kwartz_model.models), 3)

    def test_model_accuracy_weighting(self):
        # Assuming each model has a method to get its accuracy
        self.model_a.accuracy = 0.8
        self.model_b.accuracy = 0.9
        self.model_c.accuracy = 0.85
        
        weights = self.kwartz_model.calculate_weights()
        self.assertAlmostEqual(weights[self.model_a], 0.25, places=2)
        self.assertAlmostEqual(weights[self.model_b], 0.45, places=2)
        self.assertAlmostEqual(weights[self.model_c], 0.30, places=2)

    def test_model_prediction(self):
        # Mocking the predict method for each model
        self.model_a.predict = lambda x: "A"
        self.model_b.predict = lambda x: "B"
        self.model_c.predict = lambda x: "C"
        
        prediction = self.kwartz_model.predict("input_data")
        self.assertIn(prediction, ["A", "B", "C"])

if __name__ == '__main__':
    unittest.main()