import unittest
from src.kwartz.evaluation.metrics import accuracy_score, precision_score, recall_score
from src.kwartz.models.model_a import ModelA
from src.kwartz.models.model_b import ModelB
from src.kwartz.models.model_c import ModelC

class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.model_a = ModelA()
        self.model_b = ModelB()
        self.model_c = ModelC()
        self.y_true = [1, 0, 1, 1, 0]
        self.y_pred_a = [1, 0, 1, 0, 0]
        self.y_pred_b = [1, 1, 1, 1, 0]
        self.y_pred_c = [0, 0, 1, 1, 0]

    def test_accuracy_score(self):
        accuracy_a = accuracy_score(self.y_true, self.y_pred_a)
        accuracy_b = accuracy_score(self.y_true, self.y_pred_b)
        accuracy_c = accuracy_score(self.y_true, self.y_pred_c)

        self.assertAlmostEqual(accuracy_a, 0.6)
        self.assertAlmostEqual(accuracy_b, 0.8)
        self.assertAlmostEqual(accuracy_c, 0.6)

    def test_precision_score(self):
        precision_a = precision_score(self.y_true, self.y_pred_a)
        precision_b = precision_score(self.y_true, self.y_pred_b)
        precision_c = precision_score(self.y_true, self.y_pred_c)

        self.assertAlmostEqual(precision_a, 0.75)
        self.assertAlmostEqual(precision_b, 0.75)
        self.assertAlmostEqual(precision_c, 0.5)

    def test_recall_score(self):
        recall_a = recall_score(self.y_true, self.y_pred_a)
        recall_b = recall_score(self.y_true, self.y_pred_b)
        recall_c = recall_score(self.y_true, self.y_pred_c)

        self.assertAlmostEqual(recall_a, 0.75)
        self.assertAlmostEqual(recall_b, 0.75)
        self.assertAlmostEqual(recall_c, 0.5)

if __name__ == '__main__':
    unittest.main()