import sys
from src.kwartz.ensemble.kwartz import Kwartz
from src.kwartz.data.loaders import load_data
from src.kwartz.evaluation.metrics import evaluate_model

def main():
    # Load the dataset
    data = load_data()

    # Initialize the Kwartz model
    model = Kwartz()

    # Evaluate the model
    accuracy = model.evaluate(data)

    # Print the evaluation results
    print(f"Kwartz Model Accuracy: {accuracy}")

if __name__ == "__main__":
    main()