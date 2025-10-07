def validate_model_performance(models, validation_data):
    """
    Validates the performance of the given models on the validation data.
    
    Parameters:
    models (list): A list of model instances to validate.
    validation_data (tuple): A tuple containing features and labels for validation.
    
    Returns:
    dict: A dictionary with model names as keys and their performance metrics as values.
    """
    results = {}
    features, labels = validation_data
    
    for model in models:
        model_name = type(model).__name__
        predictions = model.predict(features)
        accuracy = calculate_accuracy(predictions, labels)
        results[model_name] = accuracy
    
    return results

def calculate_accuracy(predictions, labels):
    """
    Calculates the accuracy of predictions against the true labels.
    
    Parameters:
    predictions (array-like): The predicted labels.
    labels (array-like): The true labels.
    
    Returns:
    float: The accuracy as a percentage.
    """
    correct_predictions = sum(pred == true for pred, true in zip(predictions, labels))
    return correct_predictions / len(labels) * 100 if labels else 0.0