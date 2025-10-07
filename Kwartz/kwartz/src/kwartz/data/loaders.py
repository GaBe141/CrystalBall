def load_dataset(file_path):
    # Function to load a dataset from a given file path
    import pandas as pd
    return pd.read_csv(file_path)

def load_multiple_datasets(file_paths):
    # Function to load multiple datasets from a list of file paths
    datasets = {}
    for path in file_paths:
        datasets[path] = load_dataset(path)
    return datasets

def load_and_preprocess_dataset(file_path, preprocess_function):
    # Function to load a dataset and apply a preprocessing function
    dataset = load_dataset(file_path)
    return preprocess_function(dataset)