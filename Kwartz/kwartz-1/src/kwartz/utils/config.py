def get_config():
    config = {
        "model_weights": {
            "model_a": 0.33,
            "model_b": 0.33,
            "model_c": 0.34
        },
        "data": {
            "batch_size": 32,
            "num_epochs": 100,
            "learning_rate": 0.001
        },
        "logging": {
            "log_level": "INFO",
            "log_file": "kwartz.log"
        }
    }
    return config

def save_config(config, filepath):
    import yaml
    with open(filepath, 'w') as file:
        yaml.dump(config, file)

def load_config(filepath):
    import yaml
    with open(filepath, 'r') as file:
        return yaml.safe_load(file)