import os
import sys
import yaml
from kwartz.ensemble.kwartz import Kwartz
from kwartz.data.loaders import load_data
from kwartz.data.preprocess import preprocess_data
from kwartz.training.trainer import Trainer
from kwartz.utils.config import load_config

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), '../configs/config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Load and preprocess data
    data = load_data(config['data'])
    processed_data = preprocess_data(data)

    # Initialize Kwartz model
    kwartz_model = Kwartz(config['models'])

    # Initialize Trainer
    trainer = Trainer(kwartz_model, processed_data)

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()