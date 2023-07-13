from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import sys


if __name__ == '__main__':
    ingestion_config = DataIngestion().ingestion()
    train_path, test_path = DataTransformation(ingestion_config=ingestion_config).transform()
    trainer = ModelTrainer(train_path, test_path)
    trainer.train()
    
