import os 
import sys
import pickle
import pandas as pd
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from hate.ml.model import ModelArchitecture


class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts,
                 model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config

    def spliting_data(self, csv_path):
        try:
            logging.info("📥 Reading and splitting data")
            df = pd.read_csv(csv_path, index_col=False)

            # Ensure text and label are valid
            x = df[TWEET].fillna("").astype(str)  # ensure all tweets are strings
            y = df[LABEL]

            x_train, x_test, y_train, y_test = train_test_split(
                x, y, test_size=0.3, random_state=42)
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys) from e

    def tokenizing(self, x_train):
        try:
            logging.info("🔤 Tokenizing text data")

            # Ensure all data is string (prevents .lower() on float errors)
            x_train = x_train.astype(str)

            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
            return sequences_matrix, tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info("🚀 Starting model training process")
            x_train, x_test, y_train, y_test = self.spliting_data(
                csv_path=self.data_transformation_artifacts.transformed_data_path
            )

            model_architecture = ModelArchitecture()
            model = model_architecture.get_model()

            sequences_matrix, tokenizer = self.tokenizing(x_train)

            model.fit(
                sequences_matrix, y_train,
                batch_size=self.model_trainer_config.BATCH_SIZE,
                epochs=self.model_trainer_config.EPOCH,
                validation_split=self.model_trainer_config.VALIDATION_SPLIT
            )

            # Save tokenizer
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)

            # Save model and datasets
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH, index=False)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH, index=False)
            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH, index=False)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH
            )

            logging.info("✅ Model training complete. Artifacts created.")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
