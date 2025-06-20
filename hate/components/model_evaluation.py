import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from keras.utils import pad_sequences
from sklearn.metrics import confusion_matrix
from hate.constants import *
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import (
    ModelEvaluationArtifacts,
    ModelTrainerArtifacts,
    DataTransformationArtifacts
)


class ModelEvaluation:
    def __init__(
        self,
        model_evaluation_config: ModelEvaluationConfig,
        model_trainer_artifacts: ModelTrainerArtifacts,
        data_transformation_artifacts: DataTransformationArtifacts
    ):
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts

    def evaluate_model(self, model, tokenizer, x_test, y_test):
        try:
            logging.info("✅ Evaluating model...")

            # Ensure x_test is a Series of strings
            if isinstance(x_test, pd.DataFrame):
                if x_test.shape[1] == 0:
                    raise ValueError("❌ x_test DataFrame has no columns.")
                x_test = x_test.iloc[:, 0]  # Take first column if it's a DataFrame

            x_test = x_test.fillna("").astype(str)
            y_test = y_test.squeeze()

            # Tokenize and pad test text
            sequences = tokenizer.texts_to_sequences(x_test)
            padded = pad_sequences(sequences, maxlen=MAX_LEN)

            # Evaluate model
            loss, accuracy = model.evaluate(padded, y_test, verbose=0)
            predictions = model.predict(padded)
            pred_labels = [1 if p[0] >= 0.5 else 0 for p in predictions]

            # Log confusion matrix
            cm = confusion_matrix(y_test, pred_labels)
            logging.info(f"📊 Confusion Matrix:\n{cm}")
            return accuracy

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        try:
            logging.info("🚀 Starting model evaluation...")

            # Load trained model and tokenizer
            trained_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            # Load test datasets
            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path)

            trained_model_accuracy = self.evaluate_model(trained_model, tokenizer, x_test, y_test)

            best_model_path = os.path.join(
                self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                self.model_evaluation_config.MODEL_NAME
            )

            # First time training
            if not os.path.isfile(best_model_path):
                logging.info("🆕 No previous best model found. Accepting current model.")
                is_model_accepted = True
            else:
                logging.info("🔁 Comparing with best model...")
                best_model = keras.models.load_model(best_model_path)
                best_model_accuracy = self.evaluate_model(best_model, tokenizer, x_test, y_test)
                is_model_accepted = trained_model_accuracy > best_model_accuracy
                logging.info(f"✅ Model accepted: {is_model_accepted}")

            return ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)

        except Exception as e:
            raise CustomException(e, sys)
