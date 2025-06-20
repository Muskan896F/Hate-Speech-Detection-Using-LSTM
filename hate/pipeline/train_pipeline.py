import os
import sys
from hate.logger import logging
from hate.exception import CustomException

from hate.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig
)

from hate.components.data_ingestion import DataIngestion
from hate.components.data_transformation import DataTransformation
from hate.components.model_trainer import ModelTrainer
from hate.components.model_evaluation import ModelEvaluation
from hate.components.model_pusher import ModelPusher

from hate.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
    ModelPusherArtifacts
)


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_pusher_config = ModelPusherConfig()

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("🚀 Starting data ingestion...")
        try:
            ingestion = DataIngestion(self.data_ingestion_config)
            return ingestion.initiate_data_ingestion()
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, ingestion_artifact: DataIngestionArtifacts) -> DataTransformationArtifacts:
        logging.info("🔄 Starting data transformation...")
        try:
            transformation = DataTransformation(
                data_transformation_config=self.data_transformation_config,
                data_ingestion_artifacts=ingestion_artifact
            )
            return transformation.initiate_data_transformation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_trainer(self, transformation_artifact: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        logging.info("🧠 Starting model training...")
        try:
            trainer = ModelTrainer(
                data_transformation_artifacts=transformation_artifact,
                model_trainer_config=self.model_trainer_config
            )
            return trainer.initiate_model_trainer()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_evaluation(self,
                               trainer_artifact: ModelTrainerArtifacts,
                               transformation_artifact: DataTransformationArtifacts) -> ModelEvaluationArtifacts:
        logging.info("📊 Starting model evaluation...")
        try:
            evaluator = ModelEvaluation(
                model_evaluation_config=self.model_evaluation_config,
                model_trainer_artifacts=trainer_artifact,
                data_transformation_artifacts=transformation_artifact
            )
            return evaluator.initiate_model_evaluation()
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_pusher(self, trainer_artifact: ModelTrainerArtifacts) -> ModelPusherArtifacts:
        logging.info("📦 Starting model pusher...")
        try:
            pusher = ModelPusher(
                model_trainer_artifacts=trainer_artifact,
                model_pusher_config=self.model_pusher_config
            )
            return pusher.initiate_model_pusher()
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        logging.info("🔥 Running full training pipeline...")
        try:
            ingestion_artifact = self.start_data_ingestion()
            transformation_artifact = self.start_data_transformation(ingestion_artifact)
            trainer_artifact = self.start_model_trainer(transformation_artifact)
            evaluation_artifact = self.start_model_evaluation(trainer_artifact, transformation_artifact)

            if not evaluation_artifact.is_model_accepted:
                logging.warning("⚠️ Trained model is not accepted. Stopping pipeline.")
                raise Exception("🚫 Trained model is not better than existing model.")

            self.start_model_pusher(trainer_artifact)
            logging.info("✅ Pipeline executed successfully.")

        except Exception as e:
            logging.error(f"❌ Error during pipeline execution: {e}")
            raise CustomException(e, sys)
