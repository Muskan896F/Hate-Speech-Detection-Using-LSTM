import os
import sys
import shutil
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import ModelPusherConfig
from hate.entity.artifact_entity import ModelPusherArtifacts, ModelTrainerArtifacts


class ModelPusher:
    def __init__(self, model_trainer_artifacts: ModelTrainerArtifacts,
                 model_pusher_config: ModelPusherConfig):
        self.model_trainer_artifacts = model_trainer_artifacts
        self.model_pusher_config = model_pusher_config

    def initiate_model_pusher(self) -> ModelPusherArtifacts:
        try:
            logging.info("🚀 Starting model pusher...")

            os.makedirs(self.model_pusher_config.PUSHED_MODEL_DIR, exist_ok=True)

            # ✅ Push the trained model
            pushed_model_path = os.path.join(
                self.model_pusher_config.PUSHED_MODEL_DIR,
                self.model_pusher_config.MODEL_NAME
            )
            shutil.copy(
                src=self.model_trainer_artifacts.trained_model_path,
                dst=pushed_model_path
            )
            logging.info(f"✅ Model pushed to: {pushed_model_path}")

            # ✅ Push the tokenizer
            pushed_tokenizer_path = os.path.join(
                self.model_pusher_config.PUSHED_MODEL_DIR,
                "tokenizer.pickle"
            )
            shutil.copy(
                src="tokenizer.pickle",  # assuming it's saved in root after training
                dst=pushed_tokenizer_path
            )
            logging.info(f"✅ Tokenizer pushed to: {pushed_tokenizer_path}")

            return ModelPusherArtifacts(
                pushed_model_dir=self.model_pusher_config.PUSHED_MODEL_DIR,
                model_file_path=pushed_model_path
            )

        except Exception as e:
            raise CustomException(e, sys)
