import os
import sys
import shutil
from zipfile import ZipFile
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataIngestionConfig
from hate.entity.artifact_entity import DataIngestionArtifacts

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config

    def get_data_locally(self) -> None:
        try:
            logging.info("✅ Entered get_data_locally method of DataIngestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            if not os.path.exists(self.data_ingestion_config.ZIP_FILE_PATH):
                raise FileNotFoundError(f"❌ ZIP file not found at {self.data_ingestion_config.ZIP_FILE_PATH}")

            logging.info(f"📦 ZIP file found at {self.data_ingestion_config.ZIP_FILE_PATH}")
            logging.info("✅ Exited get_data_locally method of DataIngestion class")

        except Exception as e:
            raise CustomException(e, sys) from e

    def unzip_and_clean(self):
        try:
            logging.info("📂 Entered unzip_and_clean method of DataIngestion class")

            # Extract ZIP file directly into ZIP_FILE_DIR
            with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)

            logging.info("✅ Successfully extracted ZIP file.")

            # Since CSVs are directly inside the ZIP
            raw_source = os.path.join(self.data_ingestion_config.ZIP_FILE_DIR, "raw_data.csv")
            imbalance_source = os.path.join(self.data_ingestion_config.ZIP_FILE_DIR, "imbalanced_data.csv")

            raw_target = self.data_ingestion_config.RAW_DATA_FILE_PATH
            imbalance_target = self.data_ingestion_config.IMBALANCED_DATA_FILE_PATH

            os.makedirs(os.path.dirname(raw_target), exist_ok=True)
            os.makedirs(os.path.dirname(imbalance_target), exist_ok=True)

            shutil.copy(raw_source, raw_target)
            shutil.copy(imbalance_source, imbalance_target)

            logging.info(f"✅ Copied raw_data.csv to {raw_target}")
            logging.info(f"✅ Copied imbalanced_data.csv to {imbalance_target}")
            logging.info("✅ Exited unzip_and_clean method of DataIngestion class")

            return imbalance_target, raw_target

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        try:
            logging.info("🚀 Entered initiate_data_ingestion method of DataIngestion class")
            self.get_data_locally()
            imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()

            data_ingestion_artifacts = DataIngestionArtifacts(
                imbalance_data_file_path=imbalance_data_file_path,
                raw_data_file_path=raw_data_file_path
            )

            logging.info(f"🧾 Data ingestion artifacts created: {data_ingestion_artifacts}")
            logging.info("✅ Exited initiate_data_ingestion method of DataIngestion class")
            return data_ingestion_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e
