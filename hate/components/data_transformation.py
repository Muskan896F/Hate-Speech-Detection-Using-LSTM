import os
import re
import sys
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords

from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts

# Download NLTK stopwords if not already available
nltk.download('stopwords')


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts

    def imbalance_data_cleaning(self):
        try:
            logging.info("Cleaning imbalance data")
            df = pd.read_csv(self.data_ingestion_artifacts.imbalance_data_file_path)
            df.drop(
                self.data_transformation_config.ID,
                axis=self.data_transformation_config.AXIS,
                inplace=self.data_transformation_config.INPLACE
            )
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def raw_data_cleaning(self):
        try:
            logging.info("Cleaning raw data")
            df = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)

            # Drop irrelevant columns
            df.drop(
                self.data_transformation_config.DROP_COLUMNS,
                axis=self.data_transformation_config.AXIS,
                inplace=self.data_transformation_config.INPLACE
            )

            # Map values: 0 → 1, 2 → 0
            df[self.data_transformation_config.CLASS].replace({0: 1, 2: 0}, inplace=True)

            # Rename 'class' to 'label'
            df.rename(columns={self.data_transformation_config.CLASS: self.data_transformation_config.LABEL},
                      inplace=True)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def concat_dataframe(self):
        try:
            logging.info("Combining raw and imbalance datasets")
            raw_df = self.raw_data_cleaning()
            imbalance_df = self.imbalance_data_cleaning()
            final_df = pd.concat([raw_df, imbalance_df], ignore_index=True)
            return final_df
        except Exception as e:
            raise CustomException(e, sys)

    def concat_data_cleaning(self, text):
        try:
            stemmer = nltk.SnowballStemmer("english")
            stopword_set = set(stopwords.words('english'))

            text = str(text).lower()
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'https?://\S+|www\.\S+', '', text)
            text = re.sub(r'<.*?>+', '', text)
            text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub(r'\n', '', text)
            text = re.sub(r'\w*\d\w*', '', text)

            words = [word for word in text.split() if word not in stopword_set]
            stemmed = [stemmer.stem(word) for word in words]
            return " ".join(stemmed)
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        try:
            logging.info("Starting full transformation pipeline")
            df = self.concat_dataframe()

            # Clean the tweet text
            df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(
                self.concat_data_cleaning)

            # Save final cleaned dataset
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index=False)

            logging.info(f"Transformation complete, saved to {self.data_transformation_config.TRANSFORMED_FILE_PATH}")
            return DataTransformationArtifacts(
                transformed_data_path=self.data_transformation_config.TRANSFORMED_FILE_PATH
            )

        except Exception as e:
            raise CustomException(e, sys)
