import os
import sys
import keras
import pickle
import re
from hate.logger import logging
from hate.exception import CustomException
from keras.utils import pad_sequences


class PredictionPipeline:
    def __init__(self):
        try:
            # 🔍 Find the latest timestamped artifact folder
            base_artifact_path = os.path.join("artifacts")
            all_folders = [f for f in os.listdir(base_artifact_path) if os.path.isdir(os.path.join(base_artifact_path, f))]
            latest_folder = sorted(all_folders)[-1]  # Last one (latest)

            # ✅ Build absolute path to pushed_model inside latest folder
            pushed_model_dir = os.path.join(base_artifact_path, latest_folder, "pushed_model")
            model_path = os.path.join(pushed_model_dir, "model.h5")
            tokenizer_path = os.path.join(pushed_model_dir, "tokenizer.pickle")

            # 🔒 Check if files exist before loading
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")

            self.model = keras.models.load_model(model_path)
            with open(tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)

        except Exception as e:
            raise CustomException(f"❌ Error loading model/tokenizer: {e}", sys)

    def clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        text = re.sub(r"\@\w+|\#", '', text)
        text = re.sub(r"[^A-Za-z0-9\s]+", '', text)
        return text.strip()

    def predict(self, text: str) -> str:
        try:
            logging.info("🔍 Running prediction...")

            cleaned_text = self.clean_text(text)
            seq = self.tokenizer.texts_to_sequences([cleaned_text])
            padded = pad_sequences(seq, maxlen=300)

            pred = self.model.predict(padded)[0][0]
            logging.info(f"✅ Prediction value: {pred}")

            return "hate and abusive" if pred > 0.5 else "no hate"
        except Exception as e:
            raise CustomException(f"❌ Prediction failed: {e}", sys)

    def run_pipeline(self, text: str) -> str:
        try:
            return self.predict(text)
        except Exception as e:
            raise CustomException(e, sys)
