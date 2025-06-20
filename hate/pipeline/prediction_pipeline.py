import os
import sys
import keras
import pickle
import re
from hate.logger import logging
from hate.constants import MODEL_NAME
from hate.exception import CustomException
from keras.utils import pad_sequences


class PredictionPipeline:
    def __init__(self):
        try:
            # ✅ Correct path to model and tokenizer
            self.model_path = os.path.join("artifacts", "PredictModel", MODEL_NAME)
            self.tokenizer_path = os.path.join("artifacts", "PredictModel", "tokenizer.pickle")

            # ✅ Load model
            self.model = keras.models.load_model(self.model_path)

            # ✅ Load tokenizer
            with open(self.tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)

        except Exception as e:
            raise CustomException(f"❌ Error loading model/tokenizer: {e}", sys)

    def clean_text(self, text: str) -> str:
        """
        Lightweight cleaning similar to training pipeline logic.
        """
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
