# model.py - Local model architecture definition for hate speech classification

from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding, SpatialDropout1D
from keras.optimizers import RMSprop
from hate.constants import MAX_WORDS, MAX_LEN, LOSS, METRICS, ACTIVATION

class ModelArchitecture:
    def __init__(self):
        """
        Initializes the model architecture class.
        You can extend this class to add more architectures.
        """
        pass

    def get_model(self):
        """
        Builds and compiles an LSTM-based Sequential Keras model.

        Returns:
            model (keras.Model): Compiled Keras model.
        """
        model = Sequential()
        model.add(Embedding(input_dim=MAX_WORDS, output_dim=100, input_length=MAX_LEN))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation=ACTIVATION))

        model.compile(
            loss=LOSS,
            optimizer=RMSprop(),
            metrics=METRICS
        )

        model.summary()  # You can remove this if you don't want to print architecture
        return model
