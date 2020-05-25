# This code is adpated from the kaggle benchmark kernel with slight modification
# https://www.kaggle.com/dborkan/benchmark-kernel#Create-a-text-tokenizer

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from sklearn import model_selection
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.models import Model
import tensorflow as tf
from metrics import *
import config as cfg
from util import save_submission
import os

# Convert taget and identity columns to booleans
def convert_to_bool(df, col_name):
    df[col_name] = np.where(df[col_name] >= 0.5, True, False)

def convert_dataframe_to_bool(df, identity_columns):
    bool_df = df.copy()
    for col in ['target'] + identity_columns:
        convert_to_bool(bool_df, col)
    return bool_df

def pad_text(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)

def train_model(train_df, validate_df, tokenizer):
    # Prepare data
    train_text = pad_text(train_df[TEXT_COLUMN], tokenizer)
    train_labels = to_categorical(train_df[TOXICITY_COLUMN])
    validate_text = pad_text(validate_df[TEXT_COLUMN], tokenizer)
    validate_labels = to_categorical(validate_df[TOXICITY_COLUMN])
    # Load embeddings
    print('loading embeddings')
    embeddings_index = {}
    with open(EMBEDDINGS_PATH) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,
                                 EMBEDDINGS_DIMENSION))
    num_words_in_embedding = 0
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            num_words_in_embedding += 1
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # Create model layers.
    def get_convolutional_neural_net_layers():
        """Returns (input_layer, output_layer)"""
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedding_layer = Embedding(len(tokenizer.word_index) + 1,
                                    EMBEDDINGS_DIMENSION,
                                    weights=[embedding_matrix],
                                    input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=False)
        x = embedding_layer(sequence_input)
        x = Conv1D(128, 2, activation='relu', padding='same')(x)
        x = MaxPooling1D(5, padding='same')(x)
        x = Conv1D(128, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(5, padding='same')(x)
        x = Conv1D(128, 4, activation='relu', padding='same')(x)
        x = MaxPooling1D(40, padding='same')(x)
        x = Flatten()(x)
        x = Dropout(DROPOUT_RATE)(x)
        x = Dense(128, activation='relu')(x)
        preds = Dense(2, activation='softmax')(x)
        return sequence_input, preds

    # Compile model.
    print('compiling model')
    input_layer, output_layer = get_convolutional_neural_net_layers()
    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(lr=LEARNING_RATE),
                  metrics=['acc'])
    # Train model.
    print('training model')
    model.fit(train_text,
              train_labels,
              batch_size=BATCH_SIZE,
              epochs=NUM_EPOCHS,
              validation_data=(validate_text, validate_labels),
              verbose=2)
    return model


if __name__ == "__main__":

    # model settinngs
    TOXICITY_COLUMN = cfg.toxicity_column
    TEXT_COLUMN = cfg.text_column
    IDENTITY_COLUMN = cfg.identity_columns
    TRAIN_CSV = cfg.train_csv
    TEST_CSV = cfg.test_csv
    SUBMISSION_CSV = "submissions/kaggle_bench_mark.csv"
    MAX_NUM_WORDS = 10000
    # All comments must be truncated or padded to be the same length.
    MAX_SEQUENCE_LENGTH = cfg.max_seq_len
    EMBEDDINGS_PATH = cfg.glove['embedding_path']
    EMBEDDINGS_DIMENSION = cfg.glove['dimension']
    DROPOUT_RATE = 0.3
    LEARNING_RATE = 0.00005
    NUM_EPOCHS = 10
    BATCH_SIZE = 128

    # allow memory growth
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train = pd.read_csv(TRAIN_CSV)
    print('loaded %d records' % len(train))
    # Make sure all comment_text values are strings
    train[TEXT_COLUMN] = train[TEXT_COLUMN].astype(str)

    train = convert_dataframe_to_bool(train, IDENTITY_COLUMN)
    train_df, validate_df = model_selection.train_test_split(train, test_size=0.2)
    print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))

    # Create a text tokenizer.
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(train_df[TEXT_COLUMN])

    model = train_model(train_df, validate_df, tokenizer)
    MODEL_NAME = 'kaggle_benchmark'
    model.save_model(os.path.join("models", MODEL_NAME))

    validate_df[MODEL_NAME] = model.predict(pad_text(validate_df[TEXT_COLUMN], tokenizer))[:, 1]
    bias_metrics_df = compute_bias_metrics_for_model(validate_df, IDENTITY_COLUMN, MODEL_NAME, TOXICITY_COLUMN)
    get_final_metric(bias_metrics_df, calculate_overall_auc(validate_df, MODEL_NAME))

    test = pd.read_csv(TEST_CSV)
    predictions = model.predict(pad_text(test[TEXT_COLUMN], tokenizer))[:, 1]
    save_submission(predictions, SUBMISSION_CSV)