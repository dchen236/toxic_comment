from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from metrics import *
import config as cfg
import argparse


from util import save_submission
import os
MAX_SEQUENCE_LENGTH = cfg.max_seq_len
MAX_NUM_WORDS = 10000

def pad_text(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)


def prediction(model, df, TEXT_COLUMN):
    # Create a text tokenizer.
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(df[TEXT_COLUMN])
    return model.predict(pad_text(df[TEXT_COLUMN], tokenizer))[:, 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', dest='QUERY_CSV', action='store',
                        help='csv file with comments')
    parser.add_argument('--comment_col', dest='TEXT_COLUMN', action='store',
                        help='comment column name')
    args = parser.parse_args()
    query_df = pd.read_csv(args.QUERY_CSV)
    # allow memory growth
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = keras.models.load_model('models/kaggle_benchmark')
    query_df['toxicity'] = prediction(model, query_df, args.TEXT_COLUMN)
    save_name = "predicted_" + args.QUERY_CSV
    query_df.to_csv(save_name, index = False)
    print("Finished prediction, saved file as " + save_name)