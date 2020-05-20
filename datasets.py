import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pandas as pd
import config as cfg
TRAIN_COL = cfg.text_column
LABEL_COL = cfg.toxicity_column


class toxic_Dataset_Bert(Dataset):
    def __init__(self, input, input_is_df = True, maxlen = 250):
        if not input_is_df:
        # Store the contents of the file in a pandas dataframe
            self.df = pd.read_csv(input)
        else:
            self.df = input

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.maxlen = maxlen

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Selecting the sentence and label at the specified index in the data frame
        comments = self.df[TRAIN_COL].iloc[index]
        label = self.df[LABEL_COL].iloc[index]

        # Preprocessing the text to be suitable for BERT
        tokens = self.tokenizer.tokenize(comments)  # Tokenize the sentence
        tokens = ['[CLS]'] + tokens + [
            '[SEP]']  # Insering the CLS and SEP token in the beginning and end of the sentence
        if len(tokens) < self.maxlen:
            tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))]  # Padding sentences
        else:
            tokens = tokens[:self.maxlen - 1] + ['[SEP]']  # Prunning the list to be of specified max length

        tokens_ids = self.tokenizer.convert_tokens_to_ids(
            tokens)  # Obtaining the indices of the tokens in the BERT Vocabulary
        tokens_ids_tensor = torch.tensor(tokens_ids)  # Converting the list to a pytorch tensor

        # Obtaining the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
        attn_mask = (tokens_ids_tensor != 0).long()

        return tokens_ids_tensor, attn_mask, label

