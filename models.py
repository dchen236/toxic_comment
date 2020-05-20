import torch
import torch.nn as nn
from transformers import BertModel


class Toxic_Classifier_Bert(nn.Module):

    def __init__(self, freeze_bert=True):
        super(Toxic_Classifier_Bert, self).__init__()
        # Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        # Freeze bert layers
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(768, 1)

    def forward(self, seq, attn_masks):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        # Feeding the input to BERT model to obtain contextualized representations
        cont_reps, a = self.bert_layer(seq, attention_mask=attn_masks)
        # Obtaining the representation of [CLS] head
        cls_rep = cont_reps[:, 0]

        # we are doing sentence level classfication, thus using CLS to represent to whole sequence
        # alternatively, we can do max-pooling or average pooling.

        # Feeding cls_rep to the classifier layer
        output = self.cls_layer(cls_rep)

        return output