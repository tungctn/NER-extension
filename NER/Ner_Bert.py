import joblib
import os
import torch
import sys
import os
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import pickle
from tqdm import tqdm, trange
import collections
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
from pytorch_pretrained_bert.tokenization import BertTokenizer
import transformers
from transformers import BertForTokenClassification, AdamW
from transformers import BertTokenizer, AutoTokenizer
from transformers import AutoModelForTokenClassification, AutoTokenizer
import json
NER_labels = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
tag2idx = {t: i for i, t in enumerate(NER_labels)}
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english", do_lower_case=False)
class NE_Extraction:
    def __init__(self,NER_labels,tag2idx,tokenizer = tokenizer):
        # self.model = BertForTokenClassification.from_pretrained(
        #     "bert-base-cased",
        #     num_labels=len(tag2idx),
        #     output_attentions = False,
        #     output_hidden_states = False
        # )
        self.model = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/bert-large-cased-finetuned-conll03-english"
        )
        self.labels = NER_labels
        self.tokenizer = tokenizer  
        self.label2idx = tag2idx
    def extract(self, text_sentence):
        tokenized_sentence = self.tokenizer.encode(text_sentence)
        input_ids = torch.tensor([tokenized_sentence])
        
        with torch.no_grad():
            output = self.model(input_ids)
        
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        print(label_indices)
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
        new_tokens, new_labels = [], []
        
        for token, label_idx in zip(tokens, label_indices[0]):
            if token.startswith("##"):
                new_tokens[-1] = new_tokens[-1] + token[2:]
            else:
                new_labels.append(self.labels[label_idx])
                new_tokens.append(token)
        Per = []
        Org = []
        Loc = []
        Misc = []
        # label BIo
        for token, label in zip(new_tokens, new_labels):
            if label == 'B-PER' or label == 'I-PER':
                Per.append(token)
            if label == 'B-ORG' or label == 'I-ORG':
                Org.append(token)
            if label == 'B-LOC' or label == 'I-LOC':
                Loc.append(token)
            if label == 'B-MISC' or label == 'I-MISC':
                Misc.append(token)
        entities = {
        'PER': Per,
        'ORG': Org,
        'LOC': Loc,
        'MISC': Misc
    }
        marked_text = ""
        for token, label_idx in zip(new_tokens, label_indices[0]):
            if token in ("[CLS]", "[SEP]"):
                continue
            if label_idx != 0:  # Nếu nhãn không phải là 'O'
                marked_text += f"<span style='display: inline-block; padding: 5px; border: 1px solid red; background-color: #ffcccc;'>{token}</span>"
            else:
                marked_text += f"{token} "

        # print(marked_text.strip())
    
    # Convert the dictionary to a JSON string
        # json_output = json.dumps(entities)
        # print(label_indices)
        return marked_text.strip()
if __name__ == '__main__':
    text = "My name is John and I live in New York"
    ner = NE_Extraction(NER_labels,tag2idx)
    print(ner.extract(text))


