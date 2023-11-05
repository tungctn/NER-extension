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
    def __init__(self, NER_labels=NER_labels, tag2idx=tag2idx, tokenizer=tokenizer):
        self.model = AutoModelForTokenClassification.from_pretrained(
            "dbmdz/bert-large-cased-finetuned-conll03-english"
        )
        self.labels = NER_labels
        self.tokenizer = tokenizer
        self.label2idx = self.model.config.id2label
        self.model.eval()  # Make sure to set the model to evaluation mode

    # ... (phần còn lại của mã không thay đổi)

    def extract(self, text_sentence):
        tokenized_sentence = self.tokenizer.encode(text_sentence)
        input_ids = torch.tensor([tokenized_sentence])

        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = torch.argmax(outputs[0], dim=2)

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        confidences = torch.nn.functional.softmax(outputs[0], dim=2)
        marked_text = ""
        entities = []
        previous_label = 'O'
        
        # for token, prediction in zip(tokens, predictions[0].tolist()):
        for token, prediction, confidence in zip(tokens, predictions[0].tolist(), confidences[0]):
            label = self.labels[prediction]
            if label != 'O':  # Nếu nhãn không phải là 'O'
                if token.startswith("##"):
                    token = token[2:]  # Loại bỏ '##'
                    marked_text = marked_text.rstrip() + token  # Gắn vào token trước đó
                    entities[-1]["text"] += token
                else:
                    if label.startswith("B-") or previous_label != label:
                        entities.append({
                            "text": token,
                            "type": label.split("-")[-1],
                            "start": None,  # to be updated later
                            "end": None,    # to be updated later
#                             "confidence": confidence[prediction].item()
                        })
                    # If the current token is part of the previous entity, extend the entity
                    else:
                        entities[-1]["text"] += f" {token}"
                    marked_text += f" <span style='display: inline-block; padding: 5px; border: 1px solid red; background-color: #ffcccc; margin-right: 5px; border-radius: 10px;'>{token}</span>"
                if entities[-1]["start"] is None:
                    entities[-1]["start"] = text_sentence.find(entities[-1]["text"])
                entities[-1]["end"] = text_sentence.find(entities[-1]["text"]) + len(entities[-1]["text"])
                    
            else:
                if token.startswith("##"):
                    token = token[2:]  # Loại bỏ '##'
                    marked_text += token  # Gắn vào token trước đó mà không thêm khoảng trắng
                else:
                    marked_text += f" {token}"
                    
        json_output = json.dumps({"entities": entities}, indent=2)
        print(json_output)
        # Remove the special CLS and SEP tokens introduced by BERT
        marked_text = marked_text.replace('[CLS]', '').replace('[SEP]', '')
        # return marked_text.strip()
        return {
            "marked_text": marked_text.strip(),
            "entities": entities
        }

    # ... (phần còn lại của mã không thay đổi)


if __name__ == '__main__':
    text = "The earlier interventions investigating metacognitive strategies to improve math computation skills of students with LD were largely effective ,providing further evidence for successful training to improve performance in students with LD , but were relatively short in duration and taught only isolated skills . In the research presently reviewed , effective metacognitive training was implemented in the areas of self - monitoring of on - task behavior , error self - monitoring , making positive affective self - statements , self - instructional checking procedures , self - instruction of calculation procedures , and math anxiety ."
    ner = NE_Extraction()
    marked_entities = ner.extract(text)["entities"]
    print(marked_entities)



