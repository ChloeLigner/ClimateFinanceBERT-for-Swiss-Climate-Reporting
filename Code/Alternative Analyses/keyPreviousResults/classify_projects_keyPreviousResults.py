import os
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import transformers


from transformers import AutoModelForSequenceClassification, AutoTokenizer

""" This file imports the ClimateFinanceBERT classifiers to predict climate relevance and categories for the crs dataset"""

pd.set_option('display.width', 10000)
pd.set_option('display.max_columns', 10)
# specify GPU
device = torch.device("cpu")

#import data
dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",".."))
print(dir_path)


# Import the labeling dictionary
import json
###FOr importing the dictionaries
def load_dict(filename):
    with open(filename) as f:
        file = json.loads(f.read())
    return file
label_dict = load_dict(dir_path + "/Code/ClimateFinanceBERT/Code/Multiclass/dictionary_classes.txt")
print(label_dict)


"""MODEL """
base_model = 'climatebert/distilroberta-base-climate-f'
# import BERT-base pretrained model
relevance_classifier = AutoModelForSequenceClassification.from_pretrained(base_model,
                                                              num_labels=2,
                                                                )

multiclass = AutoModelForSequenceClassification.from_pretrained(base_model,
                                                              num_labels=len(label_dict),
                                                                )

# Load the BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)


class BERT_Arch(nn.Module):

    def __init__(self, model):
        super().__init__()

        self.bert = model

        # softmax activation function
        self.softmax = nn.LogSoftmax(dim=1)

    #define the forward pass
    def forward(self, sent_id, mask):

      #pass the inputs to the model
      output = self.bert(sent_id, attention_mask=mask, return_dict=False)

      # apply softmax activation
      x = self.softmax(output[0])
      return x


#initialize relevance_classifier
relevance_classifier = BERT_Arch(relevance_classifier)
#Load trained weights
relevance_classifier.load_state_dict(torch.load(
    dir_path + "/Code/saved_weight/saved_weights_relevance.pt", map_location=device))

#initialize multiclass_model
multiclass = BERT_Arch(multiclass)
#Load trained weights
multiclass.load_state_dict(torch.load(
    dir_path + "/Code/saved_weight/saved_weights_multiclass.pt", map_location=device)) 


def tokenize(sentence):
    """Tokenization"""
    valid_sentences = [s for s in sentence if isinstance(s, str) and s.strip() != '']

    token = tokenizer.batch_encode_plus(
        [s if isinstance(s, str) and s.strip() != '' else "This is a filler text for invalid input." for s in sentence],
        max_length=150,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    test_seq = token['input_ids']
    test_mask = token['attention_mask']
    return test_seq, test_mask


"""Import the data in chunks"""

def csv_import(name, delimiter="|", chunksize=10 ** 2):
    x = pd.read_csv(name, encoding='utf8', low_memory=False, delimiter=delimiter, chunksize=chunksize,
                    dtype={'keyPreviousResults': str,
                           "CH_Disbursement": float
                           }
                    )
    return x


chunk_list = []
i=0

# Process only the first chunk
for chunk in csv_import(dir_path + "/Data/Alternatives/df_preprocessed_keyPreviousResults.csv"):

    # Initialize default for climate_relevance
    original_length = chunk.shape[0]
    chunk['climate_relevance'] = 0

    with torch.no_grad():
        # Tokenize for relevance classification
        text_seq, text_mask = tokenize(chunk['keyPreviousResults'].tolist())

        # Get predictions for relevance
        relevance_preds = relevance_classifier(text_seq, text_mask)
        relevance_preds = relevance_preds.detach().numpy()

        # Get predicted relevance
        pred_relevance = np.argmax(relevance_preds, axis=1)
        chunk['climate_relevance'] = pred_relevance

        if len(pred_relevance) == original_length:
            # Assign predictions to the chunk
            chunk['climate_relevance'] = pred_relevance
        else:
            # Handle mismatch: Fill with default for invalid entries
            chunk['climate_relevance'][:len(pred_relevance)] = pred_relevance

    # Print results for relevance
    print("Testing Results for Relevance Classification:")
    print(chunk[['keyPreviousResults', 'climate_relevance']])

    print('number of documents:', chunk.keyPreviousResults.shape[0])
    print('number of relevant documents:', chunk.keyPreviousResults[chunk.climate_relevance == 1].shape[0])
    
    # Mask all texts that are relevant
    chunk['climate_class_number'] = 500

    relevant_texts = chunk[chunk['climate_relevance'] == 1]
    valid_texts = relevant_texts['keyPreviousResults'].dropna().tolist()  # Remove None values

    if valid_texts:  # Check if there are valid texts
        text_seq, text_mask = tokenize(valid_texts)

        with torch.no_grad():
            preds = multiclass(text_seq, text_mask)
            preds = preds.detach().numpy()
        
        pred_class = np.argmax(preds, axis=1)
        chunk.loc[chunk['climate_relevance'] == 1, 'climate_class_number'] = pred_class
        chunk['climate_class'] = chunk['climate_class_number'].astype(str).replace(label_dict)
    else:
        chunk['climate_class'] = '500'

    # Print the results for multiclass classification
    print("Testing Results for Multiclass Classification:")
    print(chunk[['keyPreviousResults', 'climate_relevance', 'climate_class']])

    chunk_list.append(chunk)
    # Append chunk to list
    i += 1
    print(i)

    del chunk

# Concatenate all chunks into a single DataFrame
df = pd.concat(chunk_list)

df.to_csv(dir_path + "/Data/Alternatives/climate_finance_total_keyPreviousResults.csv", encoding='utf8', index=False, header=True, sep='|')
