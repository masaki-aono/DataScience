#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import Counter
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
import json
from torch.nn import BCEWithLogitsLoss, BCELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, multilabel_confusion_matrix, f1_score, accuracy_score
import pickle
from transformers import *
from tqdm import tqdm, trange
from ast import literal_eval


# In[2]:


device_name = tf.test.gpu_device_name()
print('Found GPU at: {}'.format(device_name))


# In[3]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)


# In[4]:


import os
os.environ["NLTK_DATA"] = os.path.join(os.getcwd(), "nltk_data")

# In[5]:

from nltk.corpus.util import LazyCorpusLoader
from nltk.corpus.reader import *

# Loading the corpus
ma_reuters = LazyCorpusLoader(
    'ma_reuters', CategorizedPlaintextCorpusReader, '(training|test).*',
    cat_file='cats.txt', encoding='ISO-8859-2')

# Load MA_Reuters
documents = ma_reuters.fileids()
print (str(len(documents)) + " total articles")
# extracting training and testing data (document ID)
train_docs_id = list(filter(lambda doc: doc.startswith("train"), documents))
test_docs_id = list(filter(lambda doc: doc.startswith("test"), documents))
print (str(len(train_docs_id)) + " training data")
print (str(len(test_docs_id)) + " testing data")
# Training and testing data
train_docs = [ma_reuters.raw(doc_id) for doc_id in train_docs_id]
test_docs = [ma_reuters.raw(doc_id) for doc_id in test_docs_id]
    
# print the total number of categories
categories = ma_reuters.categories()
num_categories = len(categories)
print (num_categories, " categories")
print (categories)


# In[24]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform([ma_reuters.categories(doc_id) for doc_id in train_docs_id])
test_labels = mlb.transform([ma_reuters.categories(doc_id) for doc_id in test_docs_id])


# In[7]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
encodings = tokenizer.batch_encode_plus(train_docs, pad_to_max_length=True, truncation=True)
print('tokenizer outputs: ', encodings.keys())


# In[8]:


input_ids = encodings['input_ids']
token_type_ids = encodings['token_type_ids']
attention_masks = encodings['attention_mask']
labels = train_labels[:]


# In[9]:


def list2byte(l):
    d = l[-1].item()
    for x in l[-2::-1]:
        d <<= 1
        d += x.item()
    return d

def filter_idx(l, idxs):
    return [v for i,v in enumerate(l) if i not in idxs]

label_bitsets = list(map(list2byte, labels))
label_counts = Counter(label_bitsets)

one_freq = set(k for k, v in label_counts.items() if v == 1)
one_freq_idxs = [i for i, v in enumerate(label_bitsets) if v in one_freq]
label_bitsets = None


# In[10]:


one_freq_input_ids = [input_ids[i] for i in one_freq_idxs]
one_freq_token_types = [token_type_ids[i] for i in one_freq_idxs]
one_freq_attention_masks = [attention_masks[i] for i in one_freq_idxs]
one_freq_labels = [labels[i] for i in one_freq_idxs]


# In[11]:


filtered_labels = filter_idx(labels, one_freq_idxs)
t = train_test_split(filter_idx(input_ids, one_freq_idxs),
                     filtered_labels,
                     filter_idx(token_type_ids, one_freq_idxs),
                     filter_idx(attention_masks, one_freq_idxs),
                     random_state=2020,
                     test_size=0.10,
                     stratify=filtered_labels)
train_inputs, validation_inputs, train_labels, validation_labels, train_token_types, validation_token_types, train_masks, validation_masks = t

filtered_labels = None

train_inputs.extend(one_freq_input_ids)
train_labels.extend(one_freq_labels)
train_masks.extend(one_freq_attention_masks)
train_token_types.extend(one_freq_token_types)

train_inputs = torch.tensor(train_inputs)
train_labels = torch.tensor(train_labels)
train_masks = torch.tensor(train_masks)
train_token_types = torch.tensor(train_token_types)

validation_inputs = torch.tensor(validation_inputs)
validation_labels = torch.tensor(validation_labels)
validation_masks = torch.tensor(validation_masks)
validation_token_types = torch.tensor(validation_token_types)


# In[12]:


batch_size = 16

train_data = TensorDataset(train_inputs, train_masks, train_labels, train_token_types)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels, validation_token_types)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)


# In[13]:


torch.save(validation_dataloader,'validation_data_loader')
torch.save(train_dataloader,'train_data_loader')


# In[14]:


model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_categories)
model.cuda()

model.load_state_dict(torch.load('bert_model_reuters_epoch_19')) 

encodings = tokenizer.batch_encode_plus(test_docs, pad_to_max_length=True, truncation=True)
test_input_ids = encodings['input_ids']
test_token_type_ids = encodings['token_type_ids']
test_attention_masks = encodings['attention_mask']


# In[37]:


test_inputs = torch.tensor(test_input_ids)
test_labels_tensor = torch.tensor(test_labels)
test_masks = torch.tensor(test_attention_masks)
test_token_types = torch.tensor(test_token_type_ids)

test_data = TensorDataset(test_inputs, test_masks, test_labels_tensor, test_token_types)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

# In[38]:


# Test

# Put model in evaluation mode to evaluate loss on the validation set
model.eval()

#track variables
logit_preds,true_labels,pred_labels,tokenized_texts = [],[],[],[]

# Predict
for i, batch in enumerate(test_dataloader):
  batch = tuple(t.to(device) for t in batch)
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels, b_token_types = batch
  with torch.no_grad():
    # Forward pass
    outs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
    b_logit_pred = outs[0]
    pred_label = torch.sigmoid(b_logit_pred)

    b_logit_pred = b_logit_pred.detach().cpu().numpy()
    pred_label = pred_label.to('cpu').numpy()
    b_labels = b_labels.to('cpu').numpy()

  tokenized_texts.append(b_input_ids)
  logit_preds.append(b_logit_pred)
  true_labels.append(b_labels)
  pred_labels.append(pred_label)

# Flatten outputs
tokenized_texts = [item for sublist in tokenized_texts for item in sublist]
pred_labels = [item for sublist in pred_labels for item in sublist]
true_labels = [item for sublist in true_labels for item in sublist]
# Converting flattened binary values to boolean values
true_bools = [tl==1 for tl in true_labels]


# In[42]:

import numpy as np
# Jaccard coefficient
from sklearn.metrics import jaccard_score

pred_bools = [[1 if x > 0.50 else 0 for x in pl] for pl in pred_labels]

with open("bert_predicted.json", "w") as f:
  json.dump(pred_bools, f)

print ("Jaccard coef:",np.round(jaccard_score(test_labels, pred_bools, average='samples'),3))
