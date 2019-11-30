#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
from torchtext.data import Field, TabularDataset, Iterator
from eunjeon import Mecab
tokenizer = Mecab()


# In[2]:


TEXT = Field(sequential=True,
             use_vocab=True,
             tokenize=tokenizer.morphs,  
             lower=True, 
             batch_first=True)  
LABEL = Field(sequential=False,  
              use_vocab=False,   
              preprocessing = lambda x: int(x),
              batch_first=True, 
              is_target=True)
ID = Field(sequential=False,  
           use_vocab=False,   
           is_target=False)


# In[3]:


train_data, test_data = TabularDataset.splits(
    path='./data', format='tsv', 
    train="ratings_train.txt",
    test="ratings_test.txt",
    fields=[('id', ID), ('text', TEXT), ('label', LABEL)],
    skip_header=True)


# In[4]:


TEXT.build_vocab(train_data, min_freq=2)


# In[5]:


class SentimentCls(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_size,
                 num_layers=3, batch_first=True, bidirec=True, dropout=0.5):
        super(SentimentCls, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = num_layers
        self.n_direct = 2 if bidirec else 1
        self.embedding_layer = nn.Embedding(vocab_size, embed_size)
        self.rnn_layer = nn.LSTM(input_size=embed_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=batch_first,
                                 bidirectional=bidirec,
                                 dropout=0.5)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.n_direct*hidden_size, output_size)
        

    def forward(self, x):
        embeded = self.dropout(self.embedding_layer(x))
        hidden, cell = self.init_hiddens(x.size(0), self.hidden_size, device=x.device)
        output, (hidden, cell) = self.rnn_layer(embeded, (hidden, cell))
        last_hidden = torch.cat([h for h in hidden[-self.n_direct:]], dim=1)
        scores = self.linear(last_hidden)
        return scores.view(-1)
    
    def init_hiddens(self, batch_size, hidden_size, device):
        hidden = torch.zeros(self.n_direct*self.n_layers, batch_size, hidden_size)
        cell = torch.zeros(self.n_direct*self.n_layers, batch_size, hidden_size)
        return hidden.to(device), cell.to(device)


# In[6]:


vocab_size = len(TEXT.vocab)  # the size of vocabulary
embed_size = 128  # the size of embedding
hidden_size = 256  # the size of hidden layer
output_size = 1  # the size of output layer
num_layers = 3  # the number of RNN layer
batch_first = True  # if RNN's frist dim of input is the size of minibatch
bidirec = True  # BERT
dropdout = 0.5
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # device


# In[7]:


model = SentimentCls(vocab_size, embed_size, hidden_size, output_size,
                     num_layers, batch_first, bidirec, dropdout).to(DEVICE)


# In[8]:


model.load_state_dict(torch.load("./model.pt",map_location=torch.device('cpu')))
print("Load Complete!")


# In[9]:


def test_analysis(model, field, tokenizer, device):
    sentence = input("Enter your sentence for test: ")
    x = field.process([tokenizer.morphs(sentence)]).to(device)
    output = model(x)
    pred = torch.sigmoid(output).item()
    print("--- Result ---")
    if pred > 0.8:
        print("매우좋음")
    elif pred > 0.6:
        print("좋음")
    elif pred > 0.4:
        print("보통")
    elif pred > 0.2:
        print("나쁨")
    elif pred > 0:
        print("매우나쁨")

