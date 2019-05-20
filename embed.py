# coding: utf-8
import torch

import numpy as np

from gensim.models import KeyedVectors
from torch.autograd import Variable

filename = 'GoogleNews-vectors-negative300.bin'
model = KeyedVectors.load_word2vec_format(filename, binary=True)


def to_torch(x, dtype='float', req = False):
  tor_type = torch.LongTensor if dtype == "int" else torch.FloatTensor
  x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
  return x


embed_dim = 300
def embed(words,maxlen):
    if(type(words)!=str):

        list = []
        list.append(np.zeros((300,)))
        return list



    words = str.split(words)
    list = []
    for word in words:

        try:
            list.append(model[word])

        except:
            pass
        if len(list)==maxlen:
            return list
    return list


def transform(sentence):
    sentence = sentence.replace("\n", " ")
    sentence = sentence.replace("\t", " ")
    sentence = sentence.replace(",", " ")
    sentence = sentence.replace(".", " ")
    sentence = sentence.replace("!", " ")
    sentence = sentence.replace("'", " ")
    return sentence

def embed_sentence(sentence,maxlen):
    #sentence = transform(sentence)
    #sentence = embed(sentence)
    seq_tensor = torch.zeros(( len(sentence),maxlen, embed_dim)).float()
    for i in range(len(sentence)):
        #print(sentence[i])
        emb = embed(transform(sentence[i]),maxlen)
        #print(len(emb))
        if len(emb)==0:
            continue
        seq_tensor[i, :len(emb), :] = to_torch(np.array(emb))
    return seq_tensor

def embed_and_pad(x,y,maxlen,dtype='float'):
    x=embed_sentence(x,maxlen)
    seq_tensor = torch.zeros((len(x), maxlen, embed_dim)).float()
    for i in range(len(x)):
        seq_tensor[i,:len(x[i]),:]=to_torch(np.array(x[i]))
    seq_lengths = torch.LongTensor([maxlen for seq in x])#torch.LongTensor([len(seq) for seq in x])
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    y = y[perm_idx]
    seq_tensor = seq_tensor.transpose(0,1)
    y = to_torch(y,dtype)
    return seq_tensor, y, seq_lengths, perm_idx
