from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from embed import embed_and_pad

if torch.cuda.is_available():
  def to_torch(x, dtype, req = False):
    tor_type = torch.cuda.LongTensor if dtype == "int" else torch.cuda.FloatTensor
    x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
    return x
else:
  def to_torch(x, dtype, req = False):
    tor_type = torch.LongTensor if dtype == "int" else torch.FloatTensor
    x = Variable(torch.from_numpy(x).type(tor_type), requires_grad = req)
    return x

class CNN1(nn.Module):

    # init with (channel, height, width) and out_dim for classiication
    def __init__(self, out_dim, h,w,stop_criteria=(0.01, 1000, 120)):
        super(CNN1, self).__init__()
        self.name = "CNN1"

        self.ch = 1
        self.h=h
        self.w=w
        self.siz = 1620

        self.conv1 = nn.Conv2d(self.ch, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(1620, 50)
        self.fc2 = nn.Linear(50, out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

        self.stop_criteria = stop_criteria

    def predict(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, self.siz)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def learn_once(self, X_sub, Y_sub):
        X_sub = to_torch(X_sub, "float").view(-1, self.ch, self.h, self.w)
        Y_sub = to_torch(Y_sub, "int")

        # optimize
        self.opt.zero_grad()
        output = self.predict(X_sub)
        loss = F.nll_loss(output, Y_sub)
        loss.backward()
        self.opt.step()

        return loss

    def evaluate(self, test_corpus):
        test_data, test_label = test_corpus
        test_data = to_torch(test_data, "float").view(-1, self.ch, self.h, self.w)
        label_pred = np.argmax(self.predict(test_data).data.cpu().numpy(), axis=1)
        return np.sum(label_pred == test_label) / len(test_label)



def load_model(PATH):
    outdim = 5
    args = {
        'emb_size': 300,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.5
    }
    model = SeqLSTMClassify(args,outdim)

    model.load_state_dict(torch.load(PATH), strict=False)
    return model

class SeqLSTMClassify(nn.Module):
    def __init__(self, args, out_dim):
        super(SeqLSTMClassify,self).__init__()

        self.encoder = nn.LSTM(
            input_size=args['emb_size'],
            hidden_size=args['hidden_size'],
            num_layers=args['num_layers'],
            dropout=args['dropout'])

        self.dropout = nn.Dropout(args['dropout'])


        self.fc = nn.Linear(args['hidden_size'], out_dim)
        self.criterion = nn.MSELoss()
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)




    def forward(self, inputs, labels):
        # inputs:
        # - ids: seq len x batch, sorted in descending order by length
        #     each row: <S>, first word, ..., last word, </S>
        # - lengths: batch

        embs, lengths = inputs
        # Remove <S> from each sequence
        #embs = self.emb(ids[1:])

        enc_embs_packed = pack_padded_sequence(
            embs, lengths)

        enc_output_packed, enc_state = self.encoder(enc_embs_packed)
        enc_output, lengths = pad_packed_sequence(enc_output_packed)

        # last_enc shape: batch x emb
        last_enc = enc_output[lengths - 1, torch.arange(lengths.shape[0])]
        results = self.fc(self.dropout(last_enc))

        loss = self.criterion(results, labels)



        return enc_state, results, loss

    def learn_once(self,inputs,labels): # inputs = embs,lengths
        self.opt.zero_grad()
        enc_state, results, loss = self.forward(inputs, labels)
        loss.backward()
        self.opt.step()
        return loss

    def predict(self,inputs,labels,perm_idx):
        enc_state, results, loss = self.forward(inputs, labels)
        _, unperm_idx = perm_idx.sort(0)
        results = results[unperm_idx]
        return results.data.cpu().numpy()




def train(x,y,maxlen,EPOCH=1):
    outdim = 2
    args = {
        'emb_size': 300,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.5
    }
    BATCH_SIZE = 1
    model = SeqLSTMClassify(args,outdim)

    for epoch in range(EPOCH):
        idx = np.arange(len(x),dtype=int)
        np.random.shuffle(idx)

        for i in range(0, len(x), BATCH_SIZE):
            print(i)
            seq_tensor, y, seq_lengths, perm_idx = embed_and_pad(x[idx[i:i + BATCH_SIZE]],y[idx[i:i + BATCH_SIZE]],maxlen)
            loss = model.learn_once((seq_tensor, seq_lengths), y)
            print(loss.data.cpu().numpy())


    return model


def eval(x,y,maxlen,model):
    seq_tensor, y, seq_lengths, perm_idx = embed_and_pad(x,y,maxlen)
    loss = model.learn_once((seq_tensor, seq_lengths), y)
    print(loss.data.cpu().numpy())
    return loss.data.cpu().numpy()

def predict(df,model):
    seq_tensor, y, seq_lengths, perm_idx = embed_and_pad(df, np.arange(len(df.index)))
    return model.predict((seq_tensor, seq_lengths), y, perm_idx)
