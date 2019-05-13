from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

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

        self.conv1 = nn.Conv2d(self.ch, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)

        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)

        self.stop_criteria = stop_criteria

    def predict(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
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

