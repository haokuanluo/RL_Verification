from embed import embed_sentence
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from model import CNN1
from model import train
from inspector import gettuple
from embed import embed_and_pad
from model import SeqLSTMClassify,LSTMCNN

def trial():
    a = "I am the best soccer player in the world"
    b = "Michael Jordan plays football"

    maxlen = 50
    sent = [a, b]
    emb = embed_sentence(sent, maxlen)
    print(emb[0].shape,emb[1].shape)
    sim = cosine_similarity(emb[0], emb[1])
    print(sim.shape)
    X = []
    Y = []
    X.append(sim)

    X = np.array(X)

    model = CNN1(2, maxlen, maxlen)

    model.learn_once(X, np.array([0]))

    sent = np.array([a])
    train(sent, np.array([[0]]), maxlen)

def forreal():
    maxlen = 50
    X,Y,Z = gettuple()
    print(len(X),len(Y),len(Z))
    print('begin')
    nbatch = 64
    for i in range(0,len(X),nbatch):
        up = min(len(X),i+nbatch)
        emb1 = embed_sentence(X[i:up],maxlen)
        emb2 = embed_sentence(Y[i:up],maxlen)


        sim = np.array([cosine_similarity(emb1[i],emb2[i]) for i in range(0,len(emb1))])
        model = CNN1(3, maxlen, maxlen)
        model.learn_once(sim,Z[i:up])
        print(model.evaluate((sim,Z[i:up])))
        print(i,len(X))

def lstm():
    maxlen = 50
    X, Y, Z = gettuple()
    print(len(X), len(Y), len(Z))
    print('begin')
    nbatch = 64
    outdim = 3
    args = {
        'emb_size': 300,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.5
    }
    BATCH_SIZE = 1

    model = LSTMCNN(args, outdim)#SeqLSTMClassify(args, outdim)

    for i in range(0, len(X), nbatch):
        up = min(len(X), i + nbatch)

        seq_tensor1, y1, seq_lengths1, perm_idx1 = embed_and_pad(X[i:up], Z[i:up],
                                                             maxlen,'int')
        seq_tensor2, y2, seq_lengths2, perm_idx2 = embed_and_pad(Y[i:up], Z[i:up],
                                                                 maxlen,'int')

        loss = model.learn_once(((seq_tensor1, seq_lengths1),(seq_tensor2, seq_lengths2)), y1)
        print(loss.data.cpu().numpy())





if __name__ == '__main__':
    #trial() # (a,b) (c,b) = (a,c)
    lstm()

