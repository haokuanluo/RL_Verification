from embed import embed_sentence
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from model import CNN1
from model import train

a = "I am the best soccer player in the world"
b = "Michael Jordan plays football"

maxlen = 50

sent = [a,b]

emb = embed_sentence(sent,maxlen)






sim = cosine_similarity(emb[0],emb[1])

X=[]
Y=[]
X.append(sim)

X=np.array(X)

model = CNN1(2,maxlen,maxlen)

model.learn_once(X,np.array([0]))

sent = np.array([a])
train(sent,np.array([[0]]),maxlen)
