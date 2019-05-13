from embed import embed_sentence
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from model import CNN1

a = "I am the best soccer player in the world"
b = "Michael Jordan plays football"

ae = embed_sentence(a)
be = embed_sentence(b)





sim = cosine_similarity(ae,be)

X=[]
Y=[]
X.append(sim)

X=np.array(X)

model = CNN1(2)

model.learn_once(X,0)
