import pickle
import numpy as np


def gettuple():
    path = 'pickled_datasets/fever/train[format=dict(statement,candidate,gold)].pickle'

    d = pickle.load(open(path, 'rb'))
    X = []
    Y = []
    Z = []
    it = 0
    for i in d:
        it = it+1
        if it>10000:
            break
        for j in range(len(i['candidate'])):
            X.append(i['statement'])
            Y.append(i['candidate'][j])
            if i['gold'][j] == 'SUPPORTS':
                Z.append(0)
            elif i['gold'][j] == 'NOT ENOUGH INFO':
                Z.append(1)
            elif i['gold'][j] == 'REFUTES':
                Z.append(2)
            else:
                print(i['gold'][j])

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    return (X,Y,Z)


