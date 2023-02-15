import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import scale
import utils.tools as utils

def get_shuffle(dataset, label):
    index = [i for i in range(len(label))]
    np.random.shuffle(index)
    dataset = dataset[index]
    label = label[index]
    return dataset, label

data_ = pd.read_csv(r'circR2D2_di.csv')  
data1 = np.array(data_)
data = data1[:, 1:]  
[m1, n1] = np.shape(data)
label1 = np.ones((int(1599), 1))  
label2 = np.zeros((int(1610), 1))
label = np.append(label2, label1)

shu = scale(data) 
X, y_ = get_shuffle(shu, label) 
y = y_
skf= StratifiedKFold(n_splits=5)

k=1 
for train, test in skf.split(X, y):
    print(X[train])
    np.savetxt(str(k) + '_circ_di_train_x_yes.csv', X[train], delimiter=',')
    print(y[train])
    np.savetxt(str(k) + '_circ_di_train_y_yes.csv', y[train], delimiter=',')
    print(X[test])
    np.savetxt(str(k) + '_circ_di_val_x_yes.csv', X[test], delimiter=',')
    print(y[test])
    np.savetxt(str(k) + '_circ_di_val_y_yes.csv', y[test], delimiter=',')
    k=k+1

