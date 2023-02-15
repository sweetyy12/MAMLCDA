#PPCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

#read
data_ = pd.read_csv(r'circdi_yes.csv') 
print(data_)
data1 = np.array(data_)
x = data1[:, 1:]  
data =scale(x)

pcaModel = PCA(n_components='mle', random_state=0)
print(pcaModel)

X_new=pcaModel.fit_transform(data)
print(X_new)

#保存
np.savetxt('circdi_PPCA.csv', X_new, delimiter=',')

