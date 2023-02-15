from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score

a1_label=[1,0,1,0,1,0,1,0,0,0,1,0,0,1,1,0,0,1,0,1,1,0,0,1,0,1,0,1,0,1]
labelb1_label=[1,0,1,0,1,0,1,0,0,1,1,0,0,1,1,0,0,1,0,1,1,0,0,1,0,1,0,1,0,1]
a2_label=[0,1,1,0,0,1,0,1,1,0,1,0,1,1,0,0,1,0,1,0,1,0,0,1,1,0,0,1,1,0]
labelb2_label=[0,1,1,0,0,1,0,1,1,0,1,0,1,0,0,1,1,0,1,0,1,0,0,1,1,0,0,1,1,0]

F1_1 = f1_score(labelb1_label, a1_label)
F1_2 = f1_score(labelb2_label, a2_label)
F1_k1=(F1_1+F1_2)/2.0
print("k=1_F1：",F1_k1)
MCC_1=matthews_corrcoef(labelb1_label, a1_label)  #计算MCC
MCC_2=matthews_corrcoef(labelb2_label, a2_label)
MCC_k1=(MCC_1+MCC_2)/2.0
print("k=1_MCC：",MCC_k1)
