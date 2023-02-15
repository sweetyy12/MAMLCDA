#datadeal
import xlrd
import numpy
import numpy as np
import math
import numpy.linalg as LA
from sklearn.cluster import KMeans
import random
from sklearn import tree
a=open(r'.\circR2D_data_yes\known_association_matrix_yes.xlsx') 
g=open(r'.\circR2D_data_yes\CCSimi_yes.xlsx')
h=open(r'.\circR2D_data_yes\DDSimi_yes.xlsx')
A=numpy.zeros((1271,144))   
RCS=numpy.zeros((1271,1271))
RDS=numpy.zeros((144,144))

xlsx1=xlrd.open_workbook(r'.\circR2D_data_yes\known_association_matrix_yes.xlsx')
sheet1=xlsx1.sheets()[0] 
for i in range(1271):
    for j in range(144):
        s1=sheet1.row_values(i)   
        m1=s1[j]                             
        A[i,j]=m1 
A=A.T    
xlsx2=xlrd.open_workbook(r'.\circR2D_data_yes\CCSimi_yes.xlsx')
sheet2=xlsx2.sheets()[0] 
for i in range(1271):
    for j in range(1271):
        s2=sheet2.row_values(i)  
        m2=s2[j]                             
        RCS[i,j]=m2 

xlsx3=xlrd.open_workbook(r'.\circR2D_data_yes\DDSimi_yes.xlsx')
sheet3=xlsx3.sheets()[0] 
for i in range(144):
    for j in range(144):
        s3=sheet3.row_values(i)   
        m3=s3[j]                             
        RDS[i,j]=m3 

#K_mean clustering
unknown=[]                          
known=[]                            
for x in range(144):                 
    for y in range(1271):
        if A[x,y]==0:                 
            unknown.append((x,y)) 
        else:
            known.append((x,y))        

major=[]
for z in range(181425): 
        q=RDS[unknown[z][0],:].tolist()+RCS[unknown[z][1],:].tolist() 
        major.append(q)

kmeans=KMeans(n_clusters=23, random_state=0).fit(major) 
center=kmeans.cluster_centers_  
center_x=[]
center_y=[]
for j in range(len(center)):
    center_x.append(center[j][0])
    center_y.append(center[j][1])

labels=kmeans.labels_ 
type1_x=[]
type1_y=[]
type2_x=[]
type2_y=[]
type3_x=[]
type3_y=[]
type4_x=[]
type4_y=[]
type5_x=[]
type5_y=[]
type6_x=[]
type6_y=[]
type7_x=[]
type7_y=[]
type8_x=[]
type8_y=[]
type9_x=[]
type9_y=[]
type10_x=[]
type10_y=[]
type11_x=[]
type11_y=[]
type12_x=[]
type12_y=[]
type13_x=[]
type13_y=[]
type14_x=[]
type14_y=[]
type15_x=[]
type15_y=[]
type16_x=[]
type16_y=[]
type17_x=[]
type17_y=[]
type18_x=[]
type18_y=[]
type19_x=[]
type19_y=[]
type20_x=[]
type20_y=[]
type21_x=[]
type21_y=[]
type22_x=[]
type22_y=[]
type23_x=[]
type23_y=[]
for i in range(len(labels)):
    if labels[i]==0:
        type1_x.append(unknown[i][0])
        type1_y.append(unknown[i][1])
    if labels[i]==1:
        type2_x.append(unknown[i][0])
        type2_y.append(unknown[i][1])
    if labels[i]==2:
        type3_x.append(unknown[i][0])
        type3_y.append(unknown[i][1])
    if labels[i]==3:
        type4_x.append(unknown[i][0])
        type4_y.append(unknown[i][1])
    if labels[i]==4:
        type5_x.append(unknown[i][0])
        type5_y.append(unknown[i][1])
    if labels[i]==5:
        type6_x.append(unknown[i][0])
        type6_y.append(unknown[i][1])
    if labels[i]==6:
        type7_x.append(unknown[i][0])
        type7_y.append(unknown[i][1])
    if labels[i]==7:
        type8_x.append(unknown[i][0])
        type8_y.append(unknown[i][1])
    if labels[i]==8:
        type9_x.append(unknown[i][0])
        type9_y.append(unknown[i][1])
    if labels[i]==9:
        type10_x.append(unknown[i][0])
        type10_y.append(unknown[i][1])
    if labels[i]==10:
        type11_x.append(unknown[i][0])
        type11_y.append(unknown[i][1])
    if labels[i]==11:
        type12_x.append(unknown[i][0])
        type12_y.append(unknown[i][1])
    if labels[i]==12:
        type13_x.append(unknown[i][0])
        type13_y.append(unknown[i][1])
    if labels[i]==13:
        type14_x.append(unknown[i][0])
        type14_y.append(unknown[i][1])
    if labels[i]==14:
        type15_x.append(unknown[i][0])
        type15_y.append(unknown[i][1])
    if labels[i]==15:
        type16_x.append(unknown[i][0])
        type16_y.append(unknown[i][1])
    if labels[i]==16:
        type17_x.append(unknown[i][0])
        type17_y.append(unknown[i][1])
    if labels[i]==17:
        type18_x.append(unknown[i][0])
        type18_y.append(unknown[i][1])
    if labels[i]==18:
        type19_x.append(unknown[i][0])
        type19_y.append(unknown[i][1])
    if labels[i]==19:
        type20_x.append(unknown[i][0])
        type20_y.append(unknown[i][1])
    if labels[i]==20:
        type21_x.append(unknown[i][0])
        type21_y.append(unknown[i][1])
    if labels[i]==21:
        type22_x.append(unknown[i][0])
        type22_y.append(unknown[i][1])
    if labels[i]==22:
        type23_x.append(unknown[i][0])
        type23_y.append(unknown[i][1])

type=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]                            
mtype=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]                                     
dataSet=[]                                
for k1 in range(len(type1_x)):
    type[0].append((type1_x[k1],type1_y[k1]))      
for k2 in range(len(type2_x)):
    type[1].append((type2_x[k2],type2_y[k2]))
for k3 in range(len(type3_x)):
    type[2].append((type3_x[k3],type3_y[k3]))
for k4 in range(len(type4_x)):
    type[3].append((type4_x[k4],type4_y[k4]))
for k5 in range(len(type5_x)):
    type[4].append((type5_x[k5],type5_y[k5]))
for k6 in range(len(type6_x)):
    type[5].append((type6_x[k6],type6_y[k6]))
for k7 in range(len(type7_x)):
    type[6].append((type7_x[k7],type7_y[k7]))
for k8 in range(len(type8_x)):
    type[7].append((type8_x[k8],type8_y[k8]))     
for k9 in range(len(type9_x)):
    type[8].append((type9_x[k9],type9_y[k9]))
for k10 in range(len(type10_x)):
    type[9].append((type10_x[k10],type10_y[k10]))
for k11 in range(len(type11_x)):
    type[10].append((type11_x[k11],type11_y[k11]))
for k12 in range(len(type12_x)):
    type[11].append((type12_x[k12],type12_y[k12]))
for k13 in range(len(type13_x)):
    type[12].append((type13_x[k13],type13_y[k13]))
for k14 in range(len(type14_x)):
    type[13].append((type14_x[k14],type14_y[k14]))
for k15 in range(len(type15_x)):
    type[14].append((type15_x[k15],type15_y[k15]))     
for k16 in range(len(type16_x)):
    type[15].append((type16_x[k16],type16_y[k16]))
for k17 in range(len(type17_x)):
    type[16].append((type17_x[k17],type17_y[k17]))
for k18 in range(len(type18_x)):
    type[17].append((type18_x[k18],type18_y[k18]))
for k19 in range(len(type19_x)):
    type[18].append((type19_x[k19],type19_y[k19]))
for k20 in range(len(type20_x)):
    type[19].append((type20_x[k20],type20_y[k20]))
for k21 in range(len(type21_x)):
    type[20].append((type21_x[k21],type21_y[k21]))
for k22 in range(len(type22_x)):
    type[21].append((type22_x[k22],type22_y[k22]))
for k23 in range(len(type23_x)):
    type[22].append((type23_x[k23],type23_y[k23]))          
for k in range(23):
    mtype[k]=random.sample(type[k],70)
for m2 in range(144):
    for n2 in range(1271):
        for z2 in range(23):
            if (m2,n2) in mtype[z2]: 
                dataSet.append((m2,n2))        
for m3 in range(144):
    for n3 in range(1271):
        if A[m3,n3]==1:
            dataSet.append((m3,n3)) 

x=[]                                                  
x1=[]                                     
x2=[]
y=[]  
for xx in dataSet:
    q=RDS[xx[0],:].tolist()+RCS[xx[1],:].tolist()   
    x.append(q)                                                         
    if (xx[0],xx[1]) in known:
        y.append(1)
    else:
        y.append(0) 

train_x=np.array(x,dtype=np.float64)
train_y=np.array(y,dtype=np.float64)
np.savetxt('circR2D2_finalx_yes.csv',train_x,delimiter=',') 


