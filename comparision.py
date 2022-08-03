# -*- coding: utf-8 -*-
"""
comparision
"""

## Input  module
import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import sklearn
# import os
import keras
import sklearn
import scipy
from tqdm import tqdm 

import psutil

# memory total
zj = float(mem.total) 

# mem.used
ysy = float(mem.used)

# mem.free
kx = float(mem.free)


# os.chdir('D:/正在工作/论文2/project')
random.seed(10)

'''
###############################################################################
machine learning
###############################################################################
'''


'''data'''
raw=pd.read_csv('./data/raw01.csv').drop(['time'],axis=1)
data=raw.copy()

'''label'''

mid=124886
frep=0.4

##sw
sw_size=60
sw_step=60

raw=np.array(raw)
data_sw=[raw[j:j+sw_size].reshape(1,-1) for j in range(0,raw.shape[0]-sw_size,sw_step)]

for i in range(len(data_sw)):
    if i==0:
        temp=pd.DataFrame(data_sw[i])
    else:
        temp=temp.append(pd.DataFrame(data_sw[i]))

data_sw=temp.reset_index(drop=True)

mid_sw=int(mid/sw_step)
data_y=list(np.ones(mid_sw))+list(np.zeros(len(data_sw)-mid_sw))

p_index=list(range(mid_sw))
n_index=list(range(mid_sw,len(data_sw)))

##teat set
test_p_index=random.sample(p_index,int(frep*len(p_index)))
test_n_index=random.sample(n_index,int(frep*len(n_index)))

##train
train_p_index=list(set(p_index)^set(test_p_index))
train_n_index=list(set(n_index)^set(test_n_index))

train_data=data_sw.copy()
test_data=data_sw.copy()

train_y=data_y.copy()
test_y=data_y.copy()

if len(train_p_index)/len(train_n_index)>1:    
    a=train_p_index+train_n_index*int(len(train_p_index)/len(train_n_index))
else:
    a=train_p_index*int(len(train_n_index)/len(train_p_index))+train_n_index

##shuffle
random.shuffle(a)
##train_data train_y
train_y = type(train_y)(map(lambda i:train_y[i], a))
train_data=type(train_data)(map(lambda i:train_data.iloc[i,:], a))

a=test_p_index+test_n_index

##test_data test_y
test_y = type(test_y)(map(lambda i:test_y[i], a))
test_data=type(test_data)(map(lambda i:test_data.iloc[i,:], a))

# train_y_onehot = keras.utils.to_categorical(train_y)
# test_y_onehot = keras.utils.to_categorical(test_y)

sc = sklearn.preprocessing.StandardScaler()
sc.fit(train_data)

train_data_std = sc.transform(train_data)
test_data_std = sc.transform(test_data)
data_sw_std = sc.transform(data_sw)

# train_data_std = train_data
# test_data_std = test_data
# data_sw_std = data_sw

'''rf'''
from sklearn.ensemble import RandomForestClassifier

# mem = psutil.virtual_memory()
# rf_model_mem1=float(mem.used)

rf_model = RandomForestClassifier(n_estimators=30,n_jobs=-1)
rf_model.fit(train_data,train_y)

# mem = psutil.virtual_memory()
# rf_model_mem2=float(mem.used)
# print('rf',rf_model_mem2-rf_model_mem1)

# rf_model_time1=time.time()
# pre=rf_model.predict(test_data)
# rf_model_time2=time.time()
# print('rf',rf_model_time2-rf_model_time1)

pre=rf_model.predict(data_sw)
data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])
print(cm,f1)
    
'''xgb'''    

from xgboost.sklearn import XGBClassifier   
 
# mem = psutil.virtual_memory()  
# xgb_model_mem1=float(mem.used)
    
xgb_model = XGBClassifier(n_estimators=30,
# 	learning_rate=0.01,
# 	max_depth=8,
# 	min_child_weight=5,
# 	gamma=0.1,
# 	subsample=0.8,
# 	colsample_bytree=0.8,
# 	nthread=4,
 	scale_pos_weight=((len(data)-mid)/mid), 
    eval_metric=["logloss",'auc'],
	seed=1)
xgb_model.fit(train_data,train_y)

# mem = psutil.virtual_memory()
# xgb_model_mem2=float(mem.used)
# print('xgb',xgb_model_mem2-xgb_model_mem1)

# xgb_model_time1=time.time()
# pre=xgb_model.predict(test_data)
# xgb_model_time2=time.time()
# print('xgb',xgb_model_time2-xgb_model_time1)


pre=xgb_model.predict(data_sw)
data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])
print(cm,f1)

'''lgbm'''

from lightgbm.sklearn import LGBMClassifier

# mem = psutil.virtual_memory()
# lgb_model_mem1=float(mem.used)

lgb_model = LGBMClassifier(learning_rate=0.05,
                     n_estimators=30,
                     subsample=0.8,
                     subsample_freq=1,
                     colsample_bytree=0.8,
                      scale_pos_weight=((len(data)-mid)/mid),
                     )
lgb_model.fit(train_data,train_y)

# mem = psutil.virtual_memory()
# lgb_model_mem2=float(mem.used)
# print('lgb',lgb_model_mem2-lgb_model_mem1)

# lgb_model_time1=time.time()
# lgb=lgb_model.predict(test_data)
# lgb_model_time2=time.time()
# print('lgb',lgb_model_time2-lgb_model_time1)

lgb=lgb_model.predict(data_sw)
data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])
print(cm,f1)

'''knn'''

# mem = psutil.virtual_memory()
# knn_model_mem1=float(mem.used)

knn_model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=30)
knn_model.fit(train_data_std,train_y)

# mem = psutil.virtual_memory()
# knn_model_mem2=float(mem.used)
# print('knn',knn_model_mem2-knn_model_mem1)


# knn_model_time1=time.time()
# pre=knn_model.predict(test_data_std)
# knn_model_time2=time.time()
# print('knn',knn_model_time2-knn_model_time1)

pre=knn_model.predict(data_sw_std)
data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])
print(cm,f1)

'''svm''' 

# mem = psutil.virtual_memory()
# svm_model_mem1=float(mem.used)

svm_model = sklearn.svm.SVC()
svm_model.fit(train_data_std,train_y)

# mem = psutil.virtual_memory()
# svm_model_mem2=float(mem.used)
# print('svm',svm_model_mem2-svm_model_mem1)

# svm_model_time1=time.time()
# pre=svm_model.predict(test_data_std)
# svm_model_time2=time.time()
# print('svm',svm_model_time2-svm_model_time1)

pre=svm_model.predict(data_sw_std)
data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])
print(cm,f1)

'''GaussianNB'''

# mem = psutil.virtual_memory()
# gnb_model_mem1=float(mem.used)

gnb_model=sklearn.naive_bayes.GaussianNB()
gnb_model.fit(train_data_std,train_y)

# mem = psutil.virtual_memory()
# gnb_model_mem2=float(mem.used)
# print('gnb',gnb_model_mem2-gnb_model_mem1)

# gnb_model_time1=time.time()
# pre=gnb_model.predict(test_data_std)
# gnb_model_time2=time.time()
# print('gnb',gnb_model_time2-gnb_model_time1)

pre=gnb_model.predict(data_sw_std)
data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])
print(cm,f1)

'''logistic_model'''

# mem = psutil.virtual_memory()
# logistic_model_mem1=float(mem.used)

logistic_model = sklearn.linear_model.LogisticRegression(C=1)
logistic_model.fit(train_data_std,train_y)

# mem = psutil.virtual_memory()
# logistic_model_mem2=float(mem.used)
# print('logistic',logistic_model_mem2-logistic_model_mem1)

# logisti_model_time1=time.time()
# prepro = logistic_model.predict_proba(test_data_std)
# logisti_model_time2=time.time()
# print('logisti',logisti_model_time2-logisti_model_time1)

prepro = logistic_model.predict_proba(data_sw_std)
pre=1-(prepro[:,0]>prepro[:,1])
data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])
print(cm,f1)

'''
###############################################################################
deep learning
###############################################################################
'''

'''sw data'''

'''raw'''
raw=pd.read_csv('./data/raw01.csv').drop(['time'],axis=1)

data=raw.copy()

'''label'''
mid=124886
frep=0.4

##sw
sw_size=60
sw_step=5

data_sw=[raw.iloc[j:j+sw_size,:] for j in range(0,raw.shape[0]-sw_size,sw_step)]

# for i in range(len(data_sw)):
#     if i==0:
#         temp=data_sw[i]
#     else:
#         temp=temp.append(data_sw[i])

# data_sw=temp.reset_index(drop=True)

mid_sw=int(mid/sw_step)
data_y=list(np.ones(mid_sw))+list(np.zeros(len(data_sw)-mid_sw))

p_index=list(range(mid_sw))
n_index=list(range(mid_sw,len(data_sw)))

test_p_index=random.sample(p_index,int(frep*len(p_index)))
test_n_index=random.sample(n_index,int(frep*len(n_index)))

train_p_index=list(set(p_index)^set(test_p_index))
train_n_index=list(set(n_index)^set(test_n_index))

train_data=data_sw.copy()
test_data=data_sw.copy()

train_y=data_y.copy()
test_y=data_y.copy()

if len(train_p_index)/len(train_n_index)>1:    
    a=train_p_index+train_n_index*int(len(train_p_index)/len(train_n_index))
else:
    a=train_p_index*int(len(train_n_index)/len(train_p_index))+train_n_index


##shuffle
random.shuffle(a)

train_y = type(train_y)(map(lambda i:train_y[i], a))
train_data=type(train_data)(map(lambda i:train_data[i], a))

a=test_p_index+test_n_index
a=a[:100]*100

test_y = type(test_y)(map(lambda i:test_y[i], a))
test_data=type(test_data)(map(lambda i:test_data[i], a))

train_data=np.array(train_data)
test_data=np.array(test_data)
data_sw=np.array(data_sw)

train_y_onehot = keras.utils.to_categorical(train_y)
test_y_onehot = keras.utils.to_categorical(test_y)

'''mlp'''
# mem = psutil.virtual_memory()
# mlp_model_mem1=float(mem.used)

x_=keras.layers.Input(batch_shape=(None,sw_size,191))
x=keras.layers.BatchNormalization(axis=-1)(x_)
x=keras.layers.Dense(64, activation='relu')(x)
x=keras.layers.Dense(32, activation='relu')(x)
x=keras.layers.Dense(64, activation='relu')(x)
x=keras.layers.Dense(32, activation='relu')(x)
x=keras.layers.Dense(64, activation='relu')(x)
x=keras.layers.Dense(32, activation='relu')(x)
x=keras.layers.Dense(8, activation='relu')(x)
x=keras.layers.Dense(2, activation='softmax')(x)
mlp_model=keras.models.Model(inputs=x_, outputs=x)

optimizer = keras.optimizers.Adam(lr=0.001) 

mlp_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history=mlp_model.fit(train_data,train_y_onehot,
                      validation_data=(test_data,test_y_onehot),
                      batch_size = 32,epochs=30,verbose=1,
                      # class_weight={0:1,1:((len(data)-mid)/mid)},
                      )

# mem = psutil.virtual_memory()
# mlp_model_mem2=float(mem.used)
# print('mlp',mlp_model_mem2-mlp_model_mem1)

# mlp_model_time1=time.time()
# mlp_model.predict(test_data)
# mlp_model_time2=time.time()
# print('mlp',mlp_model_time2-mlp_model_time1)

prepro=np.round(mlp_model.predict(data_sw))
pre=1-(prepro[:,0]>prepro[:,1])
data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])

'''lstm'''

# mem = psutil.virtual_memory()
# lstm_model_mem1=float(mem.used)

x_=keras.layers.Input(batch_shape=(None,sw_size,191))
x=keras.layers.BatchNormalization(axis=-1)(x_)
x=keras.layers.LSTM(64, return_sequences=True)(x)
x=keras.layers.Dense(32, activation='relu')(x)
x=keras.layers.LSTM(64, return_sequences=True)(x)
x=keras.layers.Dense(32, activation='relu')(x)
x=keras.layers.LSTM(64, return_sequences=True)(x)
x=keras.layers.Dense(32, activation='relu')(x)
x=keras.layers.Dense(8, activation='relu')(x)
x=keras.layers.Dense(2, activation='softmax')(x)
lstm_model=keras.models.Model(inputs=x_, outputs=x)
  
lstm_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

history=lstm_model.fit(train_data,train_y_onehot,
                      validation_data=(test_data,test_y_onehot),
                      batch_size = 32,epochs=30,verbose=1,
                      # class_weight={0:1,1:((len(data)-mid)/mid)},
                      )

# mem = psutil.virtual_memory()
# lstm_model_mem2=float(mem.used)
# print('lstm',lstm_model_mem2-lstm_model_mem1)

# lstm_model_time1=time.time()
# lstm_model.predict(test_data)
# lstm_model_time2=time.time()
# print('lstm',lstm_model_time2-lstm_model_time1)

prepro=np.round(lstm_model.predict(data_sw))
pre=1-(prepro[:,0]>prepro[:,1])
data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])

'''1Dcnn'''

# mem = psutil.virtual_memory()
# cnn1d_model_mem1=float(mem.used)

x_=keras.layers.Input(batch_shape=(None,sw_size,191))
x=keras.layers.BatchNormalization(axis=-1)(x_)
x=keras.layers.Conv1D(64,8)(x)
x=keras.layers.Dense(32, activation='relu')(x)
x=keras.layers.Conv1D(64,8)(x)
x=keras.layers.Dense(32, activation='relu')(x)
x=keras.layers.Conv1D(64,8)(x)
x=keras.layers.Dense(32, activation='relu')(x)
x=keras.layers.Dense(8, activation='relu')(x)
x=keras.layers.Dense(2, activation='softmax')(x)
cnn1d_model=keras.models.Model(inputs=x_, outputs=x)
  
cnn1d_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history=cnn1d_model.fit(train_data,train_y_onehot,
                      validation_data=(test_data,test_y_onehot),
                      batch_size = 32,epochs=30,verbose=1,
                      # class_weight={0:1,1:((len(data)-mid)/mid)},
                      )

# mem = psutil.virtual_memory()
# cnn1d_model_mem2=float(mem.used)
# print('cnn1d',cnn1d_model_mem2-cnn1d_model_mem1)


# cnn1d_model_time1=time.time()
# cnn1d_model.predict(test_data)
# cnn1d_model_time2=time.time()
# print('cnn1d',cnn1d_model_time2-cnn1d_model_time1)

prepro=np.round(cnn1d_model.predict(data_sw))
pre=1-(prepro[:,0]>prepro[:,1])
data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]
f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])

'''
###############################################################################
tsne-iforest
###############################################################################
'''
# 56,164938;57,107187;58,5101;59,14145;
# 56-1,159672,不要encoding;57-2,83194;58-3,2959;59-4,91324;59-5,94562;56-6,124886

# raw=pd.read_csv('./data/raw56.csv').drop(['time','utc时间()'],axis=1)
raw=pd.read_csv('./data/raw01.csv').drop(['time'],axis=1)
data=raw.copy()

mid=94562
frep=0.4

##sw
sw_size=60
sw_step=5

raw=np.array(raw)

data_sw=[raw[j:j+sw_size].reshape(1,-1) for j in range(0,raw.shape[0]-sw_size,sw_step)]

for i in range(len(data_sw)):
    if i==0:
        temp=pd.DataFrame(data_sw[i])
    else:
        temp=temp.append(pd.DataFrame(data_sw[i]))

data_sw=temp.reset_index(drop=True)

mid_sw=int(mid/sw_step)
data_y=list(np.ones(mid_sw))+list(np.zeros(len(data_sw)-mid_sw))

p_index=list(range(mid_sw))
n_index=list(range(mid_sw,len(data_sw)))

##test set
test_p_index=random.sample(p_index,int(frep*len(p_index)))
test_n_index=random.sample(n_index,int(frep*len(n_index)))

##train
train_p_index=list(set(p_index)^set(test_p_index))
train_n_index=list(set(n_index)^set(test_n_index))

##tsne
tsne = sklearn.manifold.TSNE(n_components=3)
data_sw2=tsne.fit_transform(data_sw)

data_sw2=pd.DataFrame(data_sw2)

train_p_x=data_sw2.copy()
train_p_y=data_y.copy()

train_n_x=data_sw2.copy()
train_n_y=data_y.copy()

test_x=data_sw2.copy()
test_y=data_y.copy()

train_p_x=type(train_p_x)(map(lambda i:train_p_x.iloc[i,:], train_p_index))
train_p_y = type(train_p_y)(map(lambda i:train_p_y[i], train_p_index))

train_n_x=type(train_n_x)(map(lambda i:train_n_x.iloc[i,:], train_n_index))
train_n_y = type(train_n_y)(map(lambda i:train_n_y[i], train_n_index))

test_x=type(test_x)(map(lambda i:test_x.iloc[i,:], test_p_index+test_n_index))
test_y = type(test_y)(map(lambda i:test_y[i], test_p_index+test_n_index))

##iforest

# tsne-iforest
# clf = sklearn.ensemble.IsolationForest(random_state=0,n_estimators=100).fit(train_p_x)
# tsne-iforest+
clf = sklearn.ensemble.IsolationForest(random_state=0,n_estimators=100).fit(train_n_x)

train_p_score = clf.decision_function(train_p_x)
train_n_score = clf.decision_function(train_n_x)

##threshold
threshold=(train_p_score.mean()+train_n_score.mean())/2


# ##test time
# test_x=data_sw2.copy()
# a=test_p_index+test_n_index
# a=a[:1000]*100
# test_x=type(test_x)(map(lambda i:test_x.iloc[i,:], a))

# time1=time.time()
# test_score = clf.decision_function(test_x)
# pre=[int(i>threshold)+0 for i in test_score]
# time2=time.time()
# print('tsne-iforest',time2-time1)


test_score = clf.decision_function(data_sw2)
pre=[int(i<threshold)+0 for i in test_score]

data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]

f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])

print(cm,f1)

'''
###############################################################################
LDA-BNP-KL+
###############################################################################
'''
# 56,164938;57,107187;58,5101;59,14145;
# 56-1,159672,不要encoding;57-2,83194;58-3,2959;59-4,91324;59-5,94562;56-6,124886

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import MinMaxScaler


raw=pd.read_csv('./data/raw01.csv').drop(['time'],axis=1)
data=raw.copy()

scaler = MinMaxScaler()
scaler.fit(data)
data=np.round(scaler.transform(data)*100)

mid=124886
frep=0.1

##sw
sw_size=60
sw_step=60

data=np.array(data)

data_sw=[data[j:j+sw_size].reshape(1,-1) for j in range(0,data.shape[0]-sw_size,sw_step)]

for i in range(len(data_sw)):
    if i==0:
        temp=pd.DataFrame(data_sw[i])
    else:
        temp=temp.append(pd.DataFrame(data_sw[i]))

data_sw=temp.reset_index(drop=True)

mid_sw=int(mid/sw_step)
data_y=list(np.ones(mid_sw))+list(np.zeros(len(data_sw)-mid_sw))

p_index=list(range(mid_sw))
n_index=list(range(mid_sw,len(data_sw)))

test_p_index=random.sample(p_index,int(frep*len(p_index)))
test_n_index=random.sample(n_index,int(frep*len(n_index)))

train_p_index=list(set(p_index)^set(test_p_index))
train_n_index=list(set(n_index)^set(test_n_index))

train_p_x=data_sw.copy()
train_p_y=data_y.copy()

train_n_x=data_sw.copy()
train_n_y=data_y.copy()

test_x=data_sw.copy()
test_y=data_y.copy()

train_p_x=type(train_p_x)(map(lambda i:train_p_x.iloc[i,:], train_p_index))
train_p_y = type(train_p_y)(map(lambda i:train_p_y[i], train_p_index))

train_n_x=type(train_n_x)(map(lambda i:train_n_x.iloc[i,:], train_n_index))
train_n_y = type(train_n_y)(map(lambda i:train_n_y[i], train_n_index))

test_x=type(test_x)(map(lambda i:test_x.iloc[i,:], test_p_index+test_n_index))
test_y = type(test_y)(map(lambda i:test_y[i], test_p_index+test_n_index))

train_x=data_sw.copy()
train_y=data_y.copy()

train_x=type(train_x)(map(lambda i:train_x.iloc[i,:], train_p_index+train_n_index))
train_y = type(train_y)(map(lambda i:train_y[i], train_p_index+train_n_index))

#lda_p
lda_p=sklearn.decomposition.LatentDirichletAllocation(n_components=10,random_state=0)
lda_p.fit(train_p_x)
#Negative values in data passed to LatentDirichletAllocation.fit,

#lda-n
lda_n=sklearn.decomposition.LatentDirichletAllocation(n_components=10,random_state=0)
lda_n.fit(train_n_x)

#fit and test lda
lda_t=sklearn.decomposition.LatentDirichletAllocation(n_components=10,random_state=0)   
lda_t.fit(data_sw)

# from gensim import corpora, models
# lda = models.ldamodel.LdaModel(corpus=data_sw, id2word=id2word, num_topics=100)

# score_p=lda_p.score(np.array(data_sw.iloc[i,:]).reshape(1, -1))
# score_n=lda_n.score(np.array(data_sw.iloc[i,:]).reshape(1, -1))
# pre=(score_p>score_n)+0

# lda_n.get_params()

# pre=[]

# for i in tqdm(range(len(data_sw))):
#     perplexity_p=lda_p.perplexity(np.array(data_sw.iloc[i,:]).reshape(1, -1))##它有多不符合这个模型
#     perplexity_n=lda_n.perplexity(np.array(data_sw.iloc[i,:]).reshape(1, -1))
#     pre.append((perplexity_p<perplexity_n)+0)

topic_t=lda_t.transform(data_sw).T
topic_p=lda_p.transform(data_sw).T
topic_n=lda_n.transform(data_sw).T
KL_p = scipy.stats.entropy(topic_t, topic_p)
KL_n = scipy.stats.entropy(topic_t, topic_n) 
pre=(KL_p<KL_n)+0

data_y2=[1-i for i in data_y]
pre2=[1-i for i in pre]

f1=sklearn.metrics.f1_score(data_y2, pre2)
cm=sklearn.metrics.confusion_matrix(data_y2, pre2)
fdr=cm[1,1]/(cm[1,1]+cm[1,0])
far=cm[0,1]/(cm[0,1]+cm[0,0])

print(cm,f1)

# test_x=data_sw.copy()
# a=test_p_index+test_n_index
# a=a[:100]*10
# test_x=type(test_x)(map(lambda i:test_x.iloc[i,:], a))

# time1=time.time()
# topic_t=lda_t.transform(test_x).T
# topic_p=lda_p.transform(test_x).T
# topic_n=lda_n.transform(test_x).T
# KL_p = scipy.stats.entropy(topic_t, topic_p)
# KL_n = scipy.stats.entropy(topic_t, topic_n) 
# pre=(KL_p>KL_n)+0
# time2=time.time()
# print('LDA-KL',time2-time1)