#coding=utf-8
import pandas
import numpy as np
import gensim
import csv
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.core import Activation
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed,Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Dropout



def one_hot(labels,label='O'):
    vec=[0]*len(labels)
    vec[list(labels).index(label)]=1
    return np.array(vec)
lmodel=gensim.models.Word2Vec(train_sents,size=100)


dataset=[]
sent=[]
ne_tags=set()
with open('ner_dataset.csv',newline='\n') as csvfile:
    lines = csv.reader(csvfile,delimiter=',',quotechar='"')
    for item in lines:
        ne_tags.add(item[-1].split('-')[-1])
        if item[0]:
            if len(sent):
                dataset.append(sent)
            sent=[item[1:]]
        else:
            sent.append(item[1:])
print(len(dataset))
print(ne_tags)

max_len=max([len(sent) for sent in dataset])
dataset_size = len(dataset)
indices = np.random.permutation(dataset_size)
train_ind, test_ind = indices[:int(dataset_size*0.8)], indices[int(dataset_size*0.8):]
train_set,test_set=[],[]
for ind in train_ind:
    train_set.append(dataset[ind])
for ind in test_ind:
    test_set.append(dataset[ind])
train_sents=[[t[0] for t in sent] for sent in train_set]
train_pos=[[t[1] for t in sent] for sent in train_set]
train_ne=[[t[2].split('-')[-1] for t in sent] for sent in train_set]
test_sents=[[t[0] for t in sent] for sent in test_set]
test_pos=[[t[1] for t in sent] for sent in test_set]
test_ne=[[t[2].split('-')[-1] for t in sent] for sent in test_set]
print(len(test_set),len(train_set))


train_x,train_y=[],[]
for i,sent in enumerate(train_sents):
    sent_len=len(sent)
    ele_vec=[np.zeros(100)]*(max_len-sent_len)
#     print(np.array(ele_vec).shape)
    ele_ne=[one_hot(ne_tags)]*(max_len-sent_len)
    for j,word in enumerate(sent):
        try:
            ele_vec.append(lmodel.wv[word])
            ele_ne.append(one_hot(ne_tags,train_ne[i][j]))
        except:
            new_wv = 2*np.random.randn(100)-1 # sample from normal distn
            norm_const = np.linalg.norm(new_wv)
            new_wv /= norm_const
            ele_vec.append(new_wv)
            ele_ne.append(one_hot(ne_tags))
    train_x.append(ele_vec)
    train_y.append(ele_ne)
train_x=np.array(train_x)
train_y=np.array(train_y)
print(train_x.shape,train_y.shape)

test_x,test_y=[],[]
for i,sent in enumerate(test_sents):
    sent_len=len(sent)
    ele_vec=[np.zeros(100)]*(max_len-sent_len)
#     print(np.array(ele_vec).shape)
    ele_ne=[one_hot(ne_tags)]*(max_len-sent_len)
    for j,word in enumerate(sent):
        try:
            ele_vec.append(lmodel.wv[word])
            ele_ne.append(one_hot(ne_tags,test_ne[i][j]))
        except:
            new_wv = 2*np.random.randn(100)-1 # sample from normal distn
            norm_const = np.linalg.norm(new_wv)
            new_wv /= norm_const
            ele_vec.append(new_wv)
            ele_ne.append(one_hot(ne_tags))
    test_x.append(ele_vec)
    test_y.append(ele_ne)
test_x=np.array(test_x)
test_y=np.array(test_y)
print(test_x.shape,test_y.shape)

ner_model=Sequential()
ner_model.add(Bidirectional(LSTM(150,return_sequences=True,
                                 bias_regularizer=l2(0),
                                 activity_regularizer=l2(0.),
                                 kernel_regularizer=l2(0.))
                            ,
                            input_shape=(104,100)
                           ))
ner_model.add(Dropout(0.5))
ner_model.add(TimeDistributed(Dense(9,activation='softmax',
                                    bias_regularizer=l2(0.),
                                   activity_regularizer=l2(0.),
                                   kernel_regularizer=l2(0.))))
ner_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

ner_model.fit(train_x,train_y,epochs=5,batch_size=200)

scores = ner_model.evaluate(test_x, test_y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))