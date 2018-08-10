#coding=utf-8
import numpy as np
import nltk
from nltk.corpus import stopwords

#adding cross-validationn
#nonsense

def naive_bayes(train_pos,train_neg,test_pos,test_neg,stopword_toggle=1):
    stop=set([word.encode('ascii') for word in stopwords.words('english')])
    words=set()
    for line in train_pos+train_neg:
        words=words.union(line.split())

    if stopword_toggle==1:
        words=words-stop

    c_pos,c_neg,p_pos,p_neg={},{},{},{}
    for word in words:
        c_pos[word]=0
        c_neg[word]=0

    for line in train_pos:
        for word in line.split():
            if word in words:
                c_pos[word]+=1
    for line in train_neg:
        for word in line.split():
            if word in words:
                c_neg[word]+=1


    #MLE
    for word in words:
        p_pos[word]=float(c_pos[word]+1)/(c_pos[word]+c_neg[word]+2)
        p_neg[word]=float(c_neg[word]+1)/(c_pos[word]+c_neg[word]+2)

    

    r,w,b=0,0,0
    for line in test_pos:
        neg_prob,pos_prob=1,1
        for word in line.split():
            try:
                neg_prob*=p_neg[word]
                pos_prob*=p_pos[word]
            except:
                neg_prob*=0.5
                pos_prob*=0.5
        if pos_prob>=neg_prob:
            r+=1
        else:
            w+=1

    for line in test_neg:
        neg_prob,pos_prob=1,1
        for word in line.split():
            try:
                neg_prob*=p_neg[word]
                pos_prob*=p_pos[word]
            except:
                neg_prob*=0.5
                pos_prob*=0.5
        if pos_prob>neg_prob:
            w+=1
        else:
            r+=1

    return float(r)/(r+w)


if __name__=='__main__':
    train_pos,train_neg,test_pos,test_neg=[],[],[],[]
    pos = [line.lower() for line in open('rt-polarity.pos','r').readlines()]
    neg = [line.lower() for line in open('rt-polarity.neg','r').readlines()]



    size = len(pos)
    indices = np.random.permutation(size)
    print 'Without stopwords:'
    sets = (indices[:int(size*0.2)], indices[int(size*0.2):int(size*0.4)], indices[int(size*0.4):int(size*0.6)], indices[int(size*0.6):int(size*0.8)], indices[int(size*0.8):])
    for ind,test_inds in enumerate(sets):
        train_pos,train_neg,test_pos,test_neg=[],[],[],[]
        for i in range(size):
            if i in test_inds:
                test_pos.append(pos[i])
                test_neg.append(neg[i])
            else:
                train_pos.append(pos[i])
                train_neg.append(neg[i])
        print 'set',ind+1,'acc:',naive_bayes(train_pos,train_neg,test_pos,test_neg)    

    print 'With stopwords:'
    for ind,test_inds in enumerate(sets):
        train_pos,train_neg,test_pos,test_neg=[],[],[],[]
        for i in range(size):
            if i in test_inds:
                test_pos.append(pos[i])
                test_neg.append(neg[i])
            else:
                train_pos.append(pos[i])
                train_neg.append(neg[i])
        print 'set',ind+1,'acc:',naive_bayes(train_pos,train_neg,test_pos,test_neg,stopword_toggle=0) 
    

    





