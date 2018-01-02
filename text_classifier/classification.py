#coding=utf-8
import numpy as np
import sys,os
import glob
from math import log10
from nltk.corpus import wordnet as wn
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

stopwords=[line.strip() for line in open('stopwords','r').readlines()]

def labelmax(p):
    max_prob=0
    for label,prob in p.items():
        if max_prob<prob:
            prob=max_prob
            max_label=label
    return max_label

def clean(sent):
    delstr='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t'
    identify = string.maketrans(delstr,' '*len(delstr))
    return sent.translate(identify)
    # return sents

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

def stem(word):
    try:
        res=wn.morphy(word)
    except:
        res=''
    if res!=None and res not in stopwords and not is_number(res):
        return res.encode('ascii')
    else:
        return ''


def count(docs,word):
    total=0
    for doc in docs:
        total+=doc.count(word)
    return total

if __name__=='__main__':
    root='20_newsgroups'
    train_set={}
    test_set={}
    text_labels=[]
    text_count=[]
    newdir='all'
    for filedir in glob.glob(os.path.join(root, '*')):
        #text labels and sizes
        label = filedir.split('/')[-1]
        text_labels.append(label)
        files=glob.glob(os.path.join(filedir, '*'))
        size=len(files)
        text_count.append(size)

        #departure of train and test sets
        train_set[label]=[]
        test_set[label]=[]
        indices = np.random.permutation(size)
        for i in indices[:int(size*0.8)]:
            lines=open(files[i],'r').readlines()
            for i,line in enumerate(lines):
                if line.startswith('Lines'):
                    ind=i
                    break
            seq = [stem(word) for word in ' '.join([clean(line.strip().lower()) for line in lines[ind+1:]]).strip().split() if stem(word)!='']
            # print seq
            train_set[label].append(seq)
        for i in indices[int(size*0.8):]:
            lines=open(files[i],'r').readlines()
            for i,line in enumerate(lines):
                if line.startswith('Lines'):
                    ind=i
                    break
            seq = [stem(word) for word in ' '.join([clean(line.strip().lower()) for line in lines[ind+1:]]).strip().split() if stem(word)!='']
            # print seq
            test_set[label].append(seq)

    word_bank=set()
    for doc in train_set.values():
        for words in doc:
            word_bank|=set(words)
    print len(word_bank)

    ###Naive Bayes
    # text_labels=test_set.keys()
    # word_count={}
    # word_total={}
    # for word in word_bank:
    #     word_count[word]={}
    #     word_total[word]=0
    #     for label in test_set.keys():
    #         word_count[word][label]=count(train_set[label],word)
    #         word_total[word]+=word_count[word][label]

    # p_word={} 
    # for word in word_bank:
    #     p_word[word]={}
    #     for label in text_labels:
    #         p_word[word][label]=float(word_count[word][label]+0.1)/(word_total[word]+2)


    # r,w=0,0
    # for label,docs in test_set.items():
    #     for doc in docs:
    #         p={}
    #         for label2 in test_set.keys():
    #             p[label2]=0
    #         for word in doc:
    #             for label2 in test_set.keys():
    #                 try:
    #                     p[label2]+=log10(p_word[word][label2])
    #                 except:
    #                     p[label2]+=0
    # #         print p
    #         max_prob=-np.inf
    #         max_label=''
    #         for label2,prob in p.items():
    #             if max_prob<prob:
    #                 max_prob=prob
    #                 max_label=label2
    # #         print label,max_label
    #         if max_label==label:
    #             r+=1
    #         else:
    #             w+=1
    # print "Naive Bayes: ",float(r)/(r+w)

    ### Tf-Idf SVM
    labels=train_set.keys()
    # print labels
    corpus=[]
    label_inds=[]
    for label,docs in train_set.items():
        for doc in docs:
            label_inds.append(labels.index(label))
            corpus.append(' '.join(doc))
    for label,docs in test_set.items():
        for doc in docs:
            label_inds.append(label)
            corpus.append(' '.join(doc))
    # print len(corpus)

        label_inds=np.array(label_inds)
    vectorizer = CountVectorizer()
    tfidf = TfidfTransformer()
    X = vectorizer.fit_transform(corpus)
    # print len(vectorizer.get_feature_names())
    transformer =  TfidfTransformer()  
    tfidf = transformer.fit_transform(X)  
    # print tfidf.toarray().shape
    # print label_inds.shape

    test_split_mask = np.random.rand(len(corpus)) < 0.8
    train_X = tfidf.toarray()[test_split_mask]
    train_Y = label_inds[test_split_mask]
    test_X = tfidf.toarray()[~test_split_mask]
    test_Y = label_inds[~test_split_mask]
    # print train_X.shape, test_Y.shape

    #  KNN
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(train_X,train_Y)
    print clf.score(test_X,test_Y)


    #  SVM
    clf=SVC()
    clf.fit(train_X,train_Y)
    print clf.score(test_X,test_Y)



