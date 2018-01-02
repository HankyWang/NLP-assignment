import numpy as np

class Benchmark:
    __N, __cor, __err = 0, 0, 0

    def __init__(self,MODEL_NAME=None):
        self.__name__=MODEL_NAME

    def __del__(self):
        pass

    def add(self,my_ans,cor_ans):
        self.__N += len(cor_ans)
        i, ans = 1, set()
        for word in cor_ans:
            ans.add((i,i+len(word)))
            i+=len(word)
        i = 1
        for word in my_ans:
            if (i,i+len(word)) in ans:
                self.__cor+=1
            else:
                self.__err+=1
            i+=len(word)
        return

    def precision(self):
        print self.__name__,'Precision',float(self.__cor)/(self.__cor+self.__err)*100,'%'
        return

    def recall(self):
        print self.__name__,'Recall',float(self.__cor)/self.__N*100,'%'
        return

    def F_measure(self):
        print self.__name__,'F-measure',float(2*self.__cor*self.__cor)/(self.__cor*(self.__N+self.__cor+self.__err))*100,'%'

    def get_data(self,filename='dataset.txt'):
        word_bank = set()
        max_len=0
        sentences = []
        lines=[]

        for line in open(filename,'r').readlines():
            if line.strip():
                lines.append(line)

        size = len(lines)
        indices = np.random.permutation(size)
        train_ind, test_ind = indices[:int(size*0.8)], indices[int(size*0.8):]
        train_set,test_set=[],[]
        for ind in train_ind:
            train_set.append(lines[ind])
        for ind in test_ind:
            test_set.append(lines[ind])
        # train_set, test_set = lines[:size*0.8], lines[size*0.8:]s
        for line in train_set:
            items = line.strip().split('  ')
            words, attrs =[],[]
            # print items
            for item in items[1:]:
                words.append(item.split('/')[0])
                # attrs.append(item.split('/')[-1])
            # print words
            # print attrs
            for word in words[1:]:
                if word not in word_bank:
                    if len(word)>max_len:
                        max_len=len(word)
                    word_bank.add(word)

        for line in test_set:
            items = line.strip().split('  ')
            words=[]
            for item in items[1:]:
                words.append(item.split('/')[0])
            sentences.append((''.join(words),words))
        return sentences,word_bank,max_len

# def precision(my_ans,cor_ans):
#     i, ans = 1, set()
#     for word in my_ans:
#         ans.add((i,i+len(word)))
#         i+=len(word)
#     i, cor, err=1, 0, 0
#     for word in cor_ans:
#         if (i,i+len(word)) in ans:
#             cor+=1
#         else:
#             err+=1
#         i+=len(word)
#     return cor,err
