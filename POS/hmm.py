#coding=utf-8
import numpy as np
import argparse
import re

dying_rate=1e250

def is_m(s):
    pattern = re.compile(r'\d|１|２|３|４|９|７|５|６|０|８')
    match=pattern.match(s)
    return match

def is_t(s):
    pattern = re.compile(r'年|月|日')
    match=pattern.match(s)
    return match


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
 
def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}
 
    # Initialize base cases (t == 0)
    for y in states:
        # print y
        # print obs
        # print start_p[y]
        # print emit_p[y][obs[0]]
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
 
    # alternative Python 2.7+ initialization syntax
    # V = [{y:(start_p[y] * emit_p[y][obs[0]]) for y in states}]
    # path = {y:[y] for y in states}
 
    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
 
        for y in states:
            (prob, state) = max((V[t-1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states)
            V[t][y] = prob

            newpath[y] = path[state] + [y]
        if V[t][states[0]]<1e-250:
            for y in states:
                V[t][y]*=dying_rate
 
        # Don't need to remember the old paths
        path = newpath
 
    # print_dptable(V)
    (prob, state) = max((V[len(obs)-1][y], y) for y in states)
    return (prob, path[state])



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', dest='dataset',default='peking')
    args = parser.parse_args()
    if (args.dataset=='treebank'):
        tags=[]
        words=[]
        tagset=set(tags)
        wordset=set(words)
        train=open('treebank.5290.train').readlines()
        test=open('treebank.5290.test').readlines()

        for line in train:
            items = line.lower().strip().split(" ")
            for item in items:
                res=item.split('/')
                tags.append(res[-1])
                if is_number('/'.join(res[:-1])):
                    words.append('#')
                else:
                    words.append('/'.join(res[:-1]))
                tagset=tagset.union(res[-1].split('|'))
            tags.append('#end')
            words.append('#end')


        for line in train+test:
            items = line.lower().strip().split(' ')
            for item in items:
                if is_number('/'.join(res[:-1])):
                    words.add('#')
                else:
                    wordset.add('/'.join(item.split('/')[:-1]))
        wordset.add('#end')
        tagset.add('#end')
        V=len(wordset)

        
        #计算Start Prob
        st_count={}
        st_prob={}
        for tag in tagset:
            st_count[tag]=0
        for line in train:
            res=line.lower().strip().split(" ")[0].split('/')[-1]
            st_count[res]+=1
        for tag in tagset:
            st_prob[tag]=float(st_count[tag])/len(train)

        # 初始化Transmission
        trans_count={}
        tag_count={}
        trans_prob={}
        for tag in tagset:
            trans_count[tag]={}
            trans_prob[tag]={}
            tag_count[tag]=0
            for tag_r in tagset:
                trans_count[tag][tag_r]=0

        #计算Transmission Count
        for i,tag in enumerate(tags[1:]):
            # print tag
            # print tags[i]
            former=tags[i].split('|')
            now=tag.split('|')
            for tag_former in former:
                for tag_now in now:
                    # if tag_former=='wj' and stag_now=='end':
                        # print 1
                    trans_count[tag_former][tag_now]+=1
                    tag_count[tag_now]+=1

        #计算Tranmission Prob
        for tag in tagset:
            for tag_r in tagset:
                trans_prob[tag][tag_r]=float(trans_count[tag][tag_r])/tag_count[tag]

        #初始化Emission
        emit_count={}
        emit_prob={}
        for tag in tagset:
            emit_count[tag]={}
            emit_prob[tag]={}
            for word in wordset:
                emit_count[tag][word]=0

        #计算Emission Count
        for i,word in enumerate(words[1:]):
            res=tags[i+1].split('|')
            for tag in res:
                emit_count[tag][word]+=1

        #计算Emission Prob
        for word in wordset:
            for tag in tagset:
                emit_prob[tag][word]=float(emit_count[tag][word]+0.1)/(tag_count[tag]+V/10.0)

        r=0
        w=0
        for line in test:
            obs=[]
            right=[]
            items = line.lower().strip().split(" ")
            for item in items:
                res=item.split('/')
                if is_number('/'.join(res[:-1])):
                    obs.append('#')
                else:
                    obs.append('/'.join(res[:-1]))
                right.append(res[-1])
            obs.append('#end')
            prob, predict=viterbi(obs,tuple(tagset),st_prob,trans_prob,emit_prob)
            # print predict
            # print right
            for i in range(len(predict)-1):
                if predict[i] in set(right[i].split('|')):
                    r+=1
                else:
                    w+=1
        print 'Test:',float(r)/(r+w)

        r=0
        w=0
        for line in train:
            obs=[]
            right=[]
            items = line.lower().strip().split(" ")
            for item in items:
                res=item.split('/')
                if is_number('/'.join(res[:-1])):
                    obs.append('#')
                else:
                    obs.append('/'.join(res[:-1]))
                right.append(res[-1])
            obs.append('#end')
            prob, predict=viterbi(obs,tuple(tagset),st_prob,trans_prob,emit_prob)
            # print predict
            # print right
            for i in range(len(predict)-1):
                if predict[i] in set(right[i].split('|')):
                    r+=1
                else:
                    w+=1
        print 'Train:',float(r)/(r+w)
    else:
        tags=[]
        words=[]
        
        lines=[]
        tagset=set()
        wordset=set()
        for line in open('dataset.txt','r').readlines():
            if line.strip():
                lines.append(line)

        size = len(lines)
        indices = np.random.permutation(size)
        train_ind, test_ind = indices[:int(size*0.9)], indices[int(size*0.9):]
        train_set,test_set=[],[]
        for ind in train_ind:
            train_set.append(lines[ind])
        for ind in test_ind:
            test_set.append(lines[ind])

        for line in train_set:
            items = line.strip().split("  ")
            for item in items:
                res=item.split('/')
                tags.append(res[-1])
                if is_t(res[0]):
                    words.append('#t')
                elif is_m(res[0]):
                    words.append('#m')
                else:
                    words.append(res[0])
                tagset=tagset.union(res[-1].split(']'))
            words.append('end')
            tags.append('end')


        for line in test_set+train_set:
            items = line.strip().split('  ')
            for item in items:
                res=item.split('/')
                if is_t(res[0]):
                    wordset.add('#t')
                elif is_m(res[0]):
                    wordset.add('#m')
                else:
                    wordset.add(res[0])
        wordset.add('end')
        tagset.add('end')
        V=len(wordset)

        #计算Start Prob
        st_count={}
        st_prob={}
        for tag in tagset:
            st_count[tag]=0
        for line in train_set:
            res=line.strip().split("  ")[0].split('/')[-1]
            st_count[res]+=1
        for tag in tagset:
            st_prob[tag]=float(st_count[tag])/len(train_set)

        # 初始化Transmission
        trans_count={}
        tag_count={}
        trans_prob={}
        for tag in tagset:
            trans_count[tag]={}
            trans_prob[tag]={}
            tag_count[tag]=0
            for tag_r in tagset:
                trans_count[tag][tag_r]=0
        # print tags[:3]
        #计算Transmission Count
        for i,tag in enumerate(tags[1:]):
            # print tag
            # print tags[i]
            former=tags[i].split(']')
            now=tag.split(']')
            for tag_former in former:
                for tag_now in now:
                    # if tag_former=='wj' and stag_now=='end':
                        # print 1
                    trans_count[tag_former][tag_now]+=1
                    tag_count[tag_now]+=1

        #计算Tranmission Prob
        for tag in tagset:
            for tag_r in tagset:
                trans_prob[tag][tag_r]=float(trans_count[tag][tag_r])/tag_count[tag]

        #初始化Emission
        emit_count={}
        emit_prob={}
        for tag in tagset:
            emit_count[tag]={}
            emit_prob[tag]={}
            for word in wordset:
                emit_count[tag][word]=0

        #计算Emission Count
        for i,word in enumerate(words[1:]):
            res=tags[i+1].split(']')
            for tag in res:
                emit_count[tag][word]+=1

        #计算Emission Prob
        for word in wordset:
            for tag in tagset:
                emit_prob[tag][word]=float(emit_count[tag][word]+0.01)/(tag_count[tag]+V/100.0)

        r=0
        w=0
        for line in test_set:
            obs=[]
            right=[]
            items = line.strip().split("  ")
            for item in items:
                res=item.split('/')
                right.append(res[-1])
                if is_t(res[0]):
                    obs.append('#t')
                elif is_m(res[0]):
                    obs.append('#m')
                else:
                    obs.append(res[0])
            obs.append('end')
            # print obs
            prob, predict=viterbi(obs,tuple(tagset),st_prob,trans_prob,emit_prob)
            # print prob
            for i in range(len(predict)-1):
                if predict[i] in set(right[i].split(']')):
                    r+=1
                else:
                    w+=1
        print 'Test:',float(r)/(r+w)

        r=0
        w=0
        for line in train_set:
            obs=[]
            right=[]
            items = line.strip().split("  ")
            for item in items:
                res=item.split('/')
                right.append(res[-1])
                if is_t(res[0]):
                    obs.append('#t')
                elif is_m(res[0]):
                    obs.append('#m')
                else:
                    obs.append(res[0])
            obs.append('end')
            # print obs
            prob, predict=viterbi(obs,tuple(tagset),st_prob,trans_prob,emit_prob)
            # print prob
            for i in range(len(predict)-1):
                if predict[i] in set(right[i].split(']')):
                    r+=1
                else:
                    w+=1
        print 'Train:',float(r)/(r+w)










