#   An instance of Backward Maximum Match algorithm
#   All right reversd.
#   Hank Wang @bupt
#

from benchmark import Benchmark

def bmm_seg(sentence,word_bank,max_len=100):
    seg = []
    ind=0 if len(sentence)<max_len else len(sentence)-max_len
    while (len(sentence)):
        while (ind!=len(sentence) and sentence[ind:] not in word_bank):
            ind+=1
        if ind==len(sentence):
            ind=len(sentence)-1
        seg.append(sentence[ind:])
        sentence = sentence[:ind]
        ind=0 if len(sentence)<max_len else len(sentence)-max_len
    seg.reverse()
    return seg

if __name__ == '__main__':
    bmm_model = Benchmark('bmm')
    sentences, word_bank, max_len = bmm_model.get_data()
    for sentence,ans in sentences:
        seg = bmm_seg(sentence,word_bank,max_len)
        bmm_model.add(seg,ans)

    bmm_model.precision()
    bmm_model.recall()
    bmm_model.F_measure()
