#   An instance of Backward Maximum Match Algorithm
#   All right reversd.
#   Hank Wang @bupt
#

from benchmark import Benchmark

def fmm_seg(sentence,word_bank,max_len=100):
    seg = []
    ind=len(sentence) if len(sentence)<max_len else max_len
    while (len(sentence)):
        while (ind and sentence[:ind] not in word_bank):
            ind-=1
        if ind==0:
            ind=1
        seg.append(sentence[:ind])
        sentence = sentence[ind:]
        ind=len(sentence) if len(sentence)<max_len else max_len
    return seg

if __name__ == '__main__':
    fmm_model = Benchmark('fmm')
    sentences, word_bank, max_len = fmm_model.get_data()
    for sentence,ans in sentences:
        seg = fmm_seg(sentence,word_bank,max_len)
        fmm_model.add(seg,ans)

    fmm_model.precision()
    fmm_model.recall()
    fmm_model.F_measure()
