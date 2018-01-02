#   An instance of Bi-Direction Maximum Match algorithm
#   All right reversd.
#   Hank Wang @bupt
#
import random
from benchmark import Benchmark
from bmm import bmm_seg
from fmm import fmm_seg

def bdmm_seg(sentence,word_bank,max_len=100,unit=3):
    seg1 = fmm_seg(sentence,word_bank,max_len)
    seg2 = bmm_seg(sentence,word_bank,max_len)
    if len(seg1)==len(seg2):
        best_seg = seg1 if random.randint(0,1)==0 else seg2
    elif len([x for x in seg1 if len(x)==unit])<len([x for x in seg2 if len(x)==unit]):
        best_seg = seg1
    else:
        best_seg = seg2
    return best_seg

if __name__=='__main__':
    bdmm_model = Benchmark('bdmm')
    sentences, word_bank, max_len = bdmm_model.get_data()
    for sentence,ans in sentences:
        seg = bdmm_seg(sentence,word_bank,max_len)
        bdmm_model.add(seg,ans)

    bdmm_model.precision()
    bdmm_model.recall()
    bdmm_model.F_measure()
