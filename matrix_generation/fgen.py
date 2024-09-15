import numpy as np
from collections import defaultdict, deque
import itertools
import sympy as sp

def get_po(factors):
    if (len(factors) == 1):
        factors.append(0)
    print(factors)
    finlist = [[*[(i, i+j) for i in range(int(np.sum(factors[0:l-1])+1),int(np.sum(factors[0:l])-1)) for j in range(2,np.min([np.sum(factors[0:l])-i+1,4]))],*[(np.sum(factors[0:l]) -k, j) for k in range(0,np.min([factors[l-1],2])) for j in range(np.sum(factors[0:l])+1,np.sum(factors[0:l])+np.min([3,factors[l]+1]))]] for l in range(1,len(factors))]
    return [elem for s in [elem for s in finlist for elem in s] for elem in s]
    
def rebuildlists(k):
    lsts=[[1],[0]]
    for i in range(k-1):
        res=[]
        for j in lsts:
            if j[-1]==1:
                res.append(j+[0])
                res.append(j+[1])
            else:
                res.append(j+[1])
        lsts=res
    return res

def set_conv(lst):
    if (len(lst) == 0):
        return set()
    else:
        return set.union(*[set(i) for i in lst])

def list_set_mix(lst1, lst2):
    if (len(lst1) == 0 and len(lst2) == 0):
        return []
    elif (len(lst1) == 0):
        return [elem for s in [list(set(i)) for i in lst2] for elem in s]
    elif (len(lst2) == 0):
        return [elem for s in [list(set(i)) for i in lst1] for elem in s]
    else:
        return list(set.union(*[set(i) for i in lst2], *[set(i) for i in lst1]))
        
def lstlist2(lst):
    return [list([*i]) for i in lst]

def tupsub(tup):
    return np.abs(tup[0]-tup[1])

def s_alg(s1,s2, nodes):
    res = s1.copy()
    z1=[i for i in range(nodes-1) if s1[i]==0]
    z2=[i for i in range(nodes-1) if s2[i]==0]
    prodinds=lstlist2(list(itertools.product(*[z1,z2])))
    difs=[tupsub(i) for i in prodinds]
    zdif=[prodinds[i] for i in range(len(prodinds)) if difs[i]==0]
    odif=[prodinds[i] for i in range(len(prodinds)) if difs[i]==1]
    unqs = list_set_mix(odif, zdif)
    for i in unqs:
        res[i]=1 
    for i in list(set(z2) - set_conv(zdif)):
        res[i]=0
    return res

def genmat(sdx,n,nodes):
    A = [0]*(n*n)
    for i in range(n):
        for j in range(n):
            A[n*i + j] = tuple(s_alg(sdx[i], sdx[j], nodes))
    return A