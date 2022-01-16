import os
from glob import glob
import random


if __name__=="__main__":
    root = 'data/Fashion14k/imgs'
    clsname = os.listdir(root)
    cls2cid = {v:i for i, v in enumerate(clsname)}
    img2cls={k:[] for k in clsname}

    with open('classid.txt', 'w') as f:
        for k,v in cls2cid.items():
            f.write(f'{v}:{k}\n')

    imgpaths = glob(root+'/*/*.jpg')

    for fpath in imgpaths:
        key = os.path.dirname(fpath).split('/')[-1]
        img2cls[key].append(fpath)

    train =[]
    val =[]
    query =[]
    candidate=[]

    for k,v in img2cls.items():
        tl,i = len(v), int(len(v)*0.8)
        random.shuffle(v)
        [train.append((imgpath, cls2cid[k])) for imgpath in v[:i]]
        valid = v[i:]
        pivot=int(len(valid)*0.7)
        [candidate.append((imgpath, cls2cid[k])) for imgpath in valid[:pivot]]
        [query.append((imgpath, cls2cid[k])) for imgpath in valid[pivot:]]


    val = candidate+query

    cid_length = len(candidate)
    labels=[]

    with open('data/Fashion14k/filenames_train.txt', 'w') as f1, open('data/Fashion14k/label_train.txt', 'w') as f2:
        for i, ele in enumerate(train):
            path, label = ele[0], ele[1]
            f1.write(path+'\n')
            f2.write(f'{i} 0 {label}\n')

    with open('data/Fashion14k/filenames_valid.txt', 'w') as f1, open('data/Fashion14k/query_valid.txt', 'w') as fq, open('data/Fashion14k/candidate_valid.txt', 'w') as fc:
         for i, ele in enumerate(val):
             path, label  = ele[0], ele[1]
             f1.write(path+'\n')
             f = fc if i < cid_length else fq
             f.write(f'{i} 0 {label}\n')