#----------------------------------------------------
# This file is quoted from EPI-DLMH
#----------------------------------------------------
import itertools
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def sentence2word(str_set):
    word_seq=[]
    for sr in str_set:
        tmp=[]
        for i in range(len(sr)-5):
            if('N' in sr[i:i+6]):
                tmp.append('null')
            else:
                tmp.append(sr[i:i+6])
        word_seq.append(' '.join(tmp))
    return word_seq

def word2num(wordseq,tokenizer,MAX_LEN):
    sequences = tokenizer.texts_to_sequences(wordseq)
    numseq = pad_sequences(sequences, maxlen=MAX_LEN)
    return numseq

def sentence2num(str_set,tokenizer,MAX_LEN):
    wordseq=sentence2word(str_set)
    numseq=word2num(wordseq,tokenizer,MAX_LEN)
    return numseq

def get_tokenizer():
    f= ['a','c','g','t']
    c = itertools.product(f,f,f,f,f,f)
    res=[]
    for i in c:
        temp=i[0]+i[1]+i[2]+i[3]+i[4]+i[5]
        res.append(temp)
    res=np.array(res)
    NB_WORDS = 4097
    tokenizer = Tokenizer(num_words=NB_WORDS)
    tokenizer.fit_on_texts(res)
    acgt_index = tokenizer.word_index
    acgt_index['null'] = 0
    return tokenizer

def get_data(enhancers,promoters):
    tokenizer=get_tokenizer()
    MAX_LEN=3000
    X_en=sentence2num(enhancers,tokenizer,MAX_LEN)
    MAX_LEN=2000
    X_pr=sentence2num(promoters,tokenizer,MAX_LEN)

    return X_en,X_pr



names = ['GM12878', 'HUVEC', 'HeLa-S3', 'IMR90', 'K562', 'NHEK','all']
name = names[4]
train_dir='./data2/%s/train/'%name
imbltrain='./data2/%s/imbltrain/'%name
test_dir='./data2/%s/test/'%name
Data_dir='./data2/%s/'%name

import os
os.makedirs(train_dir, exist_ok=True)
os.makedirs(imbltrain, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(Data_dir, exist_ok=True)

print ('Experiment on %s dataset' % name)

print ('Loading seq data1...')
enhancers_tra=open(train_dir+'%s_enhancer.fasta'%name,'r').read().splitlines()[1::2]
promoters_tra=open(train_dir+'%s_promoter.fasta'%name,'r').read().splitlines()[1::2]
y_tra=np.loadtxt(train_dir+'%s_label.txt'%name)

im_enhancers_tra=open(imbltrain+'%s_enhancer.fasta'%name,'r').read().splitlines()[1::2]
im_promoters_tra=open(imbltrain+'%s_promoter.fasta'%name,'r').read().splitlines()[1::2]
y_imtra=np.loadtxt(imbltrain+'%s_label.txt'%name)

enhancers_tes=open(test_dir+'%s_enhancer_test.fasta'%name,'r').read().splitlines()[1::2]
promoters_tes=open(test_dir+'%s_promoter_test.fasta'%name,'r').read().splitlines()[1::2]
y_tes=np.loadtxt(test_dir+'%s_label_test.txt'%name)

print('balanced')
print('pos_samples:'+str(int(sum(y_tra))))
print('neg_samples:'+str(len(y_tra)-int(sum(y_tra))))
print('imblanced')
print('pos_samples:'+str(int(sum(y_imtra))))
print('neg_samples:'+str(len(y_imtra)-int(sum(y_imtra))))
print('test')
print('pos_samples:'+str(int(sum(y_tes))))
print('neg_samples:'+str(len(y_tes)-int(sum(y_tes))))



X_en_tra,X_pr_tra=get_data(enhancers_tra,promoters_tra)
X_en_imtra,X_pr_imtra=get_data(im_enhancers_tra,im_promoters_tra)
X_en_tes,X_pr_tes=get_data(enhancers_tes,promoters_tes)

np.savez(Data_dir+'%s_train.npz'%name,X_en_tra=X_en_tra,X_pr_tra=X_pr_tra,y_tra=y_tra)
np.savez(Data_dir+'im_%s_train.npz'%name,X_en_tra=X_en_imtra,X_pr_tra=X_pr_imtra,y_tra=y_imtra)
np.savez(Data_dir+'%s_test.npz'%name,X_en_tes=X_en_tes,X_pr_tes=X_pr_tes,y_tes=y_tes)

