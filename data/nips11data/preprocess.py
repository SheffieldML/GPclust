# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import string
import os
import re
import numpy as np

"""
The NIPS 2011 papers were fetched from 

http://books.nips.cc/nips24/nips24.zip

unzipped, and run through pdf2text, putting the resulting files in ./txt
before running this file.

Apologies for the dumb processing. I'm not really an NLP expert :)
"""


f = file('../english.stop')
stopwords = f.read().split() + ['-']
f.close()
fnames = os.listdir('txt')

docs = []
for fn in fnames:
    f = file('txt/'+fn)
    s = f.read()
    f.close()
    s = string.lower(s)
    s = s.replace('\xef\xac\x81','fi') # damned ligatures
    s = re.sub("[^a-z\ \n-]"," ",s) # remove unwanted chars
    s = s.split()
    s = [w for w in s if (not w in stopwords) and len(w)>2]
    docs.append(np.array(sorted(s),dtype=np.str))

vocab = np.unique(np.hstack(docs))
wordcount,doccount = np.zeros(vocab.size,dtype=np.int), np.zeros(vocab.size,dtype=np.int)

#count how many times words appear
for vi,v in enumerate(vocab):
    for d in docs:
        m = d==v
        wordcount[vi] += m.sum()
        if np.any(m): doccount[vi] += 1


#remove words which only occur in a single doc.
i = doccount > 2
vocab, wordcount, doccount = vocab[i], wordcount[i], doccount[i]

#remove words which occur less than 10 times
i = wordcount > 10
vocab, wordcount, doccount = vocab[i], wordcount[i], doccount[i]

#re-represent documents as index vectors
doc_indexes = []
for doc in docs:
    doc = [w for w in doc if w in vocab]
    idoc = [str(np.where(vocab==w)[0][0]) for w in doc]
    doc_indexes.append(idoc)

#write to file
f = file('nips11_corpus','w')
f.write(' '.join(vocab)+'\n')
for fn,i in zip(fnames,doc_indexes):
    f.write(fn+' ')
    f.write(' '.join(i)+'\n')
f.close()

