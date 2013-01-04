# Copyright (c) 2012 James Hensman
# Licensed under the GPL v3 (see LICENSE.txt)

import numpy as np
import pylab as pb
import sys
sys.path.append('..')
from colvb import LDA

#generate some documents
WORDSIZE=3 # words are square matrices with a single nonzero entry
N_DOCS = 600
DOCUMENT_LENGTHS = [np.random.randint(15,20) for i in range(N_DOCS)]
N_TOPICS = WORDSIZE*2 # topics are horizontal or vertical bars

#here's the vocabulary
V = WORDSIZE**2
if WORDSIZE==2:
    vocab = np.array([u'\u25F0',u'\u25F3',u'\u25F1',u'\u25F2'],dtype="<U2")
else:
    vocab_ = [np.zeros((WORDSIZE,WORDSIZE)) for v in range(V)]
    [np.put(v,i,1) for i,v in enumerate(vocab_)]
    vocab = np.empty(len(vocab_),dtype=np.object)
    for i,v in enumerate(vocab_):
        vocab[i] = v

#generate the topics
topics = [np.zeros((WORDSIZE,WORDSIZE)) for i in range(N_TOPICS)]
for i in range(WORDSIZE):
    topics[i][:,i] = 1
    topics[i+WORDSIZE][i,:] = 1
topics = map(np.ravel,topics)
topics = map(lambda x: x/x.sum(),topics)

#if the docs are 2x2 square, you'll have as many topics as vocab, which won't work:
if WORDSIZE==2:
    topics = topics[:2]
    N_TOPICS = 2

#generate the documents
docs = []
doc_latents = []
doc_topic_probs = []
for d in range(N_DOCS):
    topic_probs = np.random.dirichlet(np.ones(N_TOPICS)*0.8)
    latents = np.random.multinomial(1,topic_probs,DOCUMENT_LENGTHS[d]).argmax(1)
    doc_latents.append(latents)
    doc_topic_probs.append(topic_probs)
    docs.append(np.array([np.random.multinomial(1,topics[i]).argmax() for i in latents]))
docs_visual = [np.zeros((WORDSIZE,WORDSIZE)) for d in range(N_DOCS)]
for d,dv in zip(docs, docs_visual):
    for w in d:
        dv.ravel()[w] += 1

#display the documents
nrow=ncol= np.ceil(np.sqrt(N_DOCS))
vmin = np.min(map(np.min,docs_visual))
vmax = np.max(map(np.max,docs_visual))
for d,dv in enumerate(docs_visual):
    pb.subplot(nrow,ncol,d+1)
    pb.imshow(dv,vmin=vmin,vmax=vmax,cmap=pb.cm.gray)
    pb.xticks([])
    pb.yticks([])
pb.suptitle('the "documents"')

m = LDA(docs,vocab,N_TOPICS)

x = m.get_vb_param().copy()
m.optimize(method='steepest',maxiter=1000)
m.set_vb_param(x)
m.optimize(method='FR',maxiter=1000)
pb.figure()
m.plot_tracks()

#display learned topics
def plot_inferred_topics():
    nrow=ncol= np.ceil(np.sqrt(N_TOPICS))
    pb.figure()
    for i,beta in enumerate(m.beta_p):
        pb.subplot(nrow,ncol,i+1)
        pb.imshow(beta.reshape(WORDSIZE,WORDSIZE),cmap=pb.cm.gray)
        pb.xticks([])
        pb.yticks([])
plot_inferred_topics()
pb.suptitle('inferred topics')
pb.show()

#plot true topics
nrow=ncol= np.ceil(np.sqrt(N_TOPICS))
pb.figure()
for i,topic in enumerate(topics):
    pb.subplot(nrow,ncol,i+1)
    pb.imshow(topic.reshape(WORDSIZE,WORDSIZE),cmap=pb.cm.gray)
    pb.xticks([])
    pb.yticks([])
pb.suptitle('true topics')







