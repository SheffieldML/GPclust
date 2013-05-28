import numpy as np
import colvb
V = 5000
D = 10
K = 2
N = 4000

topics = np.random.dirichlet(np.ones(V),K)
docprops = np.random.dirichlet(np.ones(K),D)
betatheta = np.dot(docprops,topics)
docs = [np.random.multinomial(N,doc) for doc in betatheta]
docs = np.asarray(docs)

m = colvb.LDA2(docs,K)

