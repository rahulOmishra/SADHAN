# SADHAN
Rahul Mishra, Vinay Setty ["SADHAN: Hierarchical Attention Networks to Learn Latent Aspect Embeddings for Fake News Detection"](https://dl.acm.org/citation.cfm?doid=3341981.33442296). ICTIR 2019.  

## Dataset
Datasets of Politifact, Snopes and Fever can be downloaded from these links:

1. Politifact and Snopes: https://www.mpi-inf.mpg.de/dl-cred-analysis/
```
We use only top 30 results retrived from the web. We extract relevant snippets of text from the web documents using cosine similarity(>0.3) to include only highly relevant parts of the web documents.

```

2. Fever: http://fever.ai/data.html
```
We use random sampling method expalained in the original paper(https://www.aclweb.org/anthology/N18-1074). We use LDA(https://radimrehurek.com/gensim/models/ldamodel.html) to generate subjects/topics for the claims.

```

## Parameters/hyper-parameters and other settings

#Embeddings:
1. Word embeddings: GloVe word embeddings of 100 dimensions trained on 6 billion words (https://nlp.stanford.edu/projects/glove/)
  
2. Aspect embeddings: Each aspect embedding is of 100 dimension and initialized with uniformly random
weights.

#Loss function: Softmax cross entropy with logits

#learning rate: 0.001

#Number of hidden units in LSTM: 200

#Max sentence length: 100

#Max number of sentences:50

#keep-prob for dropout: 0.3



    
