import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence

max_features = 30000
maxlength_sentences = 400
embed_vec_size = 300
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

def load_csv(in_dir):
        global train = pd.read_csv(in_dir+'train.csv')
        global test = pd.read_csv(in_dir+'test.csv')
        print("DataSet Loaded from %" %in_dir)



def make_tokenizer(x_train,x_test):
    global tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(x_train)+list(x_test))
    with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def preprocessing():
    train['none'] = 1-train[list_classes].max(axis=1)
    list_classes.append('none')
    x_train = train["comment_text"].fillna("fillna").values
    y_train = train[list_classes].values
    x_test = test["comment_text"].fillna("fillna").values
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = sequence.pad_sequences(x_train,maxlen=maxlength_sentences)
    x_test = sequence.pad_sequences(x_test,maxlen=maxlength_sentences)


def load_embedding_index():
    global embeddings_index = dict()
    f = open(emb_dir+'glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(value[1:],dtype='float32')




# loading
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
