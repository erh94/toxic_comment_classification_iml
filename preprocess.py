import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import json

max_features = 30000
maxlength_sentences = 400
embed_vec_size = 300
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
in_dir = './input/'
emb_dir ='./embeddings/'

def load_csv(in_dir):
        global train = pd.read_csv(in_dir+'train.csv')
        global test = pd.read_csv(in_dir+'test.csv')
        print("DataSet Loaded from %" %in_dir)



def make_tokenizer(x_train,x_test):
    global tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(x_train)+list(x_test))
    #saving the tokenizer for preprocessing in predict
    with open('./models/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def preprocessing():
    train['none'] = 1-train[list_classes].max(axis=1)
    list_classes.append('none')
    global x_train = train["comment_text"].fillna("fillna").values
    global y_train = train[list_classes].values
    x_test = test["comment_text"].fillna("fillna").values
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = sequence.pad_sequences(x_train,maxlen=maxlength_sentences)
    x_test = sequence.pad_sequences(x_test,maxlen=maxlength_sentences)
    
    # load_embedding_index()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    global embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    # return embedding_matrix , x_test , x_train

def load_embedding_index():
    global embeddings_index = dict()
    f = open(emb_dir+'glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = asarray(value[1:],dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print("Loaded %s word vectors." %len(embeddings_index))


def save_preprocess_data():
    with open('./models/x_train.pkl', 'wb') as handle:
        pickle.dump(x_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('./models/x_test.pkl', 'wb') as handle:
        pickle.dump(x_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('./models/embedding_matrix.pkl', 'wb') as handle:
        pickle.dump(embedding_matrix, handle,protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {'max_features':max_features,'max_len':maxlength_sentences,'embed_size':embed_vec_size}
    
    with open('./models/metadata.pkl', 'wb') as handle:
        pickle.dump(metadata, handle,protocol=pickle.HIGHEST_PROTOCOL)




def main():
    load_csv()
    load_embedding_index()
    make_tokenizer()

    preprocessing()

    save_preprocess_data()




if __name__ == '__main__':
    main()

