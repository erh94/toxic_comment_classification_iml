import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import json
import h5py

max_features = 30000
maxlength_sentences = 400
embed_vec_size = 300
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
in_dir = './input/'
emb_dir ='./embeddings/'

def load_csv():
        global train, test
        train = pd.read_csv(in_dir+'train.csv')
        test = pd.read_csv(in_dir+'test.csv')
        print("DataSet Loaded from %s" %in_dir)



def make_tokenizer():
    global tokenizer 
    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(x_train)+list(x_test))
    #saving the tokenizer for preprocessing in predict
    with open('./models/tokenizer.pkl', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.DEFAULT_PROTOCOL)

def preprocessing():
    
    global x_train, y_train, embedding_matrix ,x_test

    train['none'] = 1-train[list_classes].max(axis=1)
    list_classes.append('none')
    x_train = train["comment_text"].fillna("fillna").values
    y_train = train[list_classes].values
    x_test = test["comment_text"].fillna("fillna").values

    make_tokenizer()

    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = sequence.pad_sequences(x_train,maxlen=maxlength_sentences)
    x_test = sequence.pad_sequences(x_test,maxlen=maxlength_sentences)
    
    # load_embedding_index()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_vec_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
    
    # return embedding_matrix , x_test , x_train

def load_embedding_index():
    global embeddings_index 
    embeddings_index = dict()
    f = open(emb_dir+'glove.6B.300d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:],dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print("Loaded %s word vectors." %len(embeddings_index))


def save_preprocess_data():
    # with open('./models/x_train.pkl', 'wb') as handle:
    #     pickle.dump(x_train, handle, protocol=pickle.DEFAULT_PROTOCOL)
    
    with h5py.File('./models/x_train.h5','w') as h5f_handle:
        h5f_handle.create_dataset('x_train',data=x_train, compression="gzip",compression_opts=9)

    with h5py.File('./models/x_test.h5','w') as h5f_handle:
        h5f_handle.create_dataset('x_test',data=x_test,compression="gzip",compression_opts=9)

    with h5py.File('./models/y_train.h5','w') as h5f_handle:
        h5f_handle.create_dataset('y_train',data=y_train,compression="gzip",compression_opts=9)

    # with open('./models/x_test.pkl', 'wb') as handle:
    #     pickle.dump(x_test, handle, protocol=pickle.DEFAULT_PROTOCOL)

    # with open('./models/y_train.pkl', 'wb') as handle:
    #     pickle.dump(y_train, handle, protocol=pickle.DEFAULT_PROTOCOL)
    with h5py.File('./models/embedding_matrix.h5','w') as h5f_handle:
        h5f_handle.create_dataset('embedding_matrix',data=embedding_matrix,compression="gzip",compression_opts=9)

    # with open('./models/embedding_matrix.pkl', 'wb') as handle:
    #     pickle.dump(embedding_matrix, handle,protocol=pickle.DEFAULT_PROTOCOL)

    print(type(x_train))
    print(x_train.shape)

    print(type(embedding_matrix))
    print(embedding_matrix.shape)

    metadata = {'max_features':max_features,'max_len':maxlength_sentences,'embed_size':embed_vec_size}
    
    with open('./models/metadata.pkl', 'wb') as handle:
        pickle.dump(metadata, handle,protocol=pickle.DEFAULT_PROTOCOL)




def main():
    load_csv()

    load_embedding_index()
    

    preprocessing()



    save_preprocess_data()




if __name__ == '__main__':
    main()

