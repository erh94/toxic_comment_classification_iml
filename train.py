import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
from keras.models import Model
from keras.layers import Input,Dense,Embedding,SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import h5py
from keras.callbacks import TensorBoard
from time import time 

global maxlen,max_features,embed_size,x_train_1,y_train,embedding_matrix

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_batch_end(self, epoch, logs={}):
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - till now: %d - score: %.6f \n" % (batch+1, score))

def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.1)(x)
    x = Bidirectional(GRU(100, return_sequences=True))(x)
    # x = SpatialDropout1D(0.1)(x)	
    # x= Dense(50,activation='tanh')(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(7, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model




if __name__ == "__main__":

	# Read inputs a) Max Feat b) Embed_size c) Embed_matrix d) maxlen 
	
	with open('./models/metadata.pkl','rb') as handle:
	 	metadata=pickle.load(handle)

	with h5py.File('./models/x_train.h5','r') as handle: 
	 	x_train=handle.get('x_train')
	 	x_train_1 = np.array(x_train)	

	with h5py.File('./models/y_train.h5','r') as handle: 
	 	y_train=handle.get('y_train')
	 	y_train = np.array(y_train)

	with h5py.File('./models/embedding_matrix.h5','r') as handle: 
	 	embedding_matrix = handle.get('embedding_matrix')
	 	embedding_matrix = np.array(embedding_matrix) 		 	

	maxlen = metadata['max_len']
	max_features = metadata['max_features']
	embed_size = metadata['embed_size']	

	model = get_model()
	print("Created Model \n")

	batch_size = 64 
	epochs = 2

	X_tra, X_val, y_tra, y_val = train_test_split(x_train_1, y_train, train_size=0.95, random_state=233)
	RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

	tensorboard = TensorBoard(log_dir="logs/{}".format(time()),write_graph=True,write_images=True)
	hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),callbacks=[tensorboard])

	# serialize model to JSON
	model_json = model.to_json()
	with open("./models/GRU_Best.json", "w") as json_file:
		json_file.write(model_json)

	# serialize weights to HDF5
	model.save_weights("./models/GRU_Best.h5")
	print("***************Saved model to disk named GRU_Dense***************")
