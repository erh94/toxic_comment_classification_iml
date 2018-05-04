import numpy as np
import pandas as pd
import pickle
from keras.models import model_from_json
from keras.models import Model
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
import warnings
warnings.filterwarnings("ignore")

mod_dir='./models/'
model_filename = 'GRU_feat_none'
tokenizer_filename = 'tokenizer.pkl'
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "none"]


def loadmodel():
	json_file = open(mod_dir+model_filename+'.json', 'r')
	loaded_model_json = json_file.read()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights(mod_dir+model_filename+".h5")
	print("Loaded model from disk")
	json_file.close()
	return loaded_model

def preprocess(comment):
	#load tokenizer
	with open(mod_dir+tokenizer_filename,'rb') as handle:
		tokenizer =  pickle.load(handle)

	# with h5py.File(mod_dir+'metadata.h5', 'r') as hf:
    # 	data = hf['name-of-dataset'][:]

	with open(mod_dir+'metadata.pkl','rb') as handle:
		metadata = pickle.load(handle)
	maxlen=metadata['max_len']
	max_features = metadata['max_features']
	embed_size = metadata['embed_size']


	p_comment = tokenizer.texts_to_sequences(comment)
	p_comment = sequence.pad_sequences(p_comment,maxlen=maxlen)

	return p_comment


def main():

	model = loadmodel()

	comment = open('data/input.txt', 'r').read()
	print(comment)
	comment= [comment]
	# print(comment)

	p_comment = preprocess(comment)

	print(p_comment)

	output = pd.DataFrame(index= ['Probabilities','Labels'],columns=list_classes)

	pred_comment_class = model.predict(p_comment)

	output.loc['Probabilities']=pred_comment_class[0]

	output.loc['Labels'] = [1 if output.loc['Probabilities'][i]>0.7 else 0 for i in range(len(list_classes))]

	output.drop(['none'],axis=1,inplace=True)

	print(output.head())

	pickle.dump(output, open('output/output.pkl', 'wb'))


if __name__ == '__main__':
    main()
