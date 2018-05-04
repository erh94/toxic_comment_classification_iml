from django.shortcuts import render
from .forms import ModelInputForm
# from seq2seq.preprocess import preprocess
from subprocess import call

import random
import pickle
import pandas as pd

# random.seed()

test_data = pd.read_csv('data/test.csv')
test_data = list(test_data['comment_text'])

# sentenceUnlMap = pickle.load(open('data/unl-eng-sentence-unl-map.pickle', 'rb'))
# sentenceUnlMapKeys = list(sentenceUnlMap.keys())

def get_output(request, comment):
	with open('data/input.txt', 'w') as f:
		f.write(comment)

	call(['python','predict.py'])

	output = pickle.load(open('output/output.pkl', 'rb'))
	output = output.to_html()
	return render(request, 'prediction/prediction.html', {'comment': comment, 'output': output})


def home_view(request):
	form = ModelInputForm()
	if request.method == 'POST':
		if 'submit' in request.POST:
			form = ModelInputForm(request.POST)

			if form.is_valid():
				print('here')
				model_inputs = form.save(commit=False)
				comment = model_inputs.comment
				print(comment)

				return get_output(request, comment)

		elif 'random' in request.POST:
			comment = random.choice(test_data)
			return get_output(request, comment)

	return render(request, 'prediction/prediction.html', {'form': form})
