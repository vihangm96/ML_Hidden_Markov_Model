import numpy as np

from util import accuracy
from hmm import HMM

# TODO:
def model_training(train_data, tags):
	"""
	Train HMM based on training data

	Inputs:
	- train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- tags: (1*num_tags) a list of POS tags

	Returns:
	- model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
	"""
	model = None
	###################################################
	# Edit here

	# states = tags
	# obs = sent/real words

	state_dict = dict()
	obs_dict = dict()
	counter = 0
	num_state = len(tags)
	distinct_words=[]
	#distinct_words = np.array([])

	for i in range(num_state):
		state_dict[tags[i]] = i


	first_counter = np.zeros([num_state],dtype='float')
	transition_counter = np.zeros([num_state,num_state],dtype='float')

	for sample in train_data:

		first_tag = sample.tags[0]
		first_counter[state_dict[first_tag]]+=1

		i=0
		if(sample.words[i] not in obs_dict.keys()):
			obs_dict[sample.words[i]] = counter
			counter+=1
			distinct_words.append(np.zeros([num_state]))
		distinct_words[int(obs_dict[sample.words[i]])][state_dict[sample.tags[i]]] += 1

		for i in range(1,len(sample.words)):

			transition_counter[state_dict[sample.tags[i-1]], state_dict[sample.tags[i]]] += 1

			if(sample.words[i] not in obs_dict.keys()):
				obs_dict[sample.words[i]] = counter
				counter+=1
				distinct_words.append(np.zeros([num_state]))

			distinct_words[int(obs_dict[sample.words[i]])][ state_dict[sample.tags[i]]] += 1


	num_obs = len(obs_dict)

	for distinct_word in distinct_words:
		distinct_word /= sum(distinct_word)

	B= np.array(distinct_words).T

	model = HMM(first_counter, transition_counter, B, obs_dict, state_dict)

	###################################################
	return model

# TODO:
def sentence_tagging(test_data, model, tags):
	"""
	Inputs:
	- test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
	- model: an object of HMM class

	Returns:
	- tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
	"""
	tagging = []
	###################################################
	# Edit here
	num_state,num_obs = model.B.shape

	for sentence in test_data:
		for word in sentence.words:
			if word not in model.obs_dict:
				model.obs_dict[word] = max(model.obs_dict.values()) + 1
				model.B = np.insert(model.B, num_obs, values=0.000001, axis=1)
		tagging.append(model.viterbi(sentence.words))

	###################################################
	return tagging
