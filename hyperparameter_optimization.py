#!usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import shutil
import pprint
import logging
import argparse
import configparser

from math import ceil
from random import randint
from datetime import datetime
from hyperopt import pyll, fmin, tpe, hp
from sklearn.model_selection import train_test_split

# script developed with instructions from here: https://vooban.com/en/tips-articles-geek-stuff/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters/

# (TEMP FIX) this counter marks the current run. 1 = first run, used to avoid updating hyperparameters on our first run
CURRENT_RUN = 1
UPDATE_CONFIG = False

class cd:
	"""Context manager for changing the current working directory."""
	def __init__(self, newPath):
		self.newPath = os.path.expanduser(newPath)

	def __enter__(self):
		self.savedPath = os.getcwd()
		os.chdir(self.newPath)

	def __exit__(self, etype, value, traceback):
		os.chdir(self.savedPath)

def initialize_argparse():
	"""Initilize and return an argparse object."""
	parser = argparse.ArgumentParser(description='Tune the hyperparameters of a NeuroNER model using Hyperopt')
	# positional arguments
	parser.add_argument('working_directory', help="Path to the 'src' directory of your NeuroNER install")
	# optional arguments
	parser.add_argument('-p', '--parameter_filepath', help="Filepath to NeuroNER parameter file", default = 'hyperopt_parameters.ini')
	parser.add_argument('-m', '--max_evals', help="Maximum number of evaluations to perform during optmization", default = 100, type=int)
	parser.add_argument('-o', '--output_folder', help="Folder to save NeuroNER output from runs", default = '../output/hyperopt')
	parser.add_argument('-ss', '--stochastic_samples', help="Print a few (random) stochastic samples from the hyperparameter space", default = False, action='store_true')
	parser.add_argument('-s', '--shuffle', help="Shuffle & split dataset into new train/test/dev partitions on each run", default = False, action='store_true')
	return parser

def extract_f1_score(file_obj):
	"""Extracts the F1 Score from a given NeuroNER text evaluation file object."""
	# preprocess performance metrics for extraction of f1 score
	performance_metrics = file_obj.readlines()[1].strip()
	performance_metrics = [line for line in performance_metrics.split(' ') if line != '']
	f1_score = round(float(performance_metrics[7]), 2)

	return f1_score

def f1_best_epoch_on_test(hyperopt_run_path):
	"""Returns the F1 score on test set for best performing epoch as measured on the valid set."""

	# makes the assumption that NeuroNER only creates one directory at neuroner_run_path
	neuroner_run_path = os.path.join(hyperopt_run_path, os.listdir(hyperopt_run_path)[0])

	with cd(neuroner_run_path):
		# set defaults
		best_f1_on_valid = 0
		best_epoch_on_valid = '000'
		# loop over every evaluation file in the ouput directory
		for file in os.listdir():
			filename = os.fsdecode(file)
			# check its a text file and that its an evaluation on the valid set
			if filename.endswith(".txt") and 'valid.txt_conll_evaluation' in filename:
				with open(filename, 'r') as valid_eval:
					# get number of best performing epoch on valid
					current_f1_score_on_valid = extract_f1_score(valid_eval)
				if current_f1_score_on_valid > best_f1_on_valid:
					best_f1_on_valid = current_f1_score_on_valid
					best_epoch_on_valid = filename.split('_')[0]

	print("[INFO] Best epoch on valid ({}) had F1 score: {}".format(best_epoch_on_valid, best_f1_on_valid))
	return best_epoch_on_valid, best_f1_on_valid

def update_config(hyperparameters):
	"""Updates the NeuroNER config file with values in hyperparameters."""
	config = configparser.ConfigParser()
	config.read(args.parameter_filepath)

	# THIS IS ENTIRELY HARD-CODED, NEED BETTER SOLUTION
	parameter_file_headers = {
	'character_embedding_dimension': 'ann',
	'character_lstm_hidden_state_dimension': 'ann',
	'learning_rate': 'training',
	'gradient_clipping_value': 'training',
	'dropout_rate': 'training',
	'output_folder': 'dataset',
	'tagging_format': 'advanced'
	}

	for hp in hyperparameters:
		config[parameter_file_headers[hp]][hp] = str(hyperparameters[hp])

	with open(param_filepath, 'w') as configfile:
		config.write(configfile)

def run_model():
	'''Run NeuroNER using parameter file specied by parameter_filepath argument.'''
	param_filepath = args.parameter_filepath
	os.system('python3 main.py --parameters_filepath {}'.format(param_filepath))

def hyperopt_step(hyperparameters, shuffle=False):
	"""Coordinates one optimzation step: by updating hyperparameters from the space, saving them to the config,
	running NeuroNER with this config, and then returning the F1 score on the test set from the best performing model
	checkpoint (as measured on the validation set)."""

	# TEMPORARY FIX
	global CURRENT_RUN
	seed = 0

	# create new output dir for NeuroNER run, add dir path to hyperparam config
	hyperopt_run_path = os.path.join(output_folder, 'hyperopt_run_{}'.format(datetime.now().strftime("%y-%m-%d-%H-%M")))
	os.makedirs(hyperopt_run_path, exist_ok=True)
	hyperparameters['output_folder'] = hyperopt_run_path

	# if this is the first run, do not update hyperparameters
	if UPDATE_CONFIG:
		if CURRENT_RUN > 1:
			print("[INFO] Updating hyperparameters...")
			update_config(hyperparameters)
		else:
			print("[INFO] First run, will NOT update hyperparameters...")
	else:
		print('[INFO] Not updating hyperparameters...')
		update_config({'output_folder':hyperopt_run_path})

	if shuffle:
		config = configparser.ConfigParser()
		config.read(args.parameter_filepath)
		dataset_dir_path = config['dataset']['dataset_text_folder']

		print("[INFO] Unzipping data")
		with cd('../data'):
			os.system('rm -rf {}'.format(dataset_dir_path))
			os.system('tar -zxvf {}.tar.gz'.format(dataset_dir_path, dataset_dir_path))

		print("[INFO] Shuffling data")
		seed = randint(1, int(10e6))
		split_brat_standoff(dataset_dir_path, 0.6, 0.3, 0.1, random_seed=seed)

	print("[INFO] Running model...")
	run_model()

	# update run counter
	CURRENT_RUN += 1

	# maximize f1 by minimizing 100 - f1
	print("[INFO] Getting F1 score on valid from best performing epoch...")
	best_epoch_on_valid, best_f1_score = f1_best_epoch_on_test(hyperopt_run_path)

	if UPDATE_CONFIG:
		if shuffle:
			logging.info("Best performing epoch (%s): F1 score on valid: %s for hyperparameters: %s and random seed: %s" %
				(best_epoch_on_valid, best_f1_score, hyperparameters, seed))
		else:
			logging.info("Best performing epoch (%s): F1 score on valid: %s for hyperparameters: %s" % (best_epoch_on_valid, best_f1_score, hyperparameters))
	elif not UPDATE_CONFIG:
		if shuffle:
			logging.info("Best performing epoch (%s): F1 score on valid: %s for random seed: %s" %
				(best_epoch_on_valid, best_f1_score, seed))
		else:
			logging.info("Best performing epoch (%s): F1 score on valid: %s" % (best_epoch_on_valid, best_f1_score))


	return best_f1_score

def objective(space, shuffle=False):
	"""Returns the F1 score of the model for the current hyperparameter values.
	Serves as the objective function for hyperopt."""

	# hyperparameters
	character_embedding_dimension = space['character_embedding_dimension']
	learning_rate = space['learning_rate']
	gradient_clipping_value = space['gradient_clipping_value']
	dropout_rate = space['dropout_rate']
	tagging_format = space['tagging_format']

	# is there a better way to store above ^ values in a dictionary?
	current_hyperparams = {
	'character_embedding_dimension': int(character_embedding_dimension),
	'character_lstm_hidden_state_dimension': int(character_embedding_dimension),
	'learning_rate': learning_rate,
	'gradient_clipping_value': gradient_clipping_value,
	'dropout_rate': dropout_rate,
	'tagging_format': tagging_format
	}

	# update params, run model, and get F1 score
	f1_score_on_step = hyperopt_step(current_hyperparams, shuffle=shuffle)
	# maximize f1 by minimizing 100 - f1
	minimization_objective = 100 - f1_score_on_step

	return minimization_objective

def split_brat_standoff(corpra_dir, train_size, test_size, valid_size, random_seed=42):
	"""
	Randomly splits the corpus into train, test and validation sets.

	Args:
		corpus_dir: path to corpus
		train_size: float, train set parition size
		test_size: float, test set parition size
		valid_size: float, validation set parition size
		random_seed: optional, seed for random parition

	"""
	assert train_size < 1.0 and train_size > 0.0, "TRAIN_SIZE must be between 0.0 and 1.0"
	assert test_size < 1.0 and test_size > 0.0, "TEST_SIZE must be between 0.0 and 1.0"
	assert valid_size < 1.0 and valid_size > 0.0, "VALID_SIZE must be between 0.0 and 1.0"
	assert ceil(train_size + test_size + valid_size) == 1, "TRAIN_SIZE, TEST_SIZE, and VALID_SIZE must sum to 1"

	print('[INFO] Moving to directory: {}'.format(corpra_dir))
	with cd(corpra_dir):
		print('[INFO] Getting all filenames in dataset...', end = ' ')
		# accumulators
		text_filenames = []
		ann_filenames =[]

		# get filenames
		for file in os.listdir():
			filename = os.fsdecode(file)
			if filename.endswith(".txt") and not filename.startswith('.'):
				text_filenames.append(filename)
			elif filename.endswith('.ann') and not filename.startswith('.'):
				ann_filenames.append(filename)

		assert len(text_filenames) == len(ann_filenames), '''Must be equal
			number of .txt and .ann files in corpus_dir'''

		# hackish way of making sure .txt and .ann files line up across the two lists
		text_filenames.sort()
		ann_filenames.sort()

		print('DONE')
		print('[INFO] Splitting corpus into {}% train, {}% test, {}% valid...'.format(train_size*100, test_size*100, valid_size*100),
			end=' ')

		# split into train and all other, then split all other into test and valid
		X_train, X_test_and_valid = train_test_split(text_filenames, train_size=train_size, random_state = random_seed)
		y_train, y_test_and_valid = train_test_split(ann_filenames, train_size=train_size, random_state = random_seed)
		X_test, X_valid = train_test_split(X_test_and_valid, train_size=test_size/(1-train_size), random_state = random_seed)
		y_test, y_valid = train_test_split(y_test_and_valid, train_size=test_size/(1-train_size), random_state = random_seed)

		# leads to less for loops
		X_train.extend(y_train)
		X_test.extend(y_test)
		X_valid.extend(y_valid)

		print('Done.')
		print('[INFO] Creating train/test/valid directories at {} if they do not already exist...'.format(corpra_dir),
			end=' ')
		# if they do not already exist
		os.makedirs('train', exist_ok=True)
		os.makedirs('test', exist_ok=True)
		os.makedirs('valid', exist_ok=True)
		print('Done.')

		for x in X_train:
			shutil.move(x, 'train/' + x)
		for x in X_test:
			shutil.move(x, 'test/' + x)
		for x in X_valid:
			shutil.move(x, 'valid/' + x)

def stochastic_sample(space):
	'''Print a few random (stochastic) samples from the space.'''
	pp = pprint.PrettyPrinter(indent=4, width=100)
	for _ in range(10):
		pp.pprint(pyll.stochastic.sample(space))
	sys.exit()

if __name__ == '__main__':

	################################### change parameters here ###################################
	# hyperparameter space to optmize
	space = {

	# [ann]
	# use this to tune both the char embedding dimension and the char_lstm_hidden state dimension
	'character_embedding_dimension': hp.quniform('character_embedding_dimension', 10, 80, 1),

	# [training]
	'learning_rate': hp.uniform('learning_rate', 0.001, 0.01),
	'gradient_clipping_value': hp.quniform('gradient_clipping_value', 4, 6, 1),
	'dropout_rate': hp.uniform('dropout_rate', 0.3, 0.6),
	# [advanced]
	'tagging_format': hp.choice('tagging_format', ['bioes', 'bio'])
	}

	################################################################################################

	# parse CL arguments
	parser = initialize_argparse()
	args = parser.parse_args()

	if args.stochastic_samples: stochastic_sample(space) # print random samples from space

	param_filepath = str(args.parameter_filepath) # path to NeuroNER parameters
	output_folder = str(args.output_folder) # path to save NeuroNER runs
	working_directory = str(args.working_directory) # path to top level of NeuroNEr install
	max_evals = int(args.max_evals) # max number of evaluations to perform during optimization
	shuffle = bool(args.shuffle) # if True, randomly re-split all data into train/test/dev on each run

	# move into src directory of NeuroNER install
	with cd(working_directory):
		# create log file, write start time and date
		logging.basicConfig(filename='hyperopt.log', level=logging.INFO)
		logging.info('Starting script at: %s'%(str(datetime.now())))
		# create output for NeuroNER runs
		os.makedirs(output_folder, exist_ok=True)

		# use hyperopt to minimize the objective function by optmizing the hyperparameter space
		best = fmin(
		fn = objective(shuffle=shuffle), # the objective function to minimize
		space = space,
		algo = tpe.suggest,
		max_evals = max_evals
		)

	logging.info('FOUND MINIMUM AFTER %s TRIALS\n%s'%(max_evals, best))
	print("FOUND MINIMUM AFTER {} TRIALS:\n{}".format(args.max_evals, best))
