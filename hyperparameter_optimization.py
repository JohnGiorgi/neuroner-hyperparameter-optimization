#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import pprint
import logging
import argparse
import configparser

from datetime import datetime
from hyperopt import pyll, fmin, tpe, hp

# (TODO): Add logging - this would make it convinient to see the best output dir
# https://vooban.com/en/tips-articles-geek-stuff/hyperopt-tutorial-for-optimizing-neural-networks-hyperparameters/

# thats it... the last thing to figure out is how many cycles to run. depends on training time
# also, need to figure out the best way to define the parameter spaces...

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
	'''Initilize and return an argparse object.'''
	parser = argparse.ArgumentParser(description='Tune the hyperparameters of a NeuroNER model using Hyperopt')
	# positional arguments
	parser.add_argument('working_directory', help="Path to the 'src' directory of your NeuroNER install")
	# optional arguments
	parser.add_argument('-p', '--parameter_filepath', help="Filepath to NeuroNER parameter file", default = 'hyperopt_parameters.ini')
	parser.add_argument('-m', '--max_evals', help="Maximum number of evaluations to perform during optmization", default = 100, type=int)
	parser.add_argument('-o', '--output_folder', help="Folder to save NeuroNER output from runs", default = '../output/hyperopt')
	parser.add_argument('-ss', '--stochastic_samples', help="Print a few (random) stochastic samples from the hyperparameter space", default = False, action='store_true')
	return parser
def extract_f1_score(file_obj):
	'''Extracts the F1 Score from a given NeuroNER text evaluation file object.'''
	# preprocess performance metrics for extraction of f1 score
	performance_metrics = file_obj.readlines()[1].strip()
	performance_metrics = [line for line in performance_metrics.split(' ') if line != '']
	f1_score = round(float(performance_metrics[7]), 2)

	return f1_score
def f1_best_epoch_on_test(hyperopt_run_path):
	'''Returns the F1 score on test set for best performing epoch as measured on the valid set.'''
	# makes the assumption that NeuroNER only creates one directory at neuroner_run_path
	neuroner_run_path = os.path.join(hyperopt_run_path, os.listdir(hyperopt_run_path)[0])
	
	with cd(neuroner_run_path):
		# set defaults
		best_f1_on_valid = 0
		best_epoch_on_valid = '000'
		# loop over every evaluation file in the ouput directory
		for file in os.listdir():
			filename = os.fsdecode(file)
			# check serves to ensure its a text file and that its an evaluation on the valid set
			if filename.endswith(".txt") and 'valid.txt_conll_evaluation' in filename:
				with open(filename, 'r') as valid_eval:
					# get number of best performing epoch on valid
					current_f1_score_on_valid = extract_f1_score(valid_eval)
					if current_f1_score_on_valid > best_f1_on_valid:
						best_f1_on_valid = current_f1_score_on_valid
						best_epoch_on_valid = filename.split('_')[0]
	
	print("[INFO] Best epoch on valid ({}) had F1 score: {}".format(best_epoch_on_valid, best_f1_on_valid))
	return best_f1_on_valid
def update_config(hyperparameters):
	'''Updates the NeuroNER config file with values in hyperparameters.''' 
	config = configparser.ConfigParser()
	config.read(args.parameter_filepath)

	# THIS IS ENTIRELY HARD-CODED, NEED BETTER SOLUTION
	parameter_file_headers = {
	'character_embedding_dimension': 'ann',
	'character_lstm_hidden_state_dimension': 'ann',
	'learning_rate': 'training',
	'gradient_clipping_value': 'training',
	'dropout_rate': 'training',
	'output_folder': 'dataset'
	}

	for hp in hyperparameters:
		config[parameter_file_headers[hp]][hp] = str(hyperparameters[hp])
	
	with open(param_filepath, 'w') as configfile:
		config.write(configfile)
def run_model():
	'''Run NeuroNER using parameter file specied by parameter_filepath argument.'''
	param_filepath = args.parameter_filepath
	os.system('python3 main.py --parameters_filepath {}'.format(param_filepath))
def hyperopt_step(hyperparameters):
	'''Coordinates one optimzation step: by updating hyperparameters from the space, saving them to the config, 
	running NeuroNER with this config, and then returning the F1 score on the test set from the best performing model 
	checkpoint (as measured on the validation set).'''
	# create a new output directory for NeuroNER run
	hyperopt_run_path = os.path.join(output_folder, 'hyperopt_run_{}'.format(datetime.now().strftime("%y-%m-%d-%H-%M")))
	os.makedirs(hyperopt_run_path, exist_ok=True)
	hyperparameters['output_folder'] = hyperopt_run_path
	
	print("[INFO] Updating hyperparameters...")
	update_config(hyperparameters)
	
	print("[INFO] Running model...")
	run_model()

	print("[INFO] Getting F1 score on valid from best performing epoch...")
	best_f1_score = f1_best_epoch_on_test(hyperopt_run_path)
	logging.info("F1 score on valid: %s for hyperparameters: %s" % (best_f1_score, hyperparameters))

	return best_f1_score	
def objective(space):
	'''Returns the F1 score of the model for the current hyperparameter values.
	Serves as the objective function for hyperopt.'''

	# hyperparameters
	character_embedding_dimension = space['character_embedding_dimension']
	learning_rate = space['learning_rate']
	gradient_clipping_value = space['gradient_clipping_value']
	dropout_rate = space['dropout_rate']

	# is there a better way to store above ^ values in a dictionary?
	current_hyperparams = {
	'character_embedding_dimension': character_embedding_dimension,
	'character_lstm_hidden_state_dimension': character_embedding_dimension,
	'learning_rate': learning_rate,
	'gradient_clipping_value': gradient_clipping_value,
	'dropout_rate': dropout_rate
	}

	# update params, run model, and get F1 score
	current_best_f1_score = hyperopt_step(current_hyperparams)

	return current_best_f1_score
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
	'character_embedding_dimension': hp.randint('character_embedding_dimension', 80),
		
	# [training]
	'learning_rate': hp.uniform('learning_rate', 0.001, 0.01),
	'gradient_clipping_value': hp.randint('gradient_clipping_value', 6),
	'dropout_rate': hp.uniform('dropout_rate', 0.2, 0.8)
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
	
	# move into src directory of NeuroNER install
	with cd(working_directory):
		# create log file, write start time and date
		logging.basicConfig(filename='hyperopt.log', level=logging.INFO)
		logging.info('Starting script at: %s\n%s'%(str(datetime.now())))
		# create output for NeuroNER runs
		os.makedirs(output_folder, exist_ok=True)

		# use hyperopt to minimize the objective function by optmizing the hyperparameter space
		best = fmin(
		fn = objective, # the objective function to minimize
		space = space,
		algo = tpe.suggest,
		max_evals = max_evals	
		)

	logging.info('FOUND MINIMUM AFTER %s TRIALS\n%s'%(max_evals, best)
	print("FOUND MINIMUM AFTER {} TRIALS:\n{}".format(args.max_evals, best))

