# Optimize Hyperparameters of NeuroNER model using Hyperopt

#### Requirements
- Python **3**.
- [hyperopt]() (`pip install hyperopt`)
- all requirements for [NeuroNER]()

#### Usage
1. Prepare training, test and dev data. *You must using all three partitions will training in order for this script to work.*
2. Setup a config file for NeuroNER (by default, the script will look for this in `path/to/NeuroNER/src/hyperopt_parameters.ini`).
3. Run `hyperparameter_optimization.py`. You must provide the path to the `src` directory of your NeuroNER install as a positional argument. *e.g*.

```
python hyperparameter_optimization.py path/to/NeuroNER/src
```

4. Optionally, you can provide path to the parameter and output folders, and set the maximum number of evaluations for `hyperopt` to run. See `python hyperparameter_optimization.py --help` for more info.

> Note that it can get a little messy if your provide a path to a config file not in `path/to/NeuroNER/src`. It is best to just save a config file called `hyperopt_parameters.ini` in `src` and use relative paths in the config.

#### How the script works?

The script works by coordinating output from `NeuroNER` runs with `hyperopt` optimization. Essentially, hyperparameters to train are given a 'space', then NeuroNER is ran using hyperparameters from this space. The F1 score on the test set (using the model checkpoint with best performance on the validation set) is the minimization objective. The optmization algorithm used is the **Tree-structured Parzen Estimator (TPE)** algorithm.
