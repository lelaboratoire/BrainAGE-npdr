from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tpot.config import regressor_config_dict
from sklearn.model_selection import TimeSeriesSplit

brainage_train_data = pd.read_csv('/home/ttle/tpotBrainAGE/BrainAGE_train.csv')
brainage_test_data = pd.read_csv('/home/ttle/tpotBrainAGE/BrainAGE_test.csv')
random_state = 1618
label = 'age'
n_gen = 100
n_pop = 100

Xdatatrain = brainage_train_data.drop(label, axis=1)
Ydatatrain = brainage_train_data[label]
Xdatatest = brainage_test_data.drop(label, axis=1)
Ydatatest = brainage_test_data[label]

tpot = TPOTRegressor(generations = n_gen,
                 population_size = n_pop,
                 verbosity = 2,
                 config_dict = regressor_config_dict,
                 scoring = 'r2',
                 random_state = random_state,
                 cv = TimeSeriesSplit(n_splits=5),
                 template = 'Selector-Transformer-Regressor',
                 n_jobs = 8)
tpot.fit(Xdatatrain.values, Ydatatrain.values)
print(tpot.score(Xdatatest.values, Ydatatest.values))
tpot.export('tpot_brainAGE_pipeline.py')
