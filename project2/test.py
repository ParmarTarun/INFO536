from evaluation import *
from keras.optimizers import Adam
from keras.layers import Dense, Activation
from keras.models import Sequential
import keras as ke
import pickle
import numpy as np
import pandas as pd


model = ke.models.load_model('mymodel.keras')

test_data = pd.read_pickle('test.pkl')


for te in test_data:
    dt = process_data(te)
    re = run(dt, model)
    print("Plate predicted:", re)
