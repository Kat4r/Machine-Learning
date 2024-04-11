import pandas as pd
import numpy as np
import tflearn
from tflearn.data_utils import load_csv
from tflearn.datasets import titanic
from sklearn.model_selection import train_test_split
titanic.download_dataset('titanic_dataser.csv')
