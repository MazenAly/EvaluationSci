from openml.apiconnector import APIConnector
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.cross_validation import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


apikey = 'b6da739f426042fa9785167b29887d1a'
connector = APIConnector(apikey=apikey)
dataset = connector.download_dataset(32)

columns_names = [  'input' + str(x) for x in range(0,16) ]
columns_names.append('target')
train = dataset.get_dataset()
train = pd.DataFrame(train , columns = columns_names ) 
y = train['target']
X = train.iloc[:,:-1]
scaler = MinMaxScaler()
X = DataFrame(scaler.fit_transform(X) , columns = columns_names[:-1] )

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

print X_train



C_range = [0.01 , 0.1  ,1 ,5 , 10 , 100 ,500,1000]
gamma_range = [ 0.0001 , 0.001 , 0.01 , 0.1 , 1 , 10 , 100 ,1000]

param_grid = dict(gamma=gamma_range, C=C_range)



 
################Question C - without nested resampling - tune on all the training dat #########


C_range = [0.01 , 0.1  ,1 ,5 , 10 , 100 ,500,1000]
gamma_range = [ 0.0001 , 0.001 , 0.01 , 0.1 , 1 , 10 , 100 ,1000]



skf = StratifiedKFold(y_train ,shuffle=True , n_folds=3 , random_state=0)
iterations_num = []
for i in range(2,9):
    print "grid i is: " , i 
    iterations_num.append(i*i)

    
print "loops done"
print iterations_num


print "####################Grid finished##################"

randomized_best_scores = []
randomized_best_score = 0
random_search_C_values = np.logspace(-2, 10, 20)
random_search_gamma_values = np.logspace(-9, 3, 20)


print random_search_C_values

print random_search_gamma_values
param_grid = dict(gamma=random_search_gamma_values, C=random_search_C_values)

for iter in iterations_num:
    print "randomized iter: " , iter
    best_score=0
    for train_index, test_index in skf:
        XX_train, XX_outfold = X_train.iloc[train_index], X_train.iloc[test_index]
        yy_train, yy_outfold = y_train.iloc[train_index], y_train.iloc[test_index]
        #RandomizedSearch has the inner loop of 3 CV
        rsearch = RandomizedSearchCV(estimator=SVC(), param_distributions=param_grid, n_iter=iter  , cv=3    )
        rsearch.fit(XX_train, yy_train)
        if rsearch.best_score_ > best_score:
            best_score = rsearch.best_score_
    if best_score > randomized_best_score:
        randomized_best_score = best_score  
    print   randomized_best_score
    randomized_best_scores.append(randomized_best_score)

print  randomized_best_scores
    
