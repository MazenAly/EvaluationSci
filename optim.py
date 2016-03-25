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


################Question A - nested resampling #########

#pointers for the outer loop of 3 CV
skf = StratifiedKFold(y_train ,shuffle=True , n_folds=3 , random_state=0)
best_score = 0
best_gamma = 0 
best_C = 0
for train_index, test_index in skf:
    XX_train, XX_outfold = X_train.iloc[train_index], X_train.iloc[test_index]
    yy_train, yy_outfold = y_train.iloc[train_index], y_train.iloc[test_index]
    #RandomizedSearch has the inner loop of 3 CV
    rsearch = RandomizedSearchCV(estimator=SVC(), param_distributions=param_grid, n_iter=10  , cv=3    )
    rsearch.fit(XX_train, yy_train)
    #if the new score is better than the best score so far
    if rsearch.best_score_ > best_score: 
        best_score = rsearch.best_score_
        best_C = rsearch.best_estimator_.C
        best_gamma = rsearch.best_estimator_.gamma

print "best score ", best_score
print "best C ", best_C
print "best gamma ", best_gamma

#Building SVM Classifier with tuned parameters of nested resampling 
clf = SVC( kernel='rbf' , C= best_C , gamma= best_gamma )
clf.fit(X_train,y_train  )
#getting the score of the test left out 20% test data
print "Getting the score of the test data"
print clf.score(X_test , y_test)


################Question B - without nested resampling - tune on all the training data #########

rsearch = RandomizedSearchCV(estimator=SVC(), param_distributions=param_grid, n_iter=10  , cv=3    )
rsearch.fit(X_train, y_train)
 
print(rsearch.best_score_)
print(rsearch.best_estimator_.gamma)
print(rsearch.best_estimator_.C)

best_gamma = rsearch.best_estimator_.gamma
best_C = rsearch.best_estimator_.C

clf = SVC( kernel='rbf' , C= best_C , gamma= best_gamma )
clf.fit(X_train,y_train  )
print clf.score(X_test , y_test)

 
################Question C - without nested resampling - tune on all the training dat #########


C_range = [0.01 , 0.1  ,1 ,5 , 10 , 100 ,500,1000]
gamma_range = [ 0.0001 , 0.001 , 0.01 , 0.1 , 1 , 10 , 100 ,1000]



skf = StratifiedKFold(y_train ,shuffle=True , n_folds=3 , random_state=0)
iterations_num = []
grid_best_scores = []
grid_best_score = 0
for i in range(2,9):
    print "grid i is: " , i 
    iterations_num.append(i*i)
    grid_C = C_range[:i]
    print "Grid C"
    print grid_C
    grid_gamma = gamma_range[:i]
    print "Grid gamma"
    print grid_gamma
    param_grid = dict(gamma=grid_gamma, C=grid_C)
    best_score=0
    for train_index, test_index in skf:
        XX_train, XX_outfold = X_train.iloc[train_index], X_train.iloc[test_index]
        yy_train, yy_outfold = y_train.iloc[train_index], y_train.iloc[test_index]
        #RandomizedSearch has the inner loop of 3 CV
        gsearch = GridSearchCV(estimator=SVC(), param_grid=param_grid )
        gsearch.fit(XX_train, yy_train)
        if gsearch.best_score_ > best_score:
            best_score = gsearch.best_score_
    if best_gamma > grid_best_score:
        grid_best_score = best_score    
    print "Grid best score:"
    print grid_best_score
    grid_best_scores.append(grid_best_score)
    
print "loops done"
print iterations_num
print grid_best_scores


print "####################Grid finished##################"

randomized_best_scores = []
randomized_best_score = 0
random_search_C_values = np.logspace(-2, 10, 20)
random_search_gamma_values = np.logspace(-9, 3, 20)

param_grid = dict(gamma=random_search_gamma_values, C=random_search_C_values)

for iter in iterations_num:
    best_score=0
    for train_index, test_index in skf:
        XX_train, XX_outfold = X_train.iloc[train_index], X_train.iloc[test_index]
        yy_train, yy_outfold = y_train.iloc[train_index], y_train.iloc[test_index]
        #RandomizedSearch has the inner loop of 3 CV
        rsearch = RandomizedSearchCV(estimator=SVC(), param_distributions=param_grid, n_iter=iter  , cv=3    )
        rsearch.fit(XX_train, yy_train)
        if rsearch.best_score_ > best_score:
            best_score = rsearch.best_score_
    if best_gamma > randomized_best_score:
        randomized_best_score = best_score    
    randomized_best_scores.append(randomized_best_score)
    
    
plt.plot(iterations_num ,  grid_best_scores  , label="Grid Search")
plt.plot(iterations_num , randomized_best_scores , label="Randmoized Search")
plt.ylim(0.7,1.01)
plt.yticks(np.linspace(0.7,1.01,20,endpoint=True))
# Place a legend above this legend, expanding itself to
# fully use the given bounding box.
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0 )
plt.show()