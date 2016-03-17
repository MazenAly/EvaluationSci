from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs
from sklearn.utils import resample

import warnings

warnings.simplefilter("error")
#Generation of a 2-dimensional dataset with 1000 examples
X, y = make_blobs(n_samples=1000, centers=3, n_features=2,    random_state=0)

X = DataFrame(X, columns={'dim1' , 'dim2'})
y = DataFrame(y, columns={'target'})


k_values = [ 2 , 3,5 , 7 ,10 , 13 , 15 , 18 ,20 , 23, 25 , 28 ,30 , 35 ,40 ,43 , 45 ,50 ]
k_values = [ 5 ,15 , 23 ,30 ,40 ,50  ]

#===============================================================================
# 
# k_scores = []
# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k , weights='uniform'   )
#     scores = cross_val_score(knn, X, y, cv=10)
#     k_scores.append(scores.mean())
# print(k_scores)
# # plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis) or the metric
# plt.plot(k_values, k_scores)
# plt.xticks(k_values)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Accuracy')
# plt.show()
#===============================================================================

def loss_func(x):
    loss_counts = [ i for i in x if i != x['target'] and  i == i  ]
    predictions_no = len([ i for i in x if i == i  ]) - 1
    if predictions_no == 0:
        return -1
    return len(loss_counts) / float(predictions_no)

def variance_func(x):
    x = x.drop('bias')
    variantes = [ i for i in x if  i == i  ]
    predictions_no = len(variantes) - 1
    if predictions_no == 0:
        return -1
    return np.var(variantes[1:])

def diff(a ,b):
    b= set(b)
    return [ x for x in a if x not in b ]   


bias_values = []
variance_values = []
for k in k_values:
    var_bias_df = y 
    scores = []
    
    for i in range(100):
        training_samples,  training_labels = resample(X, y)
        test_index = diff(X.index , training_samples.index.unique() )
        test_samples = X.iloc[test_index,:]
        test_labels =  y.iloc[test_index,:] 
        knn = KNeighborsClassifier(n_neighbors=k , weights='uniform'   )
        knn.fit(training_samples, training_labels.target.values)
        preds = knn.predict(test_samples)
        booster_preds = DataFrame( preds , index=test_samples.index , columns = ['booster_' + str(i) ]) 
        var_bias_df = var_bias_df.join(booster_preds)
        #print var_bias_df.apply(lambda x:x.isnull().sum(), axis = 0)
        score = knn.score(test_samples, test_labels.target.values)
        scores.append(score)
        #print score
    print "=====Bootstrapping done======"  
    var_bias_df['bias'] = var_bias_df.apply(  loss_func ,  axis = 1)
    var_bias_df['var'] = var_bias_df.apply(  variance_func ,  axis = 1)
    
    
    bias_list = var_bias_df['bias'].values.tolist()
    var_list = var_bias_df['var'].values.tolist()
    print k
    print np.mean(list(filter(lambda x: x!= -1  , bias_list)))
    bias_values.append( np.mean(list(filter(lambda x: x!= -1  , bias_list))))
    variance_values.append( np.mean(list(filter(lambda x: x!= -1, var_list))))
    print "==========="



plt.figure(1)
plt.subplot(211)
plt.plot(k_values, bias_values)
plt.xticks(k_values)
plt.xlabel('Value of K for KNN')
plt.ylabel('bias')

plt.subplot(212)
plt.plot(k_values, variance_values)
plt.xticks(k_values)
plt.xlabel('Value of K for KNN')
plt.ylabel('Variance')
plt.show()

#===============================================================================
# f = plt.figure(1)
# plt.plot(k_values, bias_values)
# #plt.plot(k_values, variance_values)
# plt.xticks(k_values)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('bias')
# f.show()
#===============================================================================


#plt.plot(k_values, bias_values)


#print X

#===============================================================================
# print len(X)
# print len(np.vstack(set(map(tuple, X))))
#===============================================================================

#===============================================================================
# #Generation of a 2-dimensional dataset with 1000 examples
# X, y = make_blobs(n_samples=10000, centers=3, n_features=2,    random_state=0)
# 
# 
# 
# k_values = [ 2 , 3,5 , 7 ,10 , 13 , 15 , 18 ,20 , 23, 25 , 28 ,30 , 35 ,40 ,43 , 45 ,50 ]
# k_scores = []
# for k in k_values:
#     knn = KNeighborsClassifier(n_neighbors=k , weights='uniform'   )
#     scores = cross_val_score(knn, X, y, cv=10)
#     k_scores.append(scores.mean())
# print(k_scores)
# # plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis) or the metric
# plt.plot(k_values, k_scores)
# plt.xticks(k_values)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Accuracy')
# plt.show()
# #===============================================================================
#===============================================================================