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
#Generation of a 2-dimensional dataset with 1000 examples
X, y = make_blobs(n_samples=1000, centers=3, n_features=2,    random_state=0)

X = DataFrame(X, columns={'dim1' , 'dim2'})
print X.head()
y = DataFrame(y, columns={'target'})
print y.head()

k_values = [ 2 , 3,5 , 7 ,10 , 13 , 15 , 18 ,20 , 23, 25 , 28 ,30 , 35 ,40 ,43 , 45 ,50 ]
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



def diff(a ,b):
    b= set(b)
    return [ x for x in a if x not in b ]   

#===============================================================================
# bias_k= []
# var_k =[]
#===============================================================================
var_bias_df = y 
k_values = [20]
for k in k_values:
    scores = []
    for i in range(1):
        training_samples,  training_labels = resample(X, y)
        test_index = diff(X.index , training_samples.index.unique() )
        test_samples = X.iloc[test_index,:]
        print len(test_samples)
        test_labels =  y.iloc[test_index,:] 
        knn = KNeighborsClassifier(n_neighbors=k , weights='uniform'   )
        knn.fit(training_samples, training_labels.target.values)
        preds = knn.predict(test_samples)
        booster_preds = DataFrame( preds , index=test_samples.index , columns = ['booster_' + str(i) ]) 
        var_bias_df = var_bias_df.join(booster_preds)
        print "preds"
        print preds
        print len(preds)
        print var_bias_df
        print var_bias_df.apply(lambda x:x.isnull().sum(), axis = 0)
        score = knn.score(test_samples, test_labels.target.values)
        scores.append(score)
        print score
    print "==========="
    #===========================================================================
    # bias_k.append(np.mean(scores))
    # var_k.append( np.var(scores))
    #===========================================================================

#===============================================================================
# plt.plot(k_values, bias_k)
# #plt.plot(k_values, var_k)
# plt.xticks(k_values)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('bias')
# plt.show()
#===============================================================================


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