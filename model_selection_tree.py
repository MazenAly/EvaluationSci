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
from sklearn import tree
import warnings

warnings.simplefilter("error")
#Generation of a 2-dimensional dataset with 1000 examples

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


X, y = make_blobs(n_samples=1000, centers=3, n_features=2,    random_state=0)

X = DataFrame(X, columns={'dim1' , 'dim2'})
y = DataFrame(y, columns={'target'})

d_values = [ 2,3,4,5,6,7,8,9,10 ]

d_misclass_rates = []
bias_values = []
variance_values = []

for d in d_values:
    var_bias_df = y 
    scores = []
    
    for i in range(100):
        training_samples,  training_labels = resample(X, y)
        test_index = diff(X.index , training_samples.index.unique() )
        test_samples = X.iloc[test_index,:]
        test_labels =  y.iloc[test_index,:] 
        clf = tree.DecisionTreeClassifier( min_samples_leaf=d )
        clf = clf.fit(training_samples, training_labels)
        preds = clf.predict(test_samples)
        booster_preds = DataFrame( preds , index=test_samples.index , columns = ['booster_' + str(i) ]) 
        var_bias_df = var_bias_df.join(booster_preds)
        #print var_bias_df.apply(lambda x:x.isnull().sum(), axis = 0)
        score = clf.score(test_samples, test_labels.target.values)
        scores.append(score)
        #print score
    print "=====Bootstrapping done======"  
    
    #saving mean misclassification rate for specific k
    d_misclass_rates.append(1- np.mean(scores))
    
    var_bias_df['bias'] = var_bias_df.apply(  loss_func ,  axis = 1)
    var_bias_df['var'] = var_bias_df.apply(  variance_func ,  axis = 1)
    
    
    bias_list = var_bias_df['bias'].values.tolist()
    var_list = var_bias_df['var'].values.tolist()
    print d
    print np.mean(list(filter(lambda x: x!= -1  , bias_list)))
    bias_values.append( np.mean(list(filter(lambda x: x!= -1  , bias_list))))
    variance_values.append( np.mean(list(filter(lambda x: x!= -1, var_list))))
    print "==========="

plt.plot(d_values, d_misclass_rates)
plt.xticks(d_values)
plt.title('Misclassification rates for 1000 datapoints with different values of d')
plt.xlabel('Value of d for Decision tree')
plt.ylabel('Misclassification rate')
plt.show()

plt.figure(1)
plt.subplot(211)
plt.plot(d_values, bias_values)
plt.xticks(d_values)
plt.xlabel('Value of d for Decision tree')
plt.ylabel('bias')

plt.subplot(212)
plt.plot(d_values, variance_values)
plt.xticks(d_values)
plt.xlabel('Value of d for Decision tree')
plt.ylabel('Variance')
plt.show()







d_misclass_rates = []


for d in d_values:
    clf = tree.DecisionTreeClassifier( max_depth=d )
    clf = clf.fit(X, y)
    scores = cross_val_score(clf, X, y['target'], cv=10)
    d_misclass_rates.append( 1- scores.mean())
print(d_misclass_rates)
# plot the value of K for KNN (x-axis) versus the cross-validated misclassification (y-axis) or the metric
plt.plot(d_values, d_misclass_rates)
plt.xticks(d_values)
plt.title('Misclassification rates for 1000 datapoints with different values of d')
plt.xlabel('Value of d for Decision tree')
plt.ylabel('Misclassification rate')
plt.show()






