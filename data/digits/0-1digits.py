import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

i,j = 0,1
precision = 4

# filenames
data_filename = "csv/train-%d-%d.txt" % (i,j)
output_filename = "neuron/digits-%d-%d.neuron" % (i,j)

# load dataset
dataset = pd.read_csv(data_filename)
train_features = dataset
train_labels = train_features.pop("1.48")

# train neuron
model = LogisticRegression(penalty='l1',solver='liblinear',C=.002,random_state=0) #tol=1e-8
classifier = model.fit(train_features,train_labels)

# report accuracy by sklearn
train_accuracy = 100*classifier.score(train_features,train_labels)
print("LogisticRegression: %.8f%% (training accuracy)" % (train_accuracy,))

# self-compute accuracy
A = np.array(train_features)
y = np.array(train_labels)
w = np.array(model.coef_).T 
b = np.array(model.intercept_) 
acc = sum((A@w > -b).flatten() == y)/len(y)
print("       my accuracy: %.8f%%" % (100*acc,))

# round everything to integer, and recompute accuracy
print_integer = True
alpha = 10**precision
w = np.round(w*alpha)
b = np.round(b*alpha)
acc = sum((A@w > -b).flatten() == y)/len(y)
print("      int accuracy: %.8f%%" % (100*acc,))

def write_parameter(file1,parameter):
    if print_integer:
        #parameter = alpha*parameter
        file1.write("%d" % parameter)
    else:
        file1.write("%f" % parameter)

# save neuron to file
file1 = open(output_filename,"w")
file1.write("name: example")
file1.write("\nsize: %d" % len(w))
file1.write("\nweights: ")
for parameter in w:
    file1.write(" ")
    write_parameter(file1,parameter)
file1.write("\nthreshold: ")
write_parameter(file1,-b)
file1.write("\n ")

# count the number of non-zero weights
non_zero_count = sum(1 for a in w if a != 0)
print("# of non-zero weights: %d" % non_zero_count)
