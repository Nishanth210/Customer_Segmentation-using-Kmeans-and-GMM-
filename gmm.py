import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

dataset = pd.read_csv('Mall_Customers.csv')
dataset=np.array(dataset)
X = dataset[:, [3, 4]]


gmm = GaussianMixture(n_components = 4)
labels =gmm.fit_predict(X)
gmm.predict(X)
# print(labels)
# print(gmm.predict([[1000,10]]))
pickle.dump(gmm, open('gmm.pkl','wb'))
modell = pickle.load(open('gmm.pkl','rb'))

# # Assign a label to each sample
# labels = gmm.predict(d)
# d['labels']= labels
# d0 = d[d['labels']== 0]
# d1 = d[d['labels']== 1]
# d2 = d[d['labels']== 2]
# d3 = d[d['labels']== 3]
 
# # plot three clusters in same plot
# plt.scatter(d0[0], d0[1], c ='r')
# plt.scatter(d1[0], d1[1], c ='yellow')
# plt.scatter(d2[0], d2[1], c ='g')
# plt.scatter(d2[0], d3[1], c ='b')