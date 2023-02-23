import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

dataset = pd.read_csv('Mall_Customers.csv')
dataset=np.array(dataset)
X = dataset[:, [3, 4]]



wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
# print(kmeans.predict([[10,10]]))


model=kmeans(n_clusters=25,init='K-means++',random_state=32)
model.fit(y_kmeans)

pickle.dump(kmeans, open('csml.pkl','wb'))
model = pickle.load(open('csml.pkl','rb'))

