import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, render_template,request
import pickle
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics

app = Flask(__name__)
model = pickle.load(open('csml.pkl', 'rb'))
modell = pickle.load(open('gmm.pkl','rb'))


@app.route('/')
def home():
    return render_template('gmm.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    #For rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 1) 
    return render_template('Kmeans.html', prediction_text='Using K-Means Clustering You belong to Cluster:{}'.format(output))

@app.route('/visualize',methods=['POST','GET'])
def visualize():
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
    
    plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    output=plt.show()
    return render_template('Kmeans.html',predition_text='graph is:{}'.format(output))

    
@app.route('/pre',methods=['POST','GET'])
def pre():
    return render_template('gmm.html')

@app.route('/kmeans',methods=['POST','GET'])
def kmeans():
    return render_template('Kmeans.html')

@app.route('/gmmpredict',methods=['POST','GET'])
def gmmpredict():
    #For rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 1) 
    return render_template('gmm.html', prediction_text='Using Guasian Mixture model You belong to Cluster:{}'.format(output))

@app.route('/gmmvisualize',methods=['POST','GET'])
def gmmvisualize():
    dataset = pd.read_csv('Mall_Customers.csv')
    dataset=np.array(dataset)
    X = dataset[:, [3, 4]]


    gmm = GaussianMixture(n_components = 4)
    labels =gmm.fit_predict(X)
    
    plt.scatter(X[gmm == 0, 0], X[gmm == 0, 1], s = 10, c = 'pink', label = 'Cluster 1')
    plt.scatter(X[gmm == 1, 0], X[gmm == 1, 1], s = 10, c = 'purple', label = 'Cluster 2')
    plt.scatter(X[gmm == 2, 0], X[gmm == 2, 1], s = 10, c = 'orange', label = 'Cluster 3')
    plt.scatter(X[gmm == 3, 0], X[gmm == 3, 1], s = 10, c = 'blue', label = 'Cluster 4')
#plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    # plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    
    # X['labels']= labels
    # X0 = X[X['labels']== 0]
    # X1 = X[X['labels']== 1]
    # X2 = X[X['labels']== 2]
    # X3 = X[X['labels']== 3]
    
    # # plot three clusters in same plot
    # plt.scatter(X0[0], X0[1], c ='r')
    # plt.scatter(X1[0], X1[1], c ='yellow')
    # plt.scatter(X2[0], X2[1], c ='g')
    # plt.scatter(X2[0], X3[1], c ='b')
    output=plt.show()
    return render_template('gmm.html',prediction_text='Graph is:{}'.format(output))

if __name__=='__main__':
    app.run(debug=True)