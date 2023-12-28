import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.manifold import MDS
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml
from PIL import Image as im
from tqdm import tqdm

def sdi(x,y):                                                                       # similarity function used is the Sorensen-Dice similarity, or F1 similarity
  s = 2 * np.dot(x,y) / (np.linalg.norm(x) ** 2 + np.linalg.norm(y) ** 2)
  return s

def dis(X):                                                                         # creating the dissimilarity matrix D from the input datapoints using the above similarity function
  D = np.zeros((len(X), len(X)))
  for i in tqdm(range(len(X))):
    for j in range(len(X)):
      if j >= i:
        D[i,j] = 1 - sdi(X[i, :-1], X[j, :-1])
        D[j,i] = D[i,j]
  return D

(X_mnist_1, y_mnist_1), (X_mnist_2, y_mnist_2) = mnist.load_data()                  # let us consider the second MNIST dataset with 10000 samples

X_mnist_flat = X_mnist_2.reshape(10000, 784)

scaler = MinMaxScaler(feature_range=(0,255))
scaler.fit(X_mnist_flat)
X_mnist_flat = scaler.transform(X_mnist_flat)

D = dis(X_mnist_flat) 

mds = MDS(n_components=2, dissimilarity='precomputed', verbose=1, normalized_stress='auto')

X_mds=mds.fit_transform(D)

x_coords = X_mds[:, 0]                                                          # scatterplot
y_coords = X_mds[:, 1]
x_coords.shape, y_coords.shape
data_mds_2d = {'x' : x_coords, 'y': y_coords, 'label' : y_mnist_2}
df_mds_2d = pd.DataFrame(data=data_mds_2d)
df_mds_2d.head()
fig_mds_2d = px.scatter(df_mds_2d, x='x', y='y', color='label')
fig_mds_2d.show()
