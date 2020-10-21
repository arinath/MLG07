# -------------------------------------------------------------------------------------------------------------------- #
#                                                                                                                      #
#                                             Machine learning Guild Bootcamp                                          #
#                                                © Deloitte Consulting LLP                                             #
#                                                                                                                      #
# -------------------------------------------------------------------------------------------------------------------- #

# Author list:
#    Name                                       Username                                     Email
#    Gert De Geyter                             gedgeyter                                    gedegeyter@deloitte.com

# -------------------------------------------------------------------------------------------------------------------- #
# ###                                               Program Description                                            ### #
# -------------------------------------------------------------------------------------------------------------------- #

# In this excercise we will use Kmeans to determine clusters in the Iris flower data set.
# The Iris dataset contains the data for 50 flowers from each of the 3 species - Setosa, Versicolor and Virginica. The
# data gives the measurements in centimeters of the variables sepal length and width and petal length and width for each
# of the flowers.
# -------------------------------------------------------------------------------------------------------------------- #
# ###                                            Includes and data load                                            ### #
# -------------------------------------------------------------------------------------------------------------------- #

# Make sure the working directory is set to project folder
# add this is if you are running a notebook
# %matplotlib notebook


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets

# -------------------------------------------------------------------------------------------------------------------- #
# ###                                                    Program                                                   ### #
# -------------------------------------------------------------------------------------------------------------------- #

# DATA: 50 flowers from each of the 3 species with the following columns
#
# Sepal.Length
# Sepal.Width
# Petal.Length
# Petal.Width

np.random.seed(5) #just so we can compare results

iris = datasets.load_iris()
X = iris.data
y = iris.target

# We assume there will be three clusters
name = 'k_means_iris_3'
est= KMeans(n_clusters=3)

fignum = 1
titles = ['3 clusters']

fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
est.fit(X)
labels = est.labels_

ax.scatter(X[:, 3], X[:, 0], X[:, 2],
           c=labels.astype(np.float), edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title(titles[fignum - 1])
ax.dist = 12
fignum = fignum + 1

# Plot the ground truth
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))

# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 0, 2]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title('Ground Truth')
ax.dist = 12

fig.show()

#try to implement silhouette and proof that three is the optimal number compare to 1-5
#...
# -------------------------------------------------------------------------------------------------------------------- #
