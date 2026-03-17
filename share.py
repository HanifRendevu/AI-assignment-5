'''
point cloud data is stored as a 2D matrix
each row has 3 values i.e. the x, y, z value for a point

Project has to be submitted to github in the private folder assigned to you
Readme file should have the numerical values as described in each task
Create a folder to store the images as described in the tasks.

Try to create commits and version for each task.
'''


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

## the pcd:,2 value is kept at pcd[:, 2] for ease of use, since we will be using it a lot to find the ground level and to plot the histogram of z values.

def get_ground_level(pcd):
    z = pcd[:, 2]
    counts, bin_edges = np.histogram(z, bins=120)
    max_bin_index = np.argmax(counts)
    ground_level = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    return ground_level

def plot_histogram(pcd, title):
    z = pcd[:, 2]
    plt.figure(figsize=(8,5))
    plt.hist(z, bins=120)
    plt.title(title)
    plt.xlabel("z value")
    plt.ylabel("count")
    plt.show()

# dataset 1
pcd1 = np.load("dataset1.npy")
ground1 = get_ground_level(pcd1)
print("Dataset 1 ground level:", ground1)
plot_histogram(pcd1, "Dataset 1 Z Histogram")

# dataset 2
pcd2 = np.load("dataset2.npy")
ground2 = get_ground_level(pcd2)
print("Dataset 2 ground level:", ground2)
plot_histogram(pcd2, "Dataset 2 Z Histogram")



'''
Task 2 (+1)

Find an optimized value for eps.
Plot the elbow and extract the optimal value from the plot
Apply DBSCAN again with the new eps value and confirm visually that clusters are proper

https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/
https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

For both the datasets
Report the optimal value of eps in the Readme to your github project
Add the elbow plots to your github project Readme
Add the cluster plots to your github project Readme
'''




#%%
'''
Task 3 (+1)

Find the largest cluster, since that should be the catenary, 
beware of the noise cluster.

Use the x,y span for the clusters to find the largest cluster

For both the datasets
Report min(x), min(y), max(x), max(y) for the catenary cluster in the Readme of your github project
Add the plot of the catenary cluster to the readme

'''
