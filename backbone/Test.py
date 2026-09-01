import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display,clear_output
from scipy.stats import sem
from sklearn.metrics import f1_score




def KNN_accuracy(rep,labels):
    accuracy= []

    for random_state in np.random.randint(1,10000,10):
        X_train, X_test, y_train, y_test = train_test_split(rep, labels, test_size=0.2, random_state=random_state)

        # Define the model

        neigh = KNeighborsClassifier(n_neighbors = 5)

        #Train

        neigh.fit(X_train, y_train)

        acc = sum([neigh.predict(X_test) == y_test][0])/len(y_test)

        accuracy.append(acc)
    m_accuracy = np.mean(accuracy)
    var = sem(accuracy)

    
    return round(m_accuracy*100,2),round(var*100,2)


def kmeans(reps = None, n_clusters = None):
        # Define the number of clusters    
    # Create a KMeans instance with the desired number of clusters
    kmeans = KMeans(n_clusters=n_clusters)
    
    # Fit the model to the data
    kmeans.fit(reps)
    
    # Get the cluster centers and labels
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    return labels

def clustering_accuracy(reps_ids = None,
                        labelled = None,
                       n_clusters = 10):
    
    true_labels,labelled_ids = labelled

    all_reps = reps_ids[0]
    all_reps_ids = reps_ids[1]

    """
    1: Assign KMeans labells to source ids
    """

    k_labels = kmeans(reps = reps_ids[0], n_clusters = n_clusters)

    """
    Find the cluster assignments of the labelled subsets
    Below is the K means assignments of the labelled sub-class in order of the labelled
    sub-class ids
    """
    k_labels_sub = [k_labels[all_reps_ids.index(source_id)] for source_id in set(labelled_ids) & set(all_reps_ids)]

    #remove ids not vound in the bigger list of representations
    true_labels = [true_labels[labelled_ids.index(source_id)] for source_id in set(labelled_ids) & set(all_reps_ids)]

    
    k_labels_sub = np.array(k_labels_sub)
    true_labels = np.array(true_labels)
    print("Length subset used: ",len(true_labels))
    """
    2: For each kmeans cluster, find the most common true label
    """
    correct = 0
    for cluster_id in np.unique(k_labels_sub):
        #obtatin the locations of all the clusters with the ids
        cluster_mask = (k_labels_sub == cluster_id) 
        true_labels_in_cluster = true_labels[cluster_mask]
    
        # Find the most common label in this cluster
        most_common_true_label = Counter(true_labels_in_cluster).most_common(1)[0][0]
        
        # Count how many samples match the most common label
        correct += np.sum(true_labels_in_cluster == most_common_true_label)
        

        
    print("Clustering accuracy " + str(correct/len(true_labels)))
    return correct/len(true_labels)
    
        
def KNN_f1(rep,labels, classes =3 ):
    f1_scores = []
    
    for random_state in np.random.randint(1,10000,50):
        X_train, X_test, y_train, y_test = train_test_split(rep, labels, test_size=0.2,random_state=random_state)

        # Define the model

        neigh = KNeighborsClassifier(n_neighbors = 5)

        #Train

        neigh.fit(X_train, y_train)
        f1_scored = [None,None,None]
        for i in range(classes):
            f1_scored[i] = f1_score(y_test, neigh.predict(X_test), average=None)[i]
        print(f1_scored)
        f1_scores.append(f1_scored) 
    edge = [a[0] for a in f1_scores]
    ellip = [a[1] for a in f1_scores]
    spir = [a[2] for a in f1_scores]

    
    return (np.mean(edge),np.mean(ellip),np.mean(spir)),(sem(edge),sem(ellip),sem(spir))



def silhuoette(rep,labels):

    #umap = viz.umap(rep,dim = 10,scatter = False)
    umap = viz.pca(data = rep,n_components = min([512,len(rep[1])]), variance = 0.95)
        
    sil = metrics.silhouette_score(rep,labels, metric = "euclidean", n_jobs = -1)
    
    return sil