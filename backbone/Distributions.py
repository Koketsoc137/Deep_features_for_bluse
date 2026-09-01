import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.neighbors import KDTree
from sklearn.cluster import KMeans


#matplotlib.rc('font', **font)

import numpy as np

def reduced_chi_square(observed, errors, expected = None):
    
    chi_square = np.nansum([o**2/error**2 for o,error in zip(observed,errors)])

    return chi_square/np.count_nonzero(~np.isnan(chi_square))
    
def norm(observed, expected = None):
    
    norm = np.nansum([observed**2])

    return norm/np.count_nonzero(~np.isnan(observed))




def scale_and_sample(pca_features, sub_sample_size = 8000, n_output_features = 20, seed = 42):

    #obtain an aray of the elements of highest variance
    first_elements = pca_features[:,0]
    #compute the mean and std
    mean = np.mean(first_elements)
    std_dev = np.std(first_elements)
    np.seed = seed
        
    #scale the representations
    scaled_rep = np.array([(representation-mean)/std_dev for representation in pca_features])

    #Subsample
    
    sampled = scaled_rep[np.random.choice(pca_features.shape[0], sub_sample_size, replace=False)]
    #Reduce the dimension, get only the first n features 
    smaller = []
    for sample in sampled:
        smaller.append(sample[:n_output_features])

    return smaller
    
def reduced_nonchi_square(observed, errors, expected = None):

    nonchi_square = np.sum(observed)
    nonchi_error = np.sum(errors)

    return nonchi_square/len(observed), nonchi_error/len(errors)#/np.count_nonzero(~np.isnan(observed)),nonchi_error/np.count_nonzero(~np.isnan(nonchi_error))



def symmetric_covariance(cov_diag = 2.5,dimension = 2, number = 2):

    
    return [[[cov_diag if i == j else 0 for j in range(dimension)] for i in range(dimension)] for i in range(number)]

    


def first_order_structure(bootstraps):

    Nbootstrap = len(bootstraps)
    squared_bootstraps = np.zeros((Nbootstrap, len(bootstraps[0])))
                          
    for i in range(Nbootstrap):
                              
        squared_bootstraps[i] = np.array([o**2 for o in bootstraps[i]])
    
    #Compute the mean and error, ignore the nans
    
    corr = np.ma.masked_invalid(squared_bootstraps).mean(0)
    error = np.ma.masked_invalid(squared_bootstraps).std(0)

    return corr,reduced_nonchi_square(corr, error, expected = None)



# Step 1: Generate random points in 2D space
# Function to generate random points on a disk
def generate_points_on_disk(radius, n_points, center = [0,0]):
    """
    Generate random points uniformly on a disk of given radius.

    Args:
        radius (float): Radius of the disk.
        n_points (int): Number of random points to generate.

    Returns:
        points (np.ndarray): Array of shape (n_points, 2) containing random points on the disk.
    """
    # Generate random angles uniformly between 0 and 2pi
    angles = np.random.uniform(0, 2 * np.pi, n_points)
    
    # Generate random radii with proper distribution for uniform sampling
    radii = radius * np.sqrt(np.random.uniform(0, 1, n_points))
    
    # Convert polar coordinates to Cartesian coordinates
    x = radii * np.cos(angles) +center[0]
    y = radii * np.sin(angles)  +center[0]
    
    # Combine x and y into a single array
    points = np.column_stack((x, y))
    
    return points


def generate_random_points_nd(n_points,s_l = 1,dimension = 2, seed = 42):
    """
    Generate random points in 2D space.

    Args:
        n_points (int): Number of points to generate.
        grid_size (int): Size of the 2D grid.

    Returns:
        points (np.ndarray): 2D array of points (n_points, 2).
    """
    points = np.random.uniform(-s_l, s_l, size=(n_points, dimension))
    
    return points



def generate_gaussian_points(mean, cov, n_points, dimensions=2,seed = 42):
     ##points sampled from a gaussian distribution
    np.random.seed(seed)

    
    points = np.random.multivariate_normal(mean, cov, n_points)

    
    return points


def sample_means_and_covariances(dimensions, mean_range, cov_range, num_samples,seed = 42):

    # Sample means from a uniform distribution
    np.random.seed(seed)
    means = np.random.uniform(mean_range[0], mean_range[1], size=(num_samples, dimensions))
    space_center = 50
    mean_array = []

    space_center = (mean_range[0] - mean_range[1])/2
    
    # Sample covariance matrices
    covariances = []
    #covariances_sparse = []
    for _ in range(num_samples):
        # Generate a random matrix and force it to be positive semi-definite
        A = np.random.uniform(cov_range[0], cov_range[1], size=(dimensions, dimensions))
        covariance_matrix = np.dot(A, A.T)  # A * A.T ensures the matrix is positive semi-definite
        covariances.append(covariance_matrix)

       # sparse_matrix = np.copy(covariance_matrix)

       # np.fill_diagonal(sparse_matrix, np.diagonal(sparse_matrix)*2)
  
       # covariances_sparse.append(sparse_matrix)

       
    covariances = np.array(covariances)

    
  #  covariances_sparse = np.array(covariances_sparse)
    
    return means, covariances#, covariances_sparse



def scatter_points(points,alpha = 0.2, title = ""):
    fig = plt.figure(dpi = 300)
    plt.style.use("default")
    plt.figure(figsize=(15,10))
    plt.rcParams.update({'font.size': 20}) 
    plt.scatter(points[:, 0], points[:, 1], alpha=alpha)
    plt.title(title)
    #plt.xlim(-00,500)
    #plt.ylim(-500,500)
    plt.axis('equal')
    plt.show()


def scatter_overlay(points1, points2):
    plt.scatter(points1[:, 0], points1[:, 1], color='blue', label='Set 1',s = 0.05)
    plt.scatter(points2[:, 0], points2[:, 1], color='red', label='Set 2', s = 0.05)
    plt.legend()
    plt.show()

# Step 4: Plot the 2-point autocorrelation function
def plot_autocorrelation_2d(separation, correlation,title = "Correlation"):
    """
    Plot the 2-point autocorrelation function.

    Args:
        bin_centers (np.ndarray): Bin centers for the autocorrelation.
        correlation (np.ndarray): Autocorrelation values in each bin.
    """
    fig = plt.figure(dpi = 300)
    plt.style.use("default")
    plt.figure(figsize=(15,10))
    plt.rcParams.update({'font.size': 20}) 
    plt.plot(separation, correlation, marker='o')
    plt.title(title)
    plt.xlabel('Separation')
    plt.ylabel('Frequency of separation')
    plt.grid(False)
    plt.show()

def kmeans(data, 
           n_clusters = 2):
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Example with 3 clusters
    
    kmeans.fit(data)
    
    return kmeans.labels_


def precompute_RR(bins = np.linspace(0, 1.5, 100),
                           dimension = 3,
                           n_points =50000, 
                           metric = "euclidean",
                           use_stored = False,
                           background = None,
                           statistics = "Gaussian",
                           Eff_cov = None,
                        ):

    
    

    if statistics == "Gaussian" and Eff_cov is None:
        raise ValueError("Eff_cov cannot be none for Gaussin statistics")

        
    if use_stored:
        return np.array([448028, 1330494, 2198238, 3046892, 3876978, 4693396, 
                5486932, 6266590, 7028676, 7775688, 8513268, 9226058, 
                9924878, 10606278, 11278858, 11936514, 12561862, 13183042,
                13792434, 14370928, 14942908, 15494356, 16045886, 16579952,
                17083098, 17577980, 18066512, 18530036, 18988736, 19422774,
                19852576, 20250260, 20638884, 21019068, 21385978, 21748132,
                22085560, 22403576, 22709800, 23017374, 23301002, 23553432,
                23824386, 24056018, 24271838, 24486842, 24685586, 24871376,
                25054644, 25196824, 25362646, 25521042, 25652240, 25768780,
                25877726, 25985252, 26068114, 26134084, 26193962, 26255878,
                26274226, 26306128, 26302730, 26317506, 26316020, 26286668,
                26268222, 26230330, 26197146, 26146488, 26075090, 25998468,
                25909506, 25799002, 25673556, 25565654, 25428602, 25300042,
                25150500, 24991880, 24816718, 24637116, 24461706, 24285032,
                24081964, 23882128, 23658832, 23415816, 23179138, 22913860,
                22663944, 22373258, 22071804, 21795744, 21519664, 21193786,
                20885754, 20570808, 20244604])

    else:
    
    
    
        Eff_mean = np.zeros(dimension)
    
        if background is None:

            if statistics == "Uniform":
            
                  
                background = generate_random_points_nd(n_points,s_l = 1,dimension = dimension, seed = 42)
    
            elif statistics == "Gaussian":
    
                background = generate_gaussian_points(Eff_mean, 
                                                     Eff_cov,
                                                     n_points, 
                                                    dimensions = dimension,
                                                    seed = random.randint(0,10000))
    
        
    
        #Obtrain the distance distribution
    
        KDT_R = KDTree(background,
                       metric = metric)
    
        counts_RR = KDT_R.two_point_correlation(background, 
                                                bins)
    
        RR = np.diff(counts_RR)
    
        return RR
        

