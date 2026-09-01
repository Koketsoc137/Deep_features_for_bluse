
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
from sklearn.utils import check_random_state
import random
try:
    import backbone.Distributions as dist
    import backbone.VISUAL as viz

except:
    import spectra_for_features.backbone.Distributions as dist
    import spectra_for_features.backbone.VISUAL as viz
    
import math
import skdim
import time
import importlib


def norm(observed,
        rrors = None,
        background_factor= 10,
        bins =[],
        verbose = False):

    valid = (~np.isnan(observed))
    observed = observed[valid]
    norm = np.nansum([o for b,o in zip(bins,observed)])
    #norm = np.nansum(observed)
    return norm




def two_point(data, bins, method='standard',
              data_R=None, random_state=None):
    """Two-point correlation function

    Parameters
    ----------
    data : array_like
        input data, shape = [n_samples, n_features]
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    method : string
        "standard" or "landy-szalay".
    data_R : array_like (optional)
        if specified, use this as the random comparison sample
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background

    Returns
    -------
    corr : ndarray
        the estimate of the correlation function within each bin
        shape = Nbins
    """
    data = data / np.linalg.norm(data, axis=1, keepdims=True)
    data_R = data_R / np.linalg.norm(data_R, axis=1, keepdims=True)

    data = np.asarray(data)
    bins = np.asarray(bins)
    rng = check_random_state(random_state)

    if method not in ['standard', 'landy-szalay']:
        raise ValueError("method must be 'standard' or 'landy-szalay'")

    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    if data.ndim == 1:
        data = data[:, np.newaxis]
    elif data.ndim != 2:
        raise ValueError("data should be 1D or 2D")

    n_samples, n_features = data.shape

    # shuffle all but one axis to get background distribution
    data_R = None
    if data_R is None:
        data_R = data.copy()
        for i in range(n_features - 1):
            rng.shuffle(data_R[:, i])
    else:
        data_R = np.asarray(data_R)
        if (data_R.ndim != 2) or (data_R.shape[-1] != n_features):
            raise ValueError('data_R must have same n_features as data')

    factor = len(data_R) * 1. / len(data)
    
    # Fast two-point correlation functions added in scikit-learn v. 0.14
    KDT_D = BallTree(data)
    KDT_R = BallTree(data_R)

    counts_DD = KDT_D.two_point_correlation(data, bins)
    counts_RR = KDT_R.two_point_correlation(data_R, bins)


    DD = np.diff(counts_DD)
    RR = np.diff(counts_RR)
    print(RR)

    # check for zero in the denominator
    RR_zero = (RR == 0)
    RR[RR_zero] = 1

    if method == 'standard':
        corr = factor ** 2 * DD / RR - 1
    elif method == 'landy-szalay':
        counts_DR = KDT_R.two_point_correlation(data, bins)

        DR = np.diff(counts_DR)

        corr = (factor ** 2 * DD - 2 * factor * DR + RR) / RR

    corr[RR_zero] = np.nan

    return corr

def correlate_and_plot(data = list,
                       max_dist = 1.5,
                       min_dist=0,
                       bin_number = 100,
                       plot = False,
                       bootstrap = True,
                       Nbootstrap = 1,
                       representations = [],
                       precomputed_RR = None,
                       background = None,
                       background_factor = 20,
                       method = "standard",
                       label = "correlation on features",
                       fig_name ="tpcor",
                       return_corr = False,
                       verbose = False):


    """
    Scale the data into a unit block. Center and pull the furthest point in to the edge of such a box

    """
    #Center, scale down the sample

    Eff_mean = np.mean(data, axis = 0)

    #Center to 0,0,0,...
    data = data - Eff_mean
    
    #Scale by finding the furtherst point (or 95th percentile to aviod artifacts or statistical flukes)
    
    
    distances = np.linalg.norm(data, axis=1)
    max_dist = np.percentile(np.linalg.norm(data, axis=1), 100)*2

    data = data/max_dist

    max_dist = np.percentile(np.linalg.norm(data, axis=1), 68)*2


    bins = np.linspace(0,
                       1, 
                       bin_number)


    Eff_cov = np.cov(data,rowvar = False)
    Eff_mean = np.mean(data, axis = 0)
        
    length, dimension = data.shape
        

                    
    background = dist.generate_gaussian_points(mean = Eff_mean, 
                                                cov = Eff_cov,
                                                n_points = background_factor*len(data), 
                                                dimensions = dimension,
                                                seed = random.randint(0,10000))




    corr =  two_point(data,
            data_R = background,
            bins = bins, 
            method=method,
            random_state=42)
                            

    score = norm(corr,
                bins =bins)

    return score
        




def TPCF_score(representations,
               epoch = 0,
               sub_sample = 0.3,
               Nbootstrap = 5, 
               verbose = False):
    """
    The input is hi-dimesional representations, the 2PCF score is computed on the the first 2 PCA components
    of multiple subsets of the representations
    """

    #For compatibility with pytorch deep representations
    #if type(representations) is not list:
    if isinstance(representations[0],list):
        representations = np.array(representations)

    else:
        representations = np.array([arr.tolist() for arr in representations])
    
    norm_score = []

    for i in range(Nbootstrap):



        #Flatten
        indices = random.sample(range(len(representations)),
                                int(len(representations)*sub_sample)
                               )
        
        val_flat_sample =  viz.pca(representations[indices,:],
                                   n_components = 15,
                                  verbose = verbose)


        #compute correlation


        tpcf_score = correlate_and_plot(val_flat_sample,
                                        min_dist = 0.0,
                                        max_dist =1.5,
                                        label = "Correlation on flat manifold for epoch:"+str(epoch),
                                        fig_name = "plots/2PCR@Epoch: "+str(epoch),
                                        precomputed_RR = None,
                                        bin_number = 100,
                                        method = "standard",
                                        bootstrap = False,
                                        background_factor = 10,
                                        representations = [])

        


        norm_score.append(tpcf_score)


    return (round(np.ma.masked_invalid(norm_score).mean(0),2),round(np.ma.masked_invalid(norm_score).std(0, ddof=1),2))
    
def id_score(representations,sub_sample = 0.3, Nsamples = 5,verbose = False):
        twonn = skdim.id.TwoNN()
        RepSize = len(representations)
        IDs = []
        representations = np.array(representations)
        for i in range(Nsamples):
            indices = random.sample(range(len(representations)),
                                min(5000,int(len(representations)*sub_sample))
                               )            
            ID = twonn.fit_transform(representations[indices,:])
            IDs.append(ID)
            if verbose:
                print("ID :",ID)
        return np.mean(IDs, axis = 0),np.std(IDs,axis = 0, ddof=1)
