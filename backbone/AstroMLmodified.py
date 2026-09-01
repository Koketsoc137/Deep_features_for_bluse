import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
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
import matplotlib.pyplot as plt


def norm(observed,
        errors = None,
        background_factor= 10,
        bins =[],
        verbose = False):

    """
    Remove all the zeros because the will not caoult anyway
    """
    
    errors, DD = errors
    o_zero = (observed == 0)
    observed[o_zero] = np.nan

    
    """
    Zero poission errors correspond to bins without pair counts, blows up the TPCF score whe included
    """
    errors_zero = (errors == 0)
    errors[errors_zero] = np.nan
    

    valid = (~np.isnan(observed))&(~np.isnan(errors))


    observed = observed[valid]
    errors = errors[valid]
    
    DD  = DD[valid]
    total_pairs = sum(DD)

    print(len(observed))

    norm = np.nansum([ (o)**2/e**2 for b,o,e in zip(bins,observed,errors)])/total_pairs
    #norm = np.nansum(observed)
    return norm






def two_point(data, 
              bins,
              method='standard',
              data_R=None,
              precomputed_RR = None,
              background_factor = 1,
              sub_sample_fraction =0.7,
              random_state=None):
    
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
    if precomputed_RR is None:
        if data_R is None:
            data_R = data.copy()
            for i in range(n_features - 1):
                rng.shuffle(data_R[:, i])
        else:
            data_R = np.asarray(data_R)
            if (data_R.ndim != 2) or (data_R.shape[-1] != n_features):
                raise ValueError('data_R must have same n_features as data')
    
        factor = len(data_R) * 1. / len(data)

    else:
        factor = background_factor/sub_sample_fraction


    # Fast two-point correlation functions added in scikit-learn v. 0.14
    KDT_D = KDTree(data)
    counts_DD = KDT_D.two_point_correlation(data, bins)
    DD = np.diff(counts_DD)

    if precomputed_RR is None:

        if data_R is None:
            raise ValueError("No background; no precomputed RR")
        else:

            KDT_R = KDTree(data_R)
            counts_RR = KDT_R.two_point_correlation(data_R, bins)
            RR = np.diff(counts_RR)
    else:
        RR = precomputed_RR

    # check for zero in the denominator
    RR_zero = (RR == 0)
    RR[RR_zero] = 1

    if method == 'standard':
        corr = factor ** 2 * DD / RR - 1
        
    elif method == 'landy-szalay':

        """
        The precompute speeding only works with the standard method
        """
        if precomputed_RR is not None:
            
            raise ValueError(" The precompute speeding only works with the standard method")
        else:
            
            counts_DR = KDT_R.two_point_correlation(data, bins)
    
            DR = np.diff(counts_DR)
    
            corr = (factor ** 2 * DD - 2 * factor * DR + RR) / RR

    corr[RR_zero] = np.nan

    DD_zero = (DD == 0)
    DD[DD_zero] = 1

    
    corr_err =  np.asarray([math.sqrt(1+cor)/math.sqrt(d) for cor,d in zip(corr,DD)])

    return corr, (corr_err,DD)

    


def bootstrap_two_point(data, 
                        bins, 
                        Nbootstrap=10,
                        method='standard', 
                        return_bootstraps=False,
                        random_state=None,
                        data_R = None,
                        background_factor = 5,
                        sub_sample_fraction =0.7,
                        flatten_reps = True,
                        representations =None,
                        precomputed_RR = None):

    
    """Bootstrapped two-point correlation function

    Parameters
    ----------
    data : array_like
        input data, shape = [n_samples, n_features]
    bins : array_like
        bins within which to compute the 2-point correlation.
        shape = Nbins + 1
    Nbootstrap : integer
        number of bootstrap resamples to perform (default = 10)
    method : string
        "standard" or "landy-szalay".
    return_bootstraps: bool
        if True, return full bootstrapped samples
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background

    Returns
    -------
    corr, corr_err : ndarrays
        the estimate of the correlation function and the bootstrap
        error within each bin. shape = Nbins
    """

    
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

    

    bootstraps = np.zeros((Nbootstrap, len(bins[1:])))

    corr_error = None
        
    for i in range(Nbootstrap):
        
        stamp_1 = time.time()
        indices = random.sample(range(n_samples),int(n_samples*sub_sample_fraction))

        bootstraps[i], corr_err = two_point(data[indices, :],
                                  data_R = data_R,
                                  bins = bins, 
                                  method=method,
                                  precomputed_RR=precomputed_RR,
                                  background_factor = background_factor,
                                  sub_sample_fraction = sub_sample_fraction,
                                  random_state=rng)
                        

    if return_bootstraps:
        return bootstraps,corr_err
    else:
        # use masked std dev in case of NaNs
        corr = np.ma.masked_invalid(bootstraps).mean(0)

        
        return corr, corr_err


def correlate_and_plot(data = list,
                       max_dist = 1.5,
                       min_dist=0,
                       bin_number = 100,
                       plot = False,
                       bootstrap = True,
                       Nbootstrap = 5,
                       representations = [],
                       precomputed_RR = None,
                       precompute = False,
                       background = None,
                       background_factor = 1,
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
    
    #Scaling confuses cluster separation
    
    distances = np.linalg.norm(data, axis=1)
    
    
    max_dist = distances.max()

    data = data/max_dist

    max_dist = np.percentile(np.linalg.norm(data, axis=1), 68)*2

    #Chopping up the space,importtants
    base = 10
    """
    bins = np.logspace(np.log10(max_dist/bin_number)/np.log10(base),
                       np.log10(max_dist),
                       bin_number,
                       base = base)
    
    """
    bins = np.linspace(min_dist,
                       max_dist, 
                       bin_number)



    if precomputed_RR is None:

            if verbose:
                if verbose > 2:
                    print("Computing background and RR distributions: will be slower")

    
            Eff_cov = np.cov(data,rowvar = False)
            Eff_mean = np.mean(data, axis = 0)
        
            length, dimension = data.shape
        
            #Percentile of the scaled data

            if method == "standard":
            

                if precompute:
                    
                    precomputed_RR =  dist.precompute_RR(bins = bins,
                                                           dimension = dimension,
                                                           n_points =background_factor*len(data), 
                                                           metric = "euclidean",
                                                           use_stored = False,
                                                           background = None,
                                                           statistics = "Gaussian",
                                                           Eff_cov = Eff_cov,
                                                           )
                else:
                    
                    background = dist.generate_gaussian_points(mean = Eff_mean, 
                                                                 cov = Eff_cov,
                                                                 n_points = background_factor*len(data), 
                                                                 dimensions = dimension,
                                                                 seed = random.randint(0,10000))


    

    if bootstrap:
        bootstraps,poisson_error = bootstrap_two_point(data, bins, 
                                                        data_R = background,
                                                        background_factor = background_factor,
                                                        precomputed_RR = precomputed_RR,
                                                        Nbootstrap=Nbootstrap,
                                                        sub_sample_fraction =0.3,
                                                        method = method,  
                                                        return_bootstraps =True,
                                                        flatten_reps = False,
                                                        representations = representations,
                                                        )
    
        
        corr = np.ma.masked_invalid(bootstraps).mean(0)
        dcorr = np.asarray(np.ma.masked_invalid(bootstraps).std(0, ddof=1))

    else:
        corr, dcorr = two_point(data,
                                      data_R = background,
                                      bins = bins, 
                                      method=method,
                                      precomputed_RR=precomputed_RR,
                                      background_factor = background_factor,
                                      sub_sample_fraction =1,
                                      random_state=42)
                            

    NormScore = norm(corr,
                     errors =poisson_error,
                     background_factor= background_factor,
                     bins =bins)
        
    
    #print("Repley's K: ",NormScore)

    
    if plot:
        fig = plt.figure(dpi = 300)
        plt.style.use("default")
        plt.figure(figsize=(15,10))
        plt.rcParams.update({'font.size': 20})
        plt.plot(bins[1:],corr)
        plt.fill_between(bins[1:],corr-dcorr, corr+dcorr, color = "blue",alpha = .3)
        plt.title(label)
        plt.savefig(fig_name+".png")
        plt.show()
        return NormScore


    else:
        if return_corr:
            return corr,dcorr,(NormScore,0.00001)
        else:
            return NormScore


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

    representations = np.array([arr.tolist() for arr in representations])
    norm_score = []

    for i in range(Nbootstrap):
        #viz.shade(val_flat, predictions = [0]*len(val_flat))
        indices = random.sample(range(len(representations)),int(len(representations)*sub_sample))
        val_flat_sample =  viz.pca(representations[indices,:],
                                   n_components = 2,
                                  verbose = verbose)
    

        plot = False
        scatter = False

        
        norm_score_ = correlate_and_plot(val_flat_sample,
                                        min_dist = 0.0,
                                        max_dist =1.5,
                                        label = "Correlation on flat manifold for epoch:"+str(epoch),
                                        fig_name = "plots/2PCR@Epoch: "+str(epoch),
                                        precomputed_RR = None,
                                        bin_number = 100,
                                        method = "standard",
                                        bootstrap = False,
                                        plot = plot,
                                        background_factor = 1,
                                        representations = [])

        norm_score.append(norm_score_[0])


    return (round(np.ma.masked_invalid(norm_score).mean(0),2),round(np.ma.masked_invalid(norm_score).std(0, ddof=1),2))

            

def id_score(representations,SubSampleFraction = 0.3, Nsamples = 5,verbose = False):
        twonn = skdim.id.TwoNN()
        RepSize = len(representations)
        IDs = []
        representations = np.array(representations)
        for i in range(Nsamples):
            indices = random.sample(range(RepSize),int(RepSize*SubSampleFraction))
            ID = twonn.fit_transform(representations[indices,:])
            IDs.append(ID)
            if verbose:
                print("ID :",ID)
        return np.mean(IDs, axis = 0),np.std(IDs,axis = 0, ddof=1)




