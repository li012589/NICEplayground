import numpy as np

def binning_analysis(samples,bins):
    #Perform a binning analysis over samples and return an array of the error estimate at each binning level.
    minbins = 2**bins # minimum number of bins (128 still seems to be a reasonable sample size in most cases)
    maxlevel = int(np.log2(len(samples)/minbins))
    maxsamples = int(minbins * 2**(maxlevel))
    bins = np.array(samples[-maxsamples:]) # clip to power of 2 for simplicity
    errors = np.zeros(maxlevel+1)
    for k in range(maxlevel+1):
        errors[k] = np.std(bins)/np.sqrt(len(bins)-1.)
        bins = np.array([(bins[2*i]+bins[2*i+1])/2. for i in range(len(bins)//2)])
    return errors

def auto_time(errors):
  return  .5*(errors[-1]**2/errors[0]**2 - 1.)

def autoCorrelationTime(samples,bins):
    return auto_time(binning_analysis(samples,bins))