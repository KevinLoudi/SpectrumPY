# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 12:54:28 2016
Propose: test data-fit method
@author: kevin
"""

#import matplotlib.pyplot as plt
#import scipy
#import scipy.stats
#size = 30000
#x = scipy.arange(size)
#y = scipy.int_(scipy.round_(scipy.stats.vonmises.rvs(5,size=size)*47))
#h = plt.hist(y, bins=range(48), color='w')
#
#dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']
#
#for dist_name in dist_names:
#    dist = getattr(scipy.stats, dist_name)
#    param = dist.fit(y)
#    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
#    plt.plot(pdf_fitted, label=dist_name)
#    plt.xlim(0,47)
#plt.legend(loc='upper right')
#plt.show()


#from fitter import Fitter
#from scipy import stats
#data = stats.gamma.rvs(2, loc=1.5, scale=2, size=100000)
#f = Fitter(data)
#f.fit()
## may take some time since by default, all distributions are tried
## but you call manually provide a smaller set of distributions
#f.summary()

#Test K-means



from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
from scipy.cluster.vq import kmeans,vq

# data generation
data = vstack((rand(150,2) + array([.5,.5]),rand(150,2)))

# computing K-Means with K = 2 (2 clusters)
centroids,_ = kmeans(data,2)
# assign each sample to a cluster
idx,_ = vq(data,centroids)

# some plotting using numpy's logical indexing
plot(data[idx==0,0],data[idx==0,1],'ob',
     data[idx==1,0],data[idx==1,1],'or')
plot(centroids[:,0],centroids[:,1],'sg',markersize=8)
show()
