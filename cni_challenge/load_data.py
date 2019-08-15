#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:42:31 2019

@author: HassnaIrzan
"""
# =============================================================================
# import packages
# =============================================================================
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
from glob import glob
import pandas as pd
import numpy as np
import re, os

def read_data(input_dir, atlas):
    
    timeseries = list()
    
    sbj_id= list()

    phenotype_all=pd.DataFrame()
    
    fmri_dir = sorted(glob(input_dir+"sub-*/"))

    for f in fmri_dir:
  
        base_name=os.path.basename(os.path.normpath(f))
       
        num=str(re.findall('\d+',base_name)[0])
      
        sbj_id.append(num)
 
        mat = np.genfromtxt(f+'/timeseries_'+atlas+'.csv', delimiter=',')
        
        
        timeseries.append(mat.transpose())
        
        phenotype_sbj = pd.read_csv(f+'/phenotypic.csv')
  
        phenotype_all = pd.concat([phenotype_all, phenotype_sbj])
        
        ts, rois = np.shape(mat.transpose())
  
    return timeseries


def get_fconnectome(timeseries, kind):
    
    connections = ConnectivityMeasure(cov_estimator=LedoitWolf(assume_centered=False, 
                            block_size=1000,store_precision=False), kind= kind, vectorize=True,
                            discard_diagonal=True)
    conn_coefs= connections.fit_transform(timeseries)
    
    return conn_coefs



def get_classification_data(input_dir, atlas):
    
    timeseries = read_data(input_dir, atlas)
    
    conn_coefs = get_fconnectome(timeseries, 'correlation')
    
    return conn_coefs
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
