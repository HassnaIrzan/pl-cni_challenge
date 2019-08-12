#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:19:37 2019

@author: HassnaIrzan
"""
from load_data import get_classification_data

from joblib import  load
import numpy as np

def save_file(file_name, data):
    
    with open(file_name, "w") as f:
        np.savetxt(f, data.astype(int), fmt='%i', delimiter=",")
    

def predict_diagnosis(input_dir, ouput_dir, classifier):
    
    atlas="aal"
    
    conn_coefs = get_classification_data(input_dir, atlas)

    classifier_fit = load(classifier)

    diagnosis_predict = classifier_fit.predict(conn_coefs)

    # Output and save
    output_prediction_file_name = ouput_dir+'classification.txt'
    
    save_file(output_prediction_file_name, diagnosis_predict)
    

