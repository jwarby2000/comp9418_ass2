import pickle
import numpy as np
import pandas as pd
import scipy as sp
import heapq as pq
import matplotlib as mp
import math
from itertools import product, combinations
from graphviz import Digraph
from tabulate import tabulate
import copy
from sklearn.model_selection import KFold
import time # Added time library
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import time

from DiscreteFactors_Ass2 import Factor
from Graph_Ass2 import Graph
from BayesNet_Ass2 import BayesNet, allEqualThisIndex, NaiveBayes, HiddenMarkovModel


def learn_outcome_space(sensors, labels, room, next_ = True, extras = None):
    outcomeSpace = {}
    labels = tuple(labels)
    
    for sensor in sensors:
        if 'Motion_Sensor' in sensor:
            outcomeSpace[sensor] = ('no motion', 'motion')
        else:
            outcomeSpace[sensor] = labels
        
    outcomeSpace[room] = labels
    if next_:
        outcomeSpace[room+'_next'] = labels
    else:
        outcomeSpace[room+'_prev'] = labels
    outcomeSpace[room+'_rob1'] = labels
    outcomeSpace[room+'_rob2'] = labels
    if extras is not None:
        for extra in extras:
            outcomeSpace[extra] = labels
    
    return outcomeSpace


def assess_cost(model, dataframe, class_var, model_type='naive', p = 0.5, print_evi = False, debug = False):
    """
    Parameters
    ----------
    model : bayesian model
        Model that outputs predictions
    dataframe : pandas dataframe
        Data to use to test model
    class_var : str
        Variable to be predicted
    p: float
        Probability of empty where we turn off light

    Returns
    -------
    None.

    """
    
    # Splitting the Data
    
    X = dataframe.drop(class_var, axis = 1)
    X = X.to_dict(orient = 'records') 
    Y = dataframe[class_var]  
    
    # Setting up Prediction and Cost Vectors
    
    prediction_vector = [0]*len(Y)
    cost_vector = [0]*len(Y) 
    correct_count = 0
    
    for i in range(len(Y)):
        evidence = X[i]
        
        if print_evi:
            print(evidence)
        
        # Checking for missing values
        
        missing = []
        
        for sensor in evidence:
            if pd.isnull(evidence[sensor]):
                missing.append(sensor)
        
        # Prediction of probability
        
        if model_type == 'naive':
        
            empty_prob = model.predict_proba(class_var, evidence = evidence, missing = missing)
            empty_prob = np.squeeze(empty_prob)[()] # Extracting probability from array
            
        elif model_type == 'hidden':
            
            empty_prob = model.forward(missing = missing, class_var = class_var, **evidence)
            
        elif model_type == 'hidden_extra':
            if debug:
                empty_prob = model.forward_extra(missing = missing, class_var = class_var, debug = True, sleep = False, **evidence)
            else:
                empty_prob = model.forward_extra(missing = missing, class_var = class_var, **evidence)
            
        
        else:
            
            raise Exception('Model Not Supported')
        
        # Cost Calculation    
        
        if empty_prob > p: # Turn off light
            cost = 0.04*Y.iloc[i]
            cost_vector[i] = cost
            if cost == 0:
                correct_count += 1
        else: # Turn on light
            cost = 0.01
            cost_vector[i]= cost
        
        if print_evi:
            print('Actual: ', Y.iloc[i])
        
        prediction_vector[i] = empty_prob
            
    return cost_vector, prediction_vector, correct_count # Will extend to probability soon

def learn_naive_bayes_structure(dataframe, class_var):
    '''
    Arguments:
        dataframe:   A pandas dataframe
        class_var:   Variable identifier to be classified
    Returns:
        A Graph object with the structure of the Naïve Bayes classifier for the attributes in dataframe
    '''
    graph = dict()

    for col in dataframe.columns:
        if col != class_var: # Features are leaf nodes
            graph[col] = []
            
        else:
            features = list(dataframe.drop(class_var, axis=1).columns)
            graph[col] = features # Class variable is connected to each feature variable
            
    G = Graph(graph)
            
    return G

def threshold_test(pred_vec, actual_vec, p):
    """
    Parameters
    ----------
    pred_vec : List
        DESCRIPTION.
    actual_vec : numpy array
        DESCRIPTION.
    p : float
        DESCRIPTION.

    Returns
    -------
    total_cost : TYPE
        DESCRIPTION.

    """
    
    assert len(pred_vec) == len(actual_vec)
    
    cost_vec = [0]*len(pred_vec)
    
    for i in range(len(pred_vec)):
        if pred_vec[i] > p: # Turn off light
            cost = 0.04*actual_vec[i]
            cost_vec[i] = cost
        else: # Turn on light
            cost = 0.01
            cost_vec[i]= cost
        
    return cost_vec



