# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 13:44:17 2022

@author: Jake
"""
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
from ass2_modelling_sup import learn_outcome_space, assess_cost, learn_naive_bayes_structure, threshold_test
from ass2_helper import return_rooms, return_room_index, return_room_sensors, return_room_lights

# Data

day_1 = pd.read_csv('data1.csv')
day_2 = pd.read_csv('data2.csv')
combined = pd.concat([day_1,day_2])

### Cleaning the combined dataset

combined = combined.iloc[:,1:].reset_index(drop = True) # Removing column of indexes
combined_time = combined.iloc[:, 27]
combined_sensors = combined.iloc[:, :27]
combined_actual = combined.iloc[:,28:]

### Observations in Bins


#bins = [0,0.5,2.5,4.5,np.inf]
#labels = ['0','1-2','3-4','5+']


bins = [0,0.5,1.5,2.5,3.5,4.5,np.inf]
labels = ['0','1','2','3','4','5+']

#bins = [0,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,np.inf]
#labels = ['0','1','2','3','4','5','6','7','8+']


#bins = [0,0.5,4,np.inf]
#labels = ['0','1-4','5+']

for col in combined_actual.columns:
    combined_actual[col] = pd.cut(combined_actual[col], bins = bins, labels = labels, include_lowest = True)

for col in combined_sensors.columns:
    if not ('Motion' in col) and not ('robot' in col):
        combined_sensors[col] =pd.cut(combined_sensors[col], bins = bins, labels = labels, include_lowest = True)

# Converting Robot Sensor Data into useable data

room_index = {}

for i,room in enumerate(combined_actual.columns):
    room_index[room] = i

# Robot 1

robot_info = []

for obs in combined['robot1']:
    info = [None]*len(combined_actual.columns)
    left = obs.find("'")
    right = obs.rfind("'")
    room = obs[left+1:right]
    num = int(obs[-2])
    info[room_index[room]] = num
    robot_info.append(info)
    
col_names_rob1 = [col+'_rob1' for col in combined_actual.columns]
    
rob1 = pd.DataFrame(robot_info, columns = col_names_rob1)

for col in rob1.columns:
    rob1[col] =pd.cut(rob1[col], bins = bins, labels = labels, include_lowest = True)

# Robot 2

robot_info = []

for obs in combined['robot2']:
    info = [None]*len(combined_actual.columns)
    left = obs.find("'")
    right = obs.rfind("'")
    room = obs[left+1:right]
    num = int(obs[-2])
    info[room_index[room]] = num
    robot_info.append(info)
    
col_names_rob2 = [col+'_rob2' for col in combined_actual.columns]
    
rob2 = pd.DataFrame(robot_info, columns = col_names_rob2)

for col in rob2.columns:
    rob2[col] =pd.cut(rob2[col], bins = bins, labels = labels, include_lowest = True)

train_sensors = combined_sensors.iloc[:len(day_1)]
train_actual = combined_actual.iloc[:len(day_1)]
train_robot1 = rob1.iloc[:len(day_1)]
train_robot2 = rob2.iloc[:len(day_1)]
test_sensors = combined_sensors.iloc[len(day_1):]
test_actual = combined.iloc[len(day_1):,28:] # Untransformed as we want to calculate exact cost
test_robot1 = rob1.iloc[len(day_1):]
test_robot2 = rob2.iloc[len(day_1):]

#### Benchmarks - based on day 2

# All light on benchmark

cost = 0.01
rooms = 10
total_cost = cost*rooms*len(test_actual)
print('All lights on cost:', total_cost, '\n')

# All light off benchmark

cost_pp = 0.04
rooms = 10
relevent_rooms = test_actual.iloc[:, :10]
total_cost = np.sum(relevent_rooms.values*cost_pp)

print('All lights off cost:', total_cost, '\n')

# Best possible case - off every time there is 0 people in the room, on otherwise

relevent_rooms = test_actual.iloc[:, :10]
relevent_rooms = relevent_rooms.values.flatten()
cost_vector = np.where(relevent_rooms == 0, 0, 0.01)
total_cost = np.sum(cost_vector)

print('Best Case:', total_cost, '\n')

### Room Data

room_sensors = return_room_sensors()

### Naive Bayes Models

# Training for all rooms

room_models = {}

for room in room_sensors:
    
    # Subsetting the data
    sensors = room_sensors[room]
    
    sensors_door =  []
    for sensor in sensors:
        if 'door_sensor' in sensor:
            sensors_door.append(sensor)
    
    data = pd.concat([train_sensors[sensors], train_actual[room], train_robot1[room+'_rob1'], train_robot2[room+'_rob2']], axis = 1)
    whole_data = pd.concat([combined_sensors[sensors], combined_actual[room], rob1[room+'_rob1'], rob2[room+'_rob2']], axis = 1) # For learning outcome space
    
    # Training model
    outcomeSpace = learn_outcome_space(sensors = sensors, labels = labels, room = room) 
    graph = learn_naive_bayes_structure(data, room)
    
    model = NaiveBayes(graph, outcomeSpace = outcomeSpace)
    model.learnParameters(whole_data, alpha = 1)
    
    # Saving Model
    room_models[room] = model
    
# Testing for all rooms

total_cost = 0
pred_probs = []

for room in room_sensors:
    
    print(room)
    # Subsetting the data
    sensors = room_sensors[room]
    data = pd.concat([test_sensors[sensors], test_actual[room], test_robot1[room+'_rob1'], test_robot2[room+'_rob2']], axis = 1)
    
    # Testing the model
    cost_vec, pred_vec = assess_cost(model = room_models[room], dataframe = data, class_var = room, model_type = 'naive', p=0.75)
    print(room, ' Successfully tested')
    total_cost += np.sum(cost_vec)
    pred_probs.extend(pred_vec)
    
print('Total Cost of Naive Bayes:', total_cost)


room_list = return_rooms()
room_list = room_list[:10]
room_data = test_actual[room_list]
room_data = room_data.values.flatten('F')
cost_vec = threshold_test(pred_probs, room_data, p = 0.75)




### Hidden Markov - Current 

room_models = {}

for room in room_sensors:

    # Subsetting the data, adding column for room_next
    sensors = room_sensors[room]
    variable_remap = {room+'_next':room}
    data = pd.concat([train_sensors[sensors], train_actual[room], train_actual[room].shift(-1).rename(room+'_next'), train_robot1[room+'_rob1'], train_robot2[room+'_rob2']], axis = 1)[:-1]
    whole_data = pd.concat([combined_sensors[sensors], combined_actual[room], combined_actual[room].shift(-1).rename(room+'_next'), rob1[room+'_rob1'], rob2[room+'_rob2']], axis = 1)[:-1] # For learning outcome space

    # Learning Factors through NaiveBayes class 
    outcomeSpace = learn_outcome_space(sensors = sensors, labels=labels, room=room) 
    graph = learn_naive_bayes_structure(data, room)
    model = NaiveBayes(graph, outcomeSpace = outcomeSpace)
    model.learnParameters(data, alpha = 1)
    
    # Setting up factor lists for markov network
    factor_list = model.factors
    transition = factor_list[room+'_next']
    
    # Remove room, room_next from emission dict
    emissions = dict(factor_list)
    del emissions[room]
    del emissions[room+'_next']
    
    # Set the model with factors - I think I want custom starting states rather than the learnt ones, update when this is working.
    model = HiddenMarkovModel(start_state = factor_list[room], transition = transition, emission= emissions, variable_remap = variable_remap, outcomeSpace = outcomeSpace)
    room_models[room] = model



# Testing

total_cost = 0
pred_probs = []

for room in room_sensors:
    
    print(room)
    # Subsetting the data
    sensors = room_sensors[room]
    data = pd.concat([test_sensors[sensors], test_actual[room], test_robot1[room+'_rob1'], test_robot2[room+'_rob2']], axis = 1)
    
    # Testing the model
    cost_vec, pred_vec = assess_cost(model = room_models[room], dataframe = data, class_var = room, model_type = 'hidden', p=0.75, print_evi = True)
    print(room, ' Successfully tested')
    total_cost += np.sum(cost_vec)
    pred_probs.extend(pred_vec)

### Hidden Markov with Door dependency - Currently scuffed 

# Training for all rooms

# Adding a door_sensor_prev that will be saved between iterations as evidence for the next iteration.

room_models = {}

for room in room_sensors:
    
    # Subsetting the data, adding column for room_next
    sensors = room_sensors[room]
    door_sensors = []
    for sensor in sensors:
        if 'door_sensor' in sensor:
            door_sensors.append(sensor)
            
    variable_remap = {room:room+'_prev'}
    data = pd.concat([train_sensors[sensors], train_actual[room], train_actual[room].shift(1).rename(room+'_prev'), train_robot1[room+'_rob1'], train_robot2[room+'_rob2']], axis = 1)[1:]
    whole_data = pd.concat([combined_sensors[sensors], combined_actual[room], combined_actual[room].shift(1).rename(room+'_prev'), rob1[room+'_rob1'], rob2[room+'_rob2']], axis = 1)[1:] # For learning outcome space
    # Learning Factors through NaiveBayes class 
    outcomeSpace = learn_outcome_space(sensors=sensors, labels=labels, room=room, next_ = False)
    graph = learn_naive_bayes_structure(data.drop(door_sensors+[room+'_prev'], axis = 1), room)
    graph.add_edge(room+'_prev', room)
    for sensor in door_sensors:
        graph.add_edge(room+'_prev', sensor)
        graph.add_edge(sensor,room)
    model = BayesNet(graph, outcomeSpace = outcomeSpace)
    model.learnParameters(data, alpha = 0.1)

    # Setting up factor lists for markov network
    factor_list = model.factors    
    transition = factor_list[room]
    
    # Remove room, room_next from emission dict
    emissions = dict(factor_list)
    transition_door = []
    del emissions[room]
    del emissions[room+'_prev']
    for sensor in sensors:
        if 'door_sensor' in sensor:
            del emissions[sensor]
            transition_door.append(factor_list[sensor])
    
    # Set the model with factors - I think I want custom starting states rather than the learnt ones, update when this is working.
    # TODO - Customise starting factors based on the inital spread of people in the office
    model = HiddenMarkovModel(start_state = factor_list[room+'_prev'], transition = transition, emission= emissions, variable_remap = variable_remap, outcomeSpace = outcomeSpace, transition_extra = transition_door)
    room_models[room] = model  
    
total_cost = 0
cost_vecs = []

for room in room_sensors:
    
    print(room)
    # Subsetting the data
    sensors = room_sensors[room]
    data = pd.concat([test_sensors[sensors], test_actual[room], test_robot1[room+'_rob1'], test_robot2[room+'_rob2']], axis = 1)[:-1]
    
    # Testing the model
    cost_vec, pred_vec = assess_cost(model = room_models[room], dataframe = data, class_var = room, model_type = 'hidden_extra', p=0.8, print_evi = True, debug = False)
    print(room, ' Successfully tested')
    total_cost += np.sum(cost_vec)    
    cost_vecs.append(cost_vec)
    