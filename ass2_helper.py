# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 01:59:36 2022

@author: Jake
"""
import pandas as pd
import numpy as np


def bin_data(data):
    if pd.isnull(data):
        return data
    elif data < 1:
        return '0'
    elif data < 5:
        return '1-4'
    else:
        return '5+'
    
def return_rooms():
    return ['r1','r2','r3','r4','r5','r6','r7','r8','r9','r10','c1','c2','c3','outside']

def return_room_index(rooms):
    room_index = {}

    for i,room in enumerate(rooms):
        room_index[room] = i
        
    return room_index

def return_room_sensors():
    room_sensors = {}
    room_sensors['r1'] = ['Motion_Sensor1','Camera1','door_sensor1']
    room_sensors['r2'] = ['Motion_Sensor2', 'door_sensor1', 'door_sensor10']
    room_sensors['r3'] = ['Motion_Sensor3', 'door_sensor2']
    room_sensors['r4'] = ['Motion_Sensor4', 'Camera2', 'door_sensor3', 'door_sensor4']
    room_sensors['r5'] = ['Motion_Sensor5', 'door_sensor4', 'door_sensor5']
    room_sensors['r6'] = ['Motion_Sensor6', 'door_sensor5']
    room_sensors['r7'] = ['Motion_Sensor7', 'door_sensor8', 'door_sensor9']
    room_sensors['r8'] = ['Motion_Sensor8', 'Camera3', 'door_sensor8']
    room_sensors['r9'] = ['Motion_Sensor9', 'door_sensor6']
    room_sensors['r10'] = ['Motion_Sensor10', 'door_sensor7', 'door_sensor11']
    
    return room_sensors

def return_room_lights():
    room_lights = {}
    room_lights['r1'] = 'lights1'
    room_lights['r2'] = 'lights2'
    room_lights['r3'] = 'lights3'
    room_lights['r4'] = 'lights4'
    room_lights['r5'] = 'lights5'
    room_lights['r6'] = 'lights6'
    room_lights['r7'] = 'lights7'
    room_lights['r8'] = 'lights8'
    room_lights['r9'] = 'lights9'
    room_lights['r10'] = 'lights10'

    return room_lights

def return_start_states():
    start_states = {}
    start_states['r1'] = np.array([0.99,0.01,0,0,0,0])
    start_states['r2'] = np.array([0.98,0.01,0.01,0,0,0])
    start_states['r3'] = np.array([0,0,0,0,0,1])
    start_states['r4'] = np.array([0.9,0.02,0.02,0.02,0.02,0.02])
    start_states['r5'] = np.array([0.94,0.02,0.01,0.01,0.01,0.01])
    start_states['r6'] = np.array([0.96,0.02,0.01,0.01,0,0])
    start_states['r7'] = np.array([0.96,0.02,0.01,0.01,0,0])
    start_states['r8'] = np.array([0.99,0.01,0,0,0,0])
    start_states['r9'] = np.array([0.99,0.01,0,0,0,0])
    start_states['r10'] = np.array([0.95,0.01,0.01,0.01,0.01,0.01])

    return start_states