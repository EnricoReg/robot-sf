# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 14:45:55 2020

@author: enric
"""


# required if current directory is not found

import sys 
import os

csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)


#%%

from pysocialforce.scene import PedState

import numpy as np

#%%

class PedState(PedState):
        
    @property
    def state(self):
        return self._state
        
    @state.setter
    def state(self, state):
        tau = self.default_tau * np.ones(state.shape[0])
        if state.shape[1] < 7:
            self._state = np.concatenate((state, np.expand_dims(tau, -1)), axis=-1)
        else:
            self._state = state
        if self.initial_speeds is None:
            self.initial_speeds = self.speeds()
        # new IF added: change of state requires an update in the initial speeds
        elif self.initial_speeds.shape[0] < self.state.shape[0]:
            index_new_states = self.initial_speeds.shape[0]
            self.initial_speeds = np.concatenate((self.initial_speeds, self.speeds()[index_new_states:]),axis = 0)
        self.max_speeds = self.max_speed_multiplier * self.initial_speeds
        self.ped_states.append(self._state.copy())
        
        
    def get_states(self):
        max_rows = max([i.shape[0] for i in self.ped_states])
        for i,state in enumerate(self.ped_states):
            if state.shape[0] < max_rows:
                self.ped_states[i] = np.append(state, np.zeros(( max_rows-state.shape[0] , state.shape[1])), axis = 0)
        stacked_history = np.stack(self.ped_states)
        return stacked_history, self.group_states
    
    
    def getNumGroupedPedestrians(self): #(TESTED!)
        num = 0
        for group in self.groups:
            if group:
                num += len(group)
        
        return num
    
    
    def computeCentroidGroup(self, group_index):
        """ This method compute the centroid position of the group identified 
        by the group_index """
        
        #First check if the group index input is valid
        if group_index is None:
            return False
        
        valid_group_indices = []
        
        for n, sublist in enumerate(self.groups):
            if sublist:
                valid_group_indices.append(n)
                
        if not group_index in valid_group_indices:
            return False
        
        ped_list = self.groups[group_index]
        
        x = 0
        y = 0
        
        for ped_idx in ped_list:
            x += self.state[ped_idx,0]
            y += self.state[ped_idx,1]
            
        x /= len(ped_list)
        y /= len(ped_list)
        
        return [x,y]
        
