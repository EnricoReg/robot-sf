# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:32:16 2020

@author: Matteo Caruso
"""

import copy
import numpy as np
import math
import random
import matplotlib.pyplot as plt



class Polygon:
    def __init__(self,n_vert = 5, irregularity=0,spikeness=0,normalized_in = True, radius = 1):
        self.n_vertex = n_vert
        self.normalized = normalized_in
        self.num_edges = n_vert
        self.irregularity = np.clip(irregularity, 0,1 ) * 2*math.pi / self.n_vertex
        self.spikeness = np.clip( spikeness, 0,1 ) * radius
        self.vertex = None
        self.edges = None
        self.centre = np.zeros((1,2))
        
        self.radius = radius
        self.initialized = False
        
        self.initialize()
        
        if normalized_in:
            self.normalize()
        
        
        
    def initialize(self):
        if self.initialized:
            return False
        
        angle_steps = []
        lower = (2*math.pi/self.n_vertex) - self.irregularity
        upper = (2*math.pi/self.n_vertex) - self.irregularity
        
        summ = 0
        for i in range(self.n_vertex):
            tmp = random.uniform(lower,upper)
            angle_steps.append(tmp)
            summ += tmp
        
        k = summ/(2*math.pi)
        
        for i in range(self.n_vertex):
            angle_steps[i] /= k;
        
        points = []
        angle = random.uniform(0, 2*math.pi)
        for i in range(self.n_vertex) :
            r_i = np.clip( random.gauss(self.radius, self.spikeness), 0, self.radius )
            x = r_i*math.cos(angle)
            y = r_i*math.sin(angle)
            points.append( (x,y) )    
            angle = angle + angle_steps[i]
            
        self.vertex = np.array(points);
        self.vertex = np.concatenate((self.vertex, np.reshape(self.vertex[0,:],(1,2))), axis = 0)
        
        self.computeEdges()
        self.computeCentroid()
        self.initialized = True
        
        
    
    def stretch(self,x,y):
        for i in range(self.vertex.shape[0]):
            if self.vertex[i,0] > self.centre[0,0]:
                self.vertex[i,0] += x/2
            elif self.vertex[i,0] < self.centre[0,0]:
                self.vertex[i,0] -= x/2
            
            if self.vertex[i,1] > self.centre[0,1]:
                self.vertex[i,1] += y/2
            elif self.vertex[i,1] < self.centre[0,1]:
                self.vertex[i,1] -= y/2
        
        self.computeEdges()
        self.computeCentroid()
                
        
        
    def scale(self,scale_factor, axis = None):
        if not axis:
            self.vertex[:,0] = scale_factor*(self.vertex[:,0] - self.centre[0,0]) + self.centre[0,0]
            self.vertex[:,1] = scale_factor*(self.vertex[:,1] - self.centre[0,1]) + self.centre[0,1]
            
        else:
            if axis == 0 or axis == 1:
                self.vertex[:,axis] = scale_factor*(self.vertex[:,axis] - self.centre[0,axis]) + self.centre[0,axis]
            else:
                return False
            
        self.computeEdges()
        self.computeCentroid()
        
        
    def move(self,x,y):
        self.vertex[:,0] += x
        self.vertex[:,1] += y
        self.computeEdges()
        self.computeCentroid()
        
    def rotate(self,angle):
        for i in range(self.vertex.shape[0]):
            x_curr = self.vertex[i,0]
            ycurr = self.vertex[i,1]
            self.vertex[i,0] = (x_curr - self.centre[0,0])*np.cos(angle) - (ycurr - self.centre[0,1])*np.sin(angle) + self.centre[0,0]
            self.vertex[i,1] = (x_curr - self.centre[0,0])*np.sin(angle) + (ycurr - self.centre[0,1])*np.cos(angle) + self.centre[0,1]
        
        self.computeEdges()
        self.computeCentroid()
           
        
        
    def show(self,unit_circle = False, axis_lim = True):
        plt.plot(self.vertex[:,0],self.vertex[:,1],'-ob')
        plt.plot(self.centre[0,0],self.centre[0,1],'ok')
        
        if axis_lim:
            plt.xlim(-1,1)
            plt.ylim(-1,1)
        
        if unit_circle:
            angle = np.arange(-np.pi,np.pi,2*np.pi/360)
            x = self.radius*np.cos(angle)
            y = self.radius*np.sin(angle)
            plt.plot(x,y,'-r')
        
        
    def normalize(self,xlimit_in = None, ylimit_in = None):
        if not xlimit_in and not ylimit_in:
            xlim = [0,1]
            ylim = [0,1]
        elif xlimit_in is not None and not ylimit_in:
            xlim = xlimit_in
            ylim = [0,1]
        elif not xlimit_in and ylimit_in is not None:
            ylim = ylimit_in
            xlim = [0,1]
        else:
            xlim = xlimit_in
            ylim = ylimit_in
        
        xmax = self.vertex[:,0].max()
        xmin = self.vertex[:,0].min()
        ymax = self.vertex[:,1].max()
        ymin = self.vertex[:,1].min()
        
        for i in range(self.vertex.shape[0]):
            self.vertex[i,0] = (xlim[1] - xlim[0])*(self.vertex[i,0] - xmin)/(xmax - xmin) + xlim[0]
            self.vertex[i,1] = (ylim[1] - ylim[0])*(self.vertex[i,1] - ymin)/(ymax - ymin) + ylim[0]
            
        self.computeEdges()
        self.computeCentroid()
        
    def computeCentroid(self):
        cx = 0
        cy = 0
        A = 0
        for i in range(self.vertex.shape[0]-1):
            cx += (self.vertex[i,0] + self.vertex[i+1,0])*(self.vertex[i,0]*self.vertex[i+1,1] - self.vertex[i+1,0]*self.vertex[i,1])
            cy += (self.vertex[i,1] + self.vertex[i+1,1])*(self.vertex[i,0]*self.vertex[i+1,1] - self.vertex[i+1,0]*self.vertex[i,1])
            A += (self.vertex[i,0]*self.vertex[i+1,1] - self.vertex[i+1,0]*self.vertex[i,1])
        A /= 2
        cx /= 6*A
        cy /= 6*A
        
        self.centre[0,0] = cx
        self.centre[0,1] = cy
            
        
        
    def computeEdges(self):
        #Store the edge in the format [xmin, xmax,ymin,ymax]
        tmp = []
        for i in range(self.vertex.shape[0]-1):
            tmp.append([self.vertex[i,0], self.vertex[i+1,0], self.vertex[i,1],self.vertex[i+1,1]])
        self.edges = tmp
        
            
        
        
        
    def constrainVertex(self,x_min = -np.inf, x_max= np.inf, y_min = -np.inf, y_max = np.inf):
        self.vertex[:,0][self.vertex[:,0] <= x_min] = x_min
        self.vertex[:,0][self.vertex[:,0] >= x_max] = x_max
        self.vertex[:,1][self.vertex[:,1] <= y_min] = y_min
        self.vertex[:,1][self.vertex[:,1] >= y_max] = y_max
        
        self.computeEdges()
        self.computeCentroid()
        
    def centalize(self):
        self.move(-self.centre[0,0],-self.centre[0,1])
        
        
    def copy(self):
        return copy.deepcopy(self)
    
    def save(self):
        print('To DO!')
        
    def load(self,list_of_vertex):
        #list of vertex must be a list of lists
        self.vertex = np.array(list_of_vertex)
        self.n_vertex = self.vertex.shape[0]
        self.normalized_in = False
        self.spikeness  = None
        self.irregularity= None
        self.radius = None
        
        #Compute edges and polygon centroid
        self.computeEdges()
        self.computeCentroid()
        
    
        
        

            

            
