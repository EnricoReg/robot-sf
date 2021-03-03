import numpy as np
import math
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys
import random

#test


class LiDARscanner():
    """The following class is responsible of creating a LiDAR scanner object"""
    def __init__(self,scan_range = [0.1, 1],angle_opening = [-np.pi, np.pi], angle_increment = 0.0174532925199):
        '''range: list containing minimum and maximum range values for the scanner
           angle_opening: list containing minimum and maximum opening of the scan laser
           angle_increment: resolution of the laser
           '''

        
        if not isinstance(scan_range,list) and not isinstance(scan_range,np.ndarray):
            raise ValueError('Input argument for range must be a list or a numpy array')
        if isinstance(scan_range,np.ndarray):
            if not range.shape[0] == 2:
                raise ValueError('Input argument for range must be of size 2')
        
        if not isinstance(angle_opening,list) and not isinstance(angle_opening,np.ndarray):
            raise ValueError('Input argument for angle_opening must be a list or a numpy array')
        if isinstance(angle_opening,np.ndarray):
            if not angle_opening.shape[0] == 2:
                raise ValueError('Input argument for angle_opening must be of size 2')
        

        self.range = scan_range
        self.angle_opening = angle_opening
        self.angle_increment = angle_increment
        self.distance_noise = 0
        self.angle_noise = 0
        self.num_readings = 0
        self.scan_structure = dict()
        self.scan_structure['properties'] = dict()
        self.scan_structure['data'] = dict()

        self.emptyScanStructure()
        """
        self.scan_structure['properties']['range'] = self.range
        self.scan_structure['properties']['angle_opening'] = self.angle_opening
        self.scan_structure['properties']['angle_increment'] = self.angle_increment
        self.scan_structure['properties']['distance_noise'] = self.distance_noise

        # original angles calculation edited
        self.__original_angles = np.arange(self.angle_opening[0],self.angle_opening[1]+self.angle_increment,self.angle_increment).tolist()
        self.num_readings = len(self.__original_angles)
        """



    def getScan(self,x,y,orient,input_map , scan_noise = None):
        ''' This method takes in input the state of the robot
        and an input map (map object) and returns a data structure
        containing the sensor readings'''
        self.emptyScanStructure()

        start_pt = [x,y]
        self.scan_structure['data']['pose'] = [x,y,orient]

        #scan_length = len(self.scan_structure['data']['angles'])
        scan_length = self.num_readings

        if scan_noise is not None:
            lost_scans = np.where(np.random.random(scan_length) < scan_noise[0])[0]
            corrupt_scans = np.where(np.random.random(scan_length) < scan_noise[1])[0]
        else:
            lost_scans = []
            corrupt_scans = []


        for i in range(scan_length):
            #Transform orientation from laser frame to grid frame
            #and compute end-point
            grid_orient = orient + self.scan_structure['data']['angles'][i]
            #compute end-point
            end_pt = list()
            end_pt.append(start_pt[0] + self.range[1]*math.cos(grid_orient))
            end_pt.append(start_pt[1] + self.range[1]*math.sin(grid_orient))
            try:
                ray_index = input_map.raycast(np.array([start_pt]),np.array([end_pt]))
                flag,intercept_grid,intercept = input_map.doesRayCollide(ray_index)
                # added intercept grid for debugging purpose
            except Exception:
                flag = False
            
            if flag and not i in lost_scans:
                #Ray collided with obstacle! Compute distance
                #print('entering')
                self.scan_structure['data']['ranges'][i] = \
                    math.sqrt((intercept[0] - start_pt[0])**2 + (intercept[1] - start_pt[1])**2)
                
                self.scan_structure['data']['cartesian'][i,0] = intercept[0]
                self.scan_structure['data']['cartesian'][i,1] = intercept[1]
                
                self.scan_structure['data']['grid'][i] = intercept_grid
                # added intercept grid for debugging purpose

            if scan_noise is not None:

                if i in corrupt_scans and not i in lost_scans:
                    scanned_distance = random.random()*self.range[1]
                    self.scan_structure['data']['ranges'][i] = scanned_distance
                    
                    intercept = [0,0]                
                    intercept[0] = start_pt[0] + scanned_distance*math.cos(grid_orient)
                    intercept[1] = start_pt[1] + scanned_distance*math.sin(grid_orient)
                
                    self.scan_structure['data']['cartesian'][i,0] = intercept[0]
                    self.scan_structure['data']['cartesian'][i,1] = intercept[1]
                
                    self.scan_structure['data']['grid'][i] = input_map.convert_world_to_grid_no_error(np.array([intercept]))

        #self.applyAngleNoise()
        #self.applyDistanceNoise()

        return self.scan_structure

            

    def emptyScanStructure(self):
        '''This method empty the scan data structure and repopulate
        it, according to new properties if they have changed'''

        self.__original_angles = np.arange(self.angle_opening[0],self.angle_opening[1]+self.angle_increment,self.angle_increment).tolist()
        self.num_readings = len(self.__original_angles)


        self.scan_structure['properties']['range'] = self.range
        self.scan_structure['properties']['angle_opening'] = self.angle_opening
        self.scan_structure['properties']['angle_increment'] = self.angle_increment
        self.scan_structure['properties']['distance_noise'] = self.distance_noise
        self.scan_structure['data']['angles'] = self.__original_angles
        self.scan_structure['data']['ranges'] = [math.nan]*self.num_readings
        self.scan_structure['data']['intensities'] = [math.inf]*self.num_readings
        self.scan_structure['data']['pose'] = [0]*3
        self.scan_structure['data']['cartesian'] = np.zeros((self.num_readings,2), dtype = float)
        # added intercept grid for debugging purpose
        self.scan_structure['data']['grid'] = np.zeros((self.num_readings,2), dtype = int)

    def applyDistanceNoise(self):
        ranges = np.array(self.scan_structure['data']['ranges'],dtype=float)
        new_ranges = self.distance_noise*np.random.randn(ranges.shape[0]) + ranges
        self.scan_structure['data']['ranges'] = new_ranges.tolist()

    def applyAngleNoise(self):
        angles = np.array(self.scan_structure['data']['angles'],dtype=float)
        new_angles = self.angle_noise*np.random.randn(angles.shape[0]) + angles
        self.scan_structure['data']['angles'] = new_angles.tolist()

    def show(self, map_entry=None):
        '''This method plots the rays'''
        tmp = np.zeros((len(self.scan_structure['data']['angles']),2), dtype=float)
        ranges_filtered = self.removeInvalidData(self.scan_structure['data']['ranges'])
        tmp[:,0] = self.scan_structure['data']['angles']
        tmp[:,1] = ranges_filtered
        #print(tmp)
        #Remove all rows which contains nan values, i.e. the ray didn't encounter an obstacle
        tmp[~np.isnan(tmp).any(axis=1)]

        if tmp.size==0:
            #No rays to plot
            plt.plot(self.scan_structure['data']['pose'][0],self.scan_structure['data']['pose'][1])
            plt.text(self.scan_structure['data']['pose'][0],self.scan_structure['data']['pose'][1],'G')
            plt.show()
        else:
            #define rotation_matrix
            pts_x = np.multiply(tmp[:,1],np.cos(tmp[:,0]))
            pts_y = np.multiply(tmp[:,1],np.sin(tmp[:,0]))

            theta = self.scan_structure['data']['pose'][2]
            R = np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta),math.cos(theta)]])

            #Transform data from scanner frame to world frame
            tmp = np.array([pts_x, pts_y])
            #pts_transformed = np.dot(np.transpose(R),tmp)
            pts_transformed = np.dot(R,tmp)
            #print(tmp)

            pts_transformed[0,:] += self.scan_structure['data']['pose'][0]
            pts_transformed[1,:] += self.scan_structure['data']['pose'][1]

            if map_entry:
                plt.imshow(map_entry.Occupancy,cmap = 'Greys',\
                    extent=[-map_entry.grid_origin[0], -map_entry.grid_origin[0] + map_entry.map_length, \
                    -map_entry.grid_origin[1], -map_entry.grid_origin[1] + map_entry.map_height])

            plt.plot(self.scan_structure['data']['pose'][0],self.scan_structure['data']['pose'][1],'o',color='black')
            plt.text(self.scan_structure['data']['pose'][0],self.scan_structure['data']['pose'][1],'G')
            #plt.scatter(pts_transformed[0,:],pts_transformed[1,:],marker='.',color='blue')
            pts = np.array(self.scan_structure['data']['cartesian'])
            #....
            #....
            #....
            #print(pts)
            plt.scatter(pts[:,0], pts[:,1], marker = '.', color = 'blue')
            plt.axis('equal')
            plt.show()


    def save(self,filename):
        ''' This method will save the scan data structure into a json file
        specified by filename'''
        copied_scan = self.scan_structure
        copied_scan['info'] = dict()
        copied_scan['info']['author'] = 'Matteo Caruso'
        copied_scan['info']['email'] = 'matteo.caruso@phd.units.it'
        copied_scan['info']['created'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        #Add path for saving json file
        filename,extension = os.path.splitext(filename)
        dir,name = os.path.split(filename)

        if not dir:
            #relative path
            home = os.getcwd()
            platform = sys.platform
            platform = sys.platform

            if platform == 'win32':
                path = home + '\\data\\scans\\' + name +'.json'
            elif platform == 'linux':
                path = home + '/data/scans/' + name + '.json'
            else:
                raise NameError('Platform not supported!')
        else:
            #absolute path
            path = filename + '.json'


        with open(path,'w') as f:
            json.dump(self.scan_structure,f)
            print('Scan saved at:' + path)



    def removeInvalidData(self,range_list):
        '''This method removes invalid data from the ranges array,
        i.e. data which is below of range lower limit'''
        tmp = np.array(range_list, dtype=float)
        with np.errstate(invalid='ignore'):
            tmp[tmp<self.range[0]] = np.nan
        return tmp


