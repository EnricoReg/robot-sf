import numpy as np
import math
import json
from .map import *
from .range_sensor import *




class DifferentialDrive():
    """This class create an object representing a differential drive robot"""


    def __init__(self, wheel_rad = 0.05, interaxis = 0.3, rob_collision_radius = 0.7):
        #wheel_rad is the radius of the wheels
        #interaxis is the distance between the two wheels
        self.wheel_radius = wheel_rad
        self.interaxis_lenght = interaxis
        self.last_speed = dict()
        self.last_speed['linear'] = dict()
        self.last_speed['angular'] = dict()
        self.last_speed['linear']['x'] = 0
        self.last_speed['linear']['y'] = 0
        self.last_speed['linear']['z'] = 0
        self.last_speed['angular']['x'] = 0
        self.last_speed['angular']['y'] = 0
        self.last_speed['angular']['z'] = 0

        self.current_speed = self.last_speed
        self.last_pose = dict()
        self.last_pose['position'] = dict()
        self.last_pose['orientation'] = dict()
        self.last_pose['position']['x'] = 0
        self.last_pose['position']['y'] = 0
        self.last_pose['position']['z'] = 0
        self.last_pose['orientation']['x'] = 0
        self.last_pose['orientation']['y'] = 0
        self.last_pose['orientation']['z'] = 0

        self.current_pose = self.last_pose

        self.last_wheels_speed = dict()
        self.last_wheels_speed['left'] = 0
        self.last_wheels_speed['right'] = 0

        self.wheels_speed = self.last_wheels_speed

        #Set wheels speed limit
        self.max_linear_speed = float('inf')
        self.max_angular_speed = float('inf')


        #Initialize Laser Scanner
        self.scanner = None
        self.scan_list_history = dict()

        #Initialize map
        self.map = None

        #Initialize controller
        self.controller = None

        self.initializePIDcontroller()
        
        self.rob_collision_radius = rob_collision_radius




    def resetOdometry(self):
        for key in self.last_pose.keys():
            self.last_pose[key] = 0
            self.current_pose[key] = 0
                
        self.clearOldvar()


    def setMaxSpeed(self, linear, angular):
        self.max_linear_speed = linear
        self.max_angular_speed = angular

    def getWheelsSpeed(self):
        return self.wheels_speed

    def printMaxSpeed(self):
        print('Robot''s max linear speed is: ' + str(self.max_linear_speed))
        print('Robot''s max angular speed is: ' + str(self.max_angular_speed))
            
    def getMaxSpeed(self):
        return self.max_linear_speed, self.max_angular_speed

    def stop(self):
        for key in self.current_speed.keys():
            for sub_key in self.current_speed[key].keys():
                self.current_speed[key][sub_key] = 0

    def setPosition(self, x, y, orient):
        self.current_pose['position']['x'] = x
        self.current_pose['position']['y'] = y
        self.current_pose['orientation']['z'] = orient
        self.last_pose = self.current_pose
        self.clearOldvar()
        self.updateRobotOccupancy()
        


    def updateRobotSpeed(self, dot_x, dot_orient):
        if dot_x > self.max_linear_speed or dot_x < -self.max_linear_speed:
            dot_x = np.sign(dot_x)*self.max_linear_speed

        if dot_orient > self.max_angular_speed or dot_orient < -self.max_angular_speed:
            dot_orient = np.sign(dot_orient)*self.max_angular_speed

        #Compute Kinematics
        self.wheels_speed['left'] = (dot_x - self.interaxis_lenght*dot_orient/2)/self.wheel_radius
        self.last_wheels_speed['right'] = (dot_x + self.interaxis_lenght*dot_orient/2)/self.wheel_radius

        #Update current speed
        self.current_speed['linear']['x'] = dot_x
        self.current_speed['angular']['z'] = dot_orient

    def getOdom(self):
        return self.current_pose

    def computeOdometry(self,Ts):
        new_orient = self.last_pose['orientation']['z'] + self.wheel_radius/self.interaxis_lenght*(-(self.last_wheels_speed['left'] \
            + self.wheels_speed['left'])/2 + (self.last_wheels_speed['right'] \
            + self.wheels_speed['right'])/2)*Ts

        new_x_local = self.wheel_radius/2*((self.last_wheels_speed['left'] + self.wheels_speed['left'])/2 \
            + (self.last_wheels_speed['right'] + self.wheels_speed['right'])/2)*Ts


        self.current_pose['position']['x'] += new_x_local*math.cos((new_orient + self.last_pose['orientation']['z'])/2)
        self.current_pose['position']['y'] += new_x_local*math.sin((new_orient + self.last_pose['orientation']['z'])/2)
        self.current_pose['orientation']['z'] = new_orient

        #Update old values
        self.last_pose = self.current_pose
        self.last_wheels_speed = self.wheels_speed
        
        # update robot occupancy map
        self.updateRobotOccupancy()


    def clearOldvar(self):
        for key in self.last_wheels_speed.keys():  #_last_wheels_speed previously..
            self.last_wheels_speed[key] = 0
            self.wheels_speed[key] = 0

    def initializeMap(self, ext_map = None):
        #Initialize default empty map
        if ext_map is None:
            self.map = BinaryOccupancyGrid()
        else:
            self.map = ext_map
            
    def getCurrentPosition(self):
        return [self.current_pose['position']['x'] , self.current_pose['position']['y'], self.current_pose['orientation']['z'] ]
            
    def getMap(self):
        return self.map

    def loadMap(self,filename):
        self.map.loadMap(filename)
        return True

    def showMap(self):
        self.map.show()

    def initializeScanner(self, **kwargs):
        #Initialize default scanner
        self.scanner = LiDARscanner(**kwargs)

    def getScan(self, scan_noise = None):
        scan = self.scanner.getScan(self.current_pose['position']['x'],\
            self.current_pose['position']['y'],self.current_pose['orientation']['z'],self.map, scan_noise = scan_noise)
            
        self.__udpateScanList(scan)
        return scan

    def initializePIDcontroller(self):
        pid = dict()
        pid['kp'] = 1
        pid['kd'] = 1
        pid['ki'] = 1

        direction = dict()
        direction['x'] = pid
        direction['y'] = pid
        direction['z'] = pid

        self.controller = dict()
        self.controller['linear'] = direction
        self.controller['angular'] = direction

    def moveToGoal(self,goal):
        '''This method aims at moving the robot towards a target point'''


    def checkOutOfBounds(self, margin = 0):
        """checks if robot went out of bounds """
        if not self.map.check_if_valid_world_coordinates(self.getCurrentPosition()[:2], margin).any():
            return True
        else:
            return False
            

    def getPedsDistances(self):
        # returns vector with distances of all pedestrians from robot    
        idx_x, idx_y = np.where( self.map.OccupancyRaw == True)
        idxs_peds = np.concatenate( ( idx_x[:,np.newaxis] , idx_y[:,np.newaxis]  ) , axis = 1)
        
        world_coords_peds = self.map.convert_grid_to_world(idxs_peds)
        
        return np.sqrt(np.sum((world_coords_peds - self.getCurrentPosition()[:2])**2, axis = 1))
    
    
    def getPedsDistancesAndAngles(self):
        # returns vector with distances of all pedestrians from robot    
        idx_x, idx_y = np.where( self.map.OccupancyRaw == True)
        idxs_peds = np.concatenate( ( idx_x[:,np.newaxis] , idx_y[:,np.newaxis]  ) , axis = 1)
        
        
        
        world_coords_peds = self.map.convert_grid_to_world(idxs_peds)
        
        #Compute angles
        
        
        dst =  np.sqrt(np.sum((world_coords_peds - self.getCurrentPosition()[:2])**2, axis = 1))
        
        pos = self.getCurrentPosition()
        
        alphas = np.arctan2(world_coords_peds[:,1] - pos[1], world_coords_peds[:,0] - pos[0]) - pos[2]
        
        alphas = wrap2pi(alphas)
        return list(zip(alphas, dst))
    
    
    def chunk(self, n_sections):
        data = self.getPedsDistancesAndAngles()
        
        section = np.linspace(-np.pi,np.pi, n_sections+1)
        sector_id = np.linspace(0,n_sections-1,n_sections)
        distances = len(sector_id)*[self.scanner.range[1]]
        
        for i in range(len(data)):
            for j in range(len(sector_id)):
                if data[i][0] >= section[j] and data[i][0] <= section[j+1]:
                    if data[i][1] < distances[j]:
                        distances[j] = data[i][1]
                        break
         
        return distances

    def updateRobotOccupancy(self):
        # internal update only uses attribut collision distance
        self.robot_occupancy = self.getRobotOccupancy()
        
    def getRobotOccupancy(self,*coll_distance):
        
        try:
            coll_distance = coll_distance[0]
        except Exception:
            coll_distance = self.rob_collision_radius
        
        x0 = self.current_pose['position']['x']
        y0 = self.current_pose['position']['y']
        
        rob_matrix = np.zeros(self.map.Occupancy.shape,dtype = bool)
        idx = self.map.convert_world_to_grid_no_error(np.array([x0,y0])[np.newaxis,:])
        
        self.robot_idx = idx
        
        int_radius_step = round(self.map.map_resolution*coll_distance)
        return fill_surrounding(rob_matrix,int_radius_step,idx) #[0,0],idx[0,1]) 
        

    def check_obstacle_collision(self, *coll_distance):
        if self.map is not None:
            if coll_distance:
                try:
                    return np.logical_and(self.map.OccupancyFixed,self.getRobotOccupancy(coll_distance[0])).any() 
                except Exception:
                    pass
            return np.logical_and(self.map.OccupancyFixed,self.robot_occupancy).any()
        return False
        

    def check_pedestrians_collision(self, *coll_distance):
        if self.map is not None:
            if coll_distance:
                try:
                    return np.logical_and(self.map.Occupancy,self.getRobotOccupancy(coll_distance[0])).any()
                except Exception:
                    pass
            return np.logical_and(self.map.Occupancy,self.robot_occupancy).any()

        return False


    def check_collision(self, *coll_distance):
        # check for collision with map objects (when distance from robots is less than radius)
        if self.map is not None:
            if coll_distance is None:
                return self.check_pedestrians_collision() or self.check_obstacle_collision()
            else:
                return self.check_pedestrians_collision(coll_distance[0]) or self.check_obstacle_collision(coll_distance[0])
        return False
    

    def target_rel_position(self, target_coordinates):
        x0 = self.current_pose['position']['x']
        y0 = self.current_pose['position']['y']
        
        dist  = np.linalg.norm(target_coordinates - np.array([x0,y0]))
        
        #@xl_func("numpy_row v1, numpy_row v2: float")
        def py_ang(v1, v2):
            """ Returns the angle in radians between vectors 'v1' and 'v2'    """
            cosang = np.dot(v1, v2)
            sinang = np.linalg.norm(np.cross(v1, v2))
            return np.arctan2(sinang, cosang)
        
        angle = py_ang(target_coordinates , np.array([x0,y0]))
        
        return dist, angle

    def check_target_reach(self, target_coordinates ,tolerance = 0.2):
        # check if target is reached
        return self.target_rel_position(target_coordinates)[0] <= tolerance


    def __udpateScanList(self,scan):
        if not self.scan_list_history.keys():
            self.scan_list_history['scan_1'] = scan
        else:
            key_list = list(self.scan_list_history.keys())  #edited: in Python3 'dict_keys' object are not subscriptable
            name = 'scan_'+ str(int(key_list[-1].replace('scan_','')) + 1)
            self.scan_list_history[name] = scan




