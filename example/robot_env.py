#%% basics
import os
import numpy as np
import numpy.matlib

# matplotlib
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt

# gym dependencies 
import gym
from gym import error, spaces, utils
from gym.utils import seeding

#rendering
from PIL import Image

#%%
# imports from robot_sf package
from robot_sf.map import BinaryOccupancyGrid , fill_surrounding
from robot_sf.robot import DifferentialDrive
from robot_sf.range_sensor import LiDARscanner

from robot_sf.utils.utilities import clear_pycache

from robot_sf.extenders_py_sf.extender_sim import ExtdSimulator


#%%

def initialize_robot(visualization_angle_portion, lidar_range, lidar_n_rays, init_state, robot_collision_radius):
    
    # odd number & lidar_n_rays % 3 (or 5) = 0!! (3 or 5 used as stride in conv_net)
    # allowed lidar_n_rays = 105, 135, 165, 195,...

    sim_env = ExtdSimulator()
    #initialize map
    map_margin_coeff = 1 #.5
    
    robotMap = BinaryOccupancyGrid(map_height = map_margin_coeff*2*sim_env.box_size, map_length = map_margin_coeff*2*sim_env.box_size, peds_sim_env = sim_env)
    robotMap.center_map_frame()
    
    robotMap.update_from_peds_sim(fixedObjectsMap = True)
    robotMap.update_from_peds_sim()
    
    #initialize robot with map
    robot = DifferentialDrive(rob_collision_radius = robot_collision_radius)
    robot.initializeMap(robotMap)
    robot.setPosition(init_state[0], init_state[1], init_state[2])
    #robot.setPosition(0, -10, 0)
    
        
    lidar_range = [0,lidar_range] 
    lidar_angle_opening = (visualization_angle_portion*np.array([-np.pi, np.pi])).tolist()
    lidar_angle_increment = np.diff(lidar_angle_opening) / (lidar_n_rays-1)
    
    robot.initializeScanner(scan_range = lidar_range,angle_opening = lidar_angle_opening, angle_increment =lidar_angle_increment)
    
    #robot.map.peds_sim_env.addRobot(action_radius = 0.7, coordinates = init_state)

    return robot



#%%

class RobotEnv(gym.Env):
    #metadata = {'render.modes': ['human']}

    #####################################################################################################
    def __init__(self, lidar_n_rays = 165, \
                 collision_distance = 0.7, visualization_angle_portion = 0.5, lidar_range = 10,\
                 v_linear_max = 1.5 , v_angular_max = 2 , rewards = [1,100,20], max_v_x_delta = .5, \
                 initial_margin = .3,    max_v_rot_delta = .5, dt = None, normalize_obs_state = True, \
                     sim_length = 120, difficulty = 0, scan_noise = [0.005,0.002]):
        
        self.difficulty = difficulty
        # not implemented yet
        
        self.scan_noise = scan_noise  #percentages (1 = 100%) [scan_noise[0]: lost samples, scan_noise[1]: added samples]
        self.data_render = []
        
        self.robot = []
        self.lidar_n_rays = lidar_n_rays
        self.collision_distance = collision_distance
        self.target_coordinates = []
        self.visualization_angle_portion = visualization_angle_portion
        self.lidar_range = lidar_range
        
        self.env_type = 'RobotEnv'
        
        self.rewards = rewards
        
        self.normalize_obs_state = normalize_obs_state

        self.max_v_x_delta = max_v_x_delta
        self.max_v_rot_delta = max_v_rot_delta
        
        self.linear_max =  v_linear_max
        self.angular_max = v_angular_max
        
        self.size = [] 
        self.target_distance_max = []
        self.action_space = []
        self.observation_space = []
        
        self.dt = dt
        self.initial_margin = initial_margin
        
        #self.render_mode = render_mode
        
        self.reset()
        
        self.n_actions = self.action_space.shape[0]
        self.n_observations = self.observation_space.shape[0]
        
        self.closest_obstacle = self.lidar_range

        self.sim_length = sim_length  # maximum simulation length (in seconds)

    #####################################################################################################            
    def get_max_iterations(self):
        return int(round(self.sim_length / self.dt))

    #####################################################################################################            
    def get_actions_structure(self):
        pass # implemented in wrapper

        
    #####################################################################################################            
    def get_observations_structure(self):
        return [self.lidar_n_rays , (self.n_observations - self.lidar_n_rays) ]


    #####################################################################################################            
    # this function returns a traditional control input to be used instead of the ML one
    def get_controller_input(self):
        pass

    #####################################################################################################            
    def step(self,action, *args):
        # initial distance
        dist_0 = self.robot.target_rel_position(self.target_coordinates[:2])[0]
        
        dot_x      = np.maximum(0,np.minimum(self.robot.current_speed['linear']['x'] +  action[0],  self.linear_max))##Rivedere
        dot_orient = np.maximum(-self.angular_max,np.minimum(self.robot.current_speed['angular']['z']+  action[1], self.angular_max))##Rivedere
        #dot_orient = 0
        
        #print(f'orientation = {robot.current_pose["orientation"]["z"]}')
        #print(f'angular speed = {robot.current_speed["angular"]["z"]}')
        
        self.robot.map.peds_sim_env.step(1)
        self.robot.map.update_from_peds_sim() 
        self.robot.updateRobotSpeed(dot_x, dot_orient)
        self.robot.computeOdometry(self.dt)
        
        # new distance
        dist_1 = self.robot.target_rel_position(self.target_coordinates[:2])[0]
        
        self.robot.getScan(scan_noise = self.scan_noise)
        self.rotation_counter += np.abs(dot_orient*self.dt)

        
        # if pedestrian or obstacle is hit
        if self.robot.check_pedestrians_collision(.5) or self.robot.check_obstacle_collision():
            
            final_distance_penalty = 0.5*np.clip((dist_1/(1e-5+self.distance_init))**2,0,2)
            survival_time_penalty = (1-(self.duration/self.sim_length)**2)
            
            failure_penalty = self.rewards[1]*( 1 + final_distance_penalty + survival_time_penalty )
            
            reward = -failure_penalty 
            done = True

        
        # if target is reached            
        elif self.robot.check_target_reach(self.target_coordinates[:2], tolerance = 1):

            # "0<=a<=1" indicates how much the initial distance has to be weighted in the reward
            a = 0.5
            weight = (1-a)+a*self.distance_init/self.target_distance_max
            
            t_max= self.sim_length
            t_min = self.distance_init/self.linear_max
            t = self.duration
            rel_rew = (t_max-t)/(t_max - t_min)

            cum_rotations = (self.rotation_counter/(2*np.pi))
            rotations_penalty = self.rewards[2] * cum_rotations / (self.duration)
            
            # reward is proportional to distance covered / speed in getting to target
            reward = self.rewards[1]*rel_rew* weight - rotations_penalty
            done = True
                        
        # if nothing major happened         
        else:
            self.duration += self.dt
            # if time expired
            if self.duration >= self.sim_length:
                reward = - self.rewards[1]* ( 0.5 + 0.5*np.clip((dist_1/(1e-5+self.distance_init))**2,0,2))
                done = True
                
            else:
                # standard  reward calculation
                speed_penalty = 1- dot_x/self.linear_max
                ang_speed_penalty = np.abs(dot_orient)/self.angular_max
                
                peds_distances = self.robot.getPedsDistances()
                danger_penalty = np.sum( 0.2*(1 - (peds_distances[peds_distances < self.lidar_range]/self.lidar_range)**2) )

                reward = self.rewards[0]* (1 - 0.3*speed_penalty - 0.3*ang_speed_penalty\
                                           - 0.4*danger_penalty) #+ 10*(dist_0-dist_1)/self.target_distance_max)
                done = False

            
        return self._get_obs(), reward, done, {}
    
    #####################################################################################################
    def reset(self,**kwargs):
        self.duration = 0
        self.rotation_counter = 0
        
        self.robot = initialize_robot(self.visualization_angle_portion, self.lidar_range,self.lidar_n_rays, [0,0,0], self.collision_distance)
        self.robot.setMaxSpeed(self.linear_max, self.angular_max)

        low_bound,high_bound = self._getPositionBounds()   
        
        count = 0
        min_distance = (high_bound[0]-low_bound[0])/10
        while True:
            self.target_coordinates = np.random.uniform(low = np.array(low_bound), high = np.array(high_bound),size=3)
            if np.amin(np.sqrt(np.sum((self.robot.map.obstaclesCoordinates - self.target_coordinates[:2])**2, axis = 1)))>min_distance:
                break
            count +=1
            if count >= 100:
                raise('suitable initial coordinates not found')
            
        # if initial condition is too close (1.5m) to obstacle or pedestrians, generate new initial condition
        while self.robot.check_collision(1.5) or self.robot.checkOutOfBounds(margin = 0.2):
            init_coordinates = np.random.uniform(low = low_bound, high = high_bound,size=3)
            self.robot.setPosition(init_coordinates[0], init_coordinates[1], init_coordinates[2])
            
        # initialize Scan to get dimension of state (depends on ray cast) 
        self.robot.getScan()
        self.size = len(self.robot.scanner.scan_structure['data']['angles'])
        self.target_distance_max = np.sqrt(self.robot.map.map_height**2+self.robot.map.map_length**2)

        state_max = np.concatenate(  (self.lidar_range*np.ones((self.size,)), np.array([self.linear_max ,self.angular_max  , self.target_distance_max , np.pi])  ), axis = 0 )
        state_min = np.concatenate(  ( np.zeros((self.size,)), np.array([0 , -self.angular_max ,0,  -np.pi])  ), axis = 0 )

        self.action_low = np.array([-self.max_v_x_delta,-self.max_v_rot_delta])
        self.action_high = np.array([ self.max_v_x_delta, self.max_v_rot_delta])

        self.action_space = spaces.Box(low=self.action_low, high=self.action_high , dtype=np.float64) #, shape=(size__,))
        self.observation_space = spaces.Box(low=state_min, high=state_max , dtype=np.float64) # , shape=(len(state_max),))

        self.distance_init = self.robot.target_rel_position(self.target_coordinates[:2])[0]
        
        # aligning peds_sim_env and robot step_width
        if self.dt is None:
            self.dt = self.robot.map.peds_sim_env.peds.step_width
        else:
            self.robot.map.peds_sim_env.peds.step_width = self.dt
        
        return self._get_obs()
    
        


    #####################################################################################################
    def _getPositionBounds(self):
        # define bounds for initial and target position of robot (margin = margin from map limit)
        margin = self.initial_margin
        x_idx_min = round(margin*self.robot.map.grid_size['x'])
        x_idx_max = round((1-margin)*self.robot.map.grid_size['x'])

        y_idx_min = round(margin*self.robot.map.grid_size['y'])
        y_idx_max = round((1-margin)*self.robot.map.grid_size['y'])
        
        low_bound = [ self.robot.map.X[0,x_idx_min] , self.robot.map.Y[y_idx_min,0] , -np.pi]
        high_bound = [ self.robot.map.X[0,x_idx_max] , self.robot.map.Y[y_idx_max,0] , np.pi]
        
        return low_bound,high_bound


    #####################################################################################################
    def _get_obs(self):
        ranges = self.robot.scanner.scan_structure['data']['ranges']
        ranges_np = np.nan_to_num(np.array(ranges), nan = self.lidar_range)
        speed_x = self.robot.current_speed['linear']['x']
        speed_rot = self.robot.current_speed['angular']['z']
        target_distance = self.robot.target_rel_position(self.target_coordinates[:2])[0]
        target_angle =    self.robot.target_rel_position(self.target_coordinates[:2])[1]
        
        self.closest_obstacle = np.amin(ranges_np)

        if self.normalize_obs_state:
            ranges_np /= self.lidar_range
            speed_x /= self.linear_max
            speed_rot = (speed_rot+self.angular_max)/(2*self.angular_max)
            target_distance /= self.target_distance_max
            target_angle = (target_angle+np.pi)/2*np.pi

        self.state =np.concatenate((ranges_np,np.array([speed_x,speed_rot, target_distance,target_angle])  ),axis = 0).astype(np.float32)
    
        return self.state


    #####################################################################################################    
    def render(self,mode = 'human',iter_params = (0,0,0)):

        # mode="rgb_array"
        #shape_grid = self.robot.map.Occupancy.shape
        
        # get robot coordinates and convert them to grid              
        robot_coords = self.robot.getCurrentPosition()  # world coordinates
        idx_y,idx_x = self.robot.map.convert_world_to_grid_no_error(np.array(robot_coords[:2])[np.newaxis,:]).tolist()[0]
        robot_angle =    robot_coords[-1] #+ np.pi/2
        
        # grid scanned points
        scan_pts_grid = self.robot.scanner.scan_structure['data']['grid'] #remove after debugging
        
        # occupancy map
        occupancy_map = self.robot.map.OccupancyOverall
        
        
        # rlative width from closest obstacle
        closest_obstacle = self.closest_obstacle
        rel_width = 1- ( (self.closest_obstacle - self.robot.rob_collision_radius)/self.lidar_range )
        
        # target coordinates (grid based)
        y_trg_grd, x_trg_grd = self.robot.map.convert_world_to_grid_no_error(self.target_coordinates[:2].reshape(1,2))[0,:]
        
        
        img  , t1,t2, t3,t_ped_dist_line, trg, rbt,rbt_dir, scn, ped_dist_bar_empty,\
            ped_dist_bar,t_target_dist_line, target_dist_bar_empty, target_dist_bar = \
            render_image(occupancy_map,idx_x,idx_y,robot_angle, scan_pts_grid,  closest_obstacle, \
            rel_width, x_trg_grd, y_trg_grd, iter_params, \
                target_distance  = self.robot.target_rel_position(self.target_coordinates[:2])[0], \
                target_rel_width = 1-self.state[-2])

        self.data_render.append([occupancy_map,idx_x,idx_y,robot_angle, scan_pts_grid,  closest_obstacle, rel_width, x_trg_grd,y_trg_grd, iter_params])

        #plt.show()
        if mode == 'data':
            return idx_x,idx_y,robot_angle, scan_pts_grid, occupancy_map, closest_obstacle, rel_width, x_trg_grd,y_trg_grd, iter_params
        elif mode == 'animation':
            return img  , t1,t2, t3,t_ped_dist_line, trg, rbt,rbt_dir, scn, ped_dist_bar_empty, ped_dist_bar,t_target_dist_line, target_dist_bar_empty, target_dist_bar
        elif mode == 'plot':
            return plt.gcf()
        pass
            
#%%
    
def render_image(occupancy_map,idx_x,idx_y,robot_angle, scan_pts_grid,  closest_obstacle, rel_width, x_trg_grd,y_trg_grd, iter_params, target_distance = 10, target_rel_width = 0.5):
    
    # display parameters
    robot_width = 1*15
    robot_height = 1*10
    robot_angle=-robot_angle
    arrow_len = 20
    
    shape_grid = occupancy_map.shape
    # define new image object
    img = Image.new('1', (shape_grid[0], shape_grid[1]))
    
    # plot occupancy data            
    img = plt.imshow(occupancy_map, cmap='Greys',  interpolation='nearest', animated=True)
    
    # draw robot as a Rectangle patch (grid based) and add it to the axes object of img
    
    rect = patches.Rectangle((idx_x-robot_width/2, idx_y-robot_height/2 ),robot_width,robot_height, 0,linewidth=0.3,edgecolor='k',facecolor='b')
    
    transf_rect = mpl.transforms.Affine2D().rotate_deg_around(idx_x,idx_y,np.degrees(robot_angle)) + img.axes.transData
    rect.set_transform(transf_rect)
    #print(robot_angle)
    rect_direction = patches.Arrow(idx_x,idx_y,arrow_len*np.cos(robot_angle), arrow_len*np.sin(robot_angle) , width = 10)
    rbt = img.axes.add_patch(rect)
    rbt_dir = img.axes.add_patch(rect_direction)
    
    #robot_pos= img.axes.scatter(idx_x,idx_y,s= 60,c= 'k')
    # plot target (grid based)
    
    trg = img.axes.scatter(x_trg_grd, y_trg_grd,s= 60,c= 'g')
    
    # plot scanned points -- temporary, will have to move to polar coordinates
    ##pts = self.robot.map.convert_world_to_grid(np.array(self.robot.scanner.scan_structure['data']['cartesian']))
    ##scn = img.axes.scatter(pts[:,0], pts[:,1], marker = '.', color = 'green')
    
    # grid-based plot
    scn = img.axes.scatter(scan_pts_grid[:,1], scan_pts_grid[:,0], marker = '.',s= 20,c= 'r')
    
    # add training info
    iteration, single_run, cum_reward = iter_params

    t1 = img.axes.text(20,15,'iteration = '+ str(iteration), fontsize=10, color = 'g')
    t2 = img.axes.text(20,30,'run = '+ str(single_run), fontsize=10, color = 'b')
    t3 = img.axes.text(20,45,'tot reward = '+ str(round(cum_reward,2)), fontsize=10, color = 'm')
    
    dist_bar_max_width = 100
    dist_bar_height = 10
    
    
    t_ped_dist_line = img.axes.text(190,15,'ped alert = '+ str(round(closest_obstacle,1)), fontsize=10, color = 'k')
    
    ped_dist_line_empty = patches.Rectangle((190,20),dist_bar_max_width,dist_bar_height,0,edgecolor='k',facecolor='w'  )
    ped_dist_line = patches.Rectangle((190,20),rel_width*dist_bar_max_width,dist_bar_height,0,edgecolor='k',facecolor='r'  )
    ped_dist_bar_empty = img.axes.add_patch(ped_dist_line_empty)
    ped_dist_bar = img.axes.add_patch(ped_dist_line)
    
    
    t_target_dist_line = img.axes.text(190,45,'target reach = '+ str(round(target_distance,1)), fontsize=10, color = 'k')
    
    target_dist_line_empty = patches.Rectangle((190,50),dist_bar_max_width,dist_bar_height,0,edgecolor='k',facecolor='w'  )
    target_dist_line = patches.Rectangle((190,50),target_rel_width*dist_bar_max_width,dist_bar_height,0,edgecolor='k',facecolor='g'  )
    target_dist_bar_empty = img.axes.add_patch(target_dist_line_empty)
    target_dist_bar = img.axes.add_patch(target_dist_line)
    
        
    return img  , t1,t2, t3, t_ped_dist_line, trg, rbt,rbt_dir, scn, ped_dist_bar_empty, ped_dist_bar,t_target_dist_line, target_dist_bar_empty, target_dist_bar
    
    
#%%

def turn(steps,dv,dtheta):
    env = RobotEnv()
    env.robot.setPosition(0, -10, 0)
    for i in range(steps):
        env.step([dv,dtheta])
        env.render(mode = 'animation')
        
def goStraight(steps,dv,orient):
    env = RobotEnv()
    env.robot.setPosition(0, -10, orient)
    for i in range(steps):
        env.step([dv,0])
        env.render(mode = 'animation')
        
def s_like_path():
    env=RobotEnv()
    env.robot.setPosition(0, -10, 0)
    done = False
    
    for i in range(30):
        _, reward , done, _ = env.step([0.1,0.02])
        env.render(mode='animation')
        print(round(reward,3), done)
        if done:
            break
        
    if not done:
        for i in range(30):
            _, reward , done, _ = env.step([0.1,-0.04])
            env.render(mode='animation')
            print(round(reward,3), done)
            if done:
                break

#%%    

if __name__ == "__main__":
    env = RobotEnv()
    # env.step([1,1])
    # env.render(mode='animation')
    
    
    #turn(50,0.1,0.01) #Turn left
    #turn(50,0.1,-0.02) #Turn right
    #goStraight(50, 0.1,0) #Go straight to the right
    #goStraight(50,0.1,-np.pi) #Go straight to the left
    #goStraight(50,0.1,np.pi/2) #Go straight to the top
    #turn(30,0.1,0.02)
    s_like_path()
    
    current_folder = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    clear_pycache(current_folder)
    
    
    
#%%
"""
# test class
env = RobotEnv()
env.reset()
env.robot.map.show()
print(env.robot.map.OccupancyFixed.sum())

#%%
env.step([.2,-.05 ])
print(env.robot.map.OccupancyFixed.sum())
env.render(mode = 'plot').show()

env.step([.2,-.05 ])
print(env.robot.map.OccupancyFixed.sum())
env.render(mode = 'plot').show()
"""
#%%%%

#below here try first test of RL
#observation_size = env.observation_space.shape[0]
