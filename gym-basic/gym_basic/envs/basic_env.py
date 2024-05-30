import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import PIL.Image as Image
import gym
import random

from gym import Env, spaces
import time

font = cv2.FONT_HERSHEY_COMPLEX_SMALL 

class DroneMaze(gym.Env):
    def __init__(self):
        super(DroneMaze, self).__init__()


        self.observation_shape = (1000,1450, 3)

        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape),
                                            high = np.ones(self.observation_shape),
                                            dtype = np.float16)

        self.action_space = spaces.Discrete(6,)

        self.elements = []

        self.max_fuel = 1000

        self.y_min = int(self.observation_shape[0] * 0.1)
        self.x_min = int(self.observation_shape[1] * 0.1)
        self.y_max = int(self.observation_shape[0] * 0.9)
        self.x_max = int(self.observation_shape[1] * 0.9)
    
    def draw_elements_on_canvas(self):
        self.canvas = np.ones(self.observation_shape) * 1

        for elem in self.elements:
            elem_shape = elem.icon.shape
            x,y = elem.x, elem.y
            self.canvas[y : y + elem_shape[1], x:x + elem_shape[0]] = elem.icon

        text = 'Rewards: {}'.format(self.ep_return)

        # Put the info on canvas 
        self.canvas = cv2.putText(self.canvas, text, (10,20), font,  
                                    0.8, (0,0,0), 1, cv2.LINE_AA)


    def reset(self):
        # Reset the reward
        self.ep_return  = 0

        # Determine a place to intialise the chopper in
        x = random.randrange(int(self.observation_shape[0] * 0.3), int(self.observation_shape[0] * 0.50))
        y = random.randrange(int(self.observation_shape[1] * 0.3), int(self.observation_shape[1] * 0.50))
        
        # Intialise the chopper
        self.Drone = Drone("Drone", self.x_max, self.x_min, self.y_max, self.y_min)
        self.Drone.set_position(x,y)

        # Intialise the elements 
        self.elements = [self.Drone]

        # Reset the Canvas 
        self.canvas = np.ones(self.observation_shape) * 1

        # Draw elements on the canvas
        self.draw_elements_on_canvas()


        # return the observation
        return self.canvas 


class Point(object):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        self.x = 0
        self.y = 0
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name
    
    def set_position(self, x, y):
        self.x = self.clamp(x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(y, self.y_min, self.y_max - self.icon_h)
    
    def get_position(self):
        return (self.x, self.y)
    
    def move(self, del_x, del_y):
        self.x += del_x
        self.y += del_y
        
        self.x = self.clamp(self.x, self.x_min, self.x_max - self.icon_w)
        self.y = self.clamp(self.y, self.y_min, self.y_max - self.icon_h)

    def clamp(self, n, minn, maxn):
        return max(min(maxn, n), minn)



class Drone(Point):
    def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Drone, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("Drone.png")
        self.icon_w = 64
        self.icon_h = 64
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))
    
class Wall(Point):
     def __init__(self, name, x_max, x_min, y_max, y_min):
        super(Wall, self).__init__(name, x_max, x_min, y_max, y_min)
        self.icon = cv2.imread("Wall.png") / 255.0
        self.icon_w = 80
        self.icon_h = 5
        self.icon = cv2.resize(self.icon, (self.icon_h, self.icon_w))


env = DroneMaze()
obs = env.reset()
plt.imshow(obs)