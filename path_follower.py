
import numpy as np
from numpy.linalg import norm
from math import *
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from random import random
from scipy.spatial import ConvexHull
from matplotlib import path
import time
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tools import init_fonts,drawStartNGoal, drawRRTpath, drawRRTSmoothpath
from path_shortening import isCollisionFreeVertex,isCollisionFreeEdge
from Obstacle import ObjecClass

class follower:
    def __init__(self):
    	pass
        
    def goto_point(self, point, ax, animate):
        if animate:
            ax.scatter3D(point[0], point[1], point[2], color='green', s=150, zorder=20)
    
    def closestObstacleDistNvel(self, Obstacles, currentPose):
        distances = []
        velocities =[]
        for obstacle  in Obstacles:
            dist = norm(obstacle.pose - currentPose)
            velocity = obstacle.velocity
            distances.append(dist)
            velocities.append(velocity)
            
        minDist = min(distances) #taking closest obstacle
        argmin = distances.index(minDist) #getting the index
        colsestObstacle_velociity = velocities[argmin]
        return minDist, colsestObstacle_velociity
