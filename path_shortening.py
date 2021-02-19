#!/usr/bin/env python
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
from random import random

def isCollisionFreeVertex(obstacles, point):
    x,y,z = point
    for obstacle in obstacles:
        dx, dy, dz = obstacle.dimensions
        x0, y0, z0 = obstacle.pose
        if abs(x-x0)<=dx/2 and abs(y-y0)<=dy/2 and abs(z-z0)<=dz/2:
            return 0
    return 1

def isCollisionFreeEdge(obstacles, closest_vert, p):
    closest_vert = np.array(closest_vert); p = np.array(p)
    collFree = True
    l = norm(closest_vert - p)
    map_resolution = 0.01; M = int(l / map_resolution)
    if M <= 2: M = 20
    t = np.linspace(0,1,M)
    for i in range(1,M-1):
        point = (1-t[i])*closest_vert + t[i]*p # calculate configuration
        collFree = isCollisionFreeVertex(obstacles, point) 
        if collFree == False: return False

    return collFree



#============================================================Pruning functions
def uniPruning(path, obstacles):     #Pruning function
    unidirectionalPath=path[0]
    pointTem=path[0]
    for i in range(3,len(path)):
        if not isCollisionFreeEdge(obstacles, pointTem,path[i]):
            pointTem=path[i-2]
            unidirectionalPath = np.vstack((unidirectionalPath,pointTem))
    unidirectionalPath = np.vstack((unidirectionalPath,path[-1]))
    return unidirectionalPath