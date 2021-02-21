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
from path_shortening import isCollisionFreeVertex,isCollisionFreeEdge,uniPruning


class Node3D:
    def __init__(self):
        self.p     = [0, 0, 0]
        self.i     = 0
        self.iPrev = 0


def closestNode3D(rrt, p):
    distance = []
    for node in rrt:
        distance.append( sqrt((p[0] - node.p[0])**2 + (p[1] - node.p[1])**2 + (p[2] - node.p[2])**2) )
    distance = np.array(distance)
    
    dmin = min(distance)
    ind_min = distance.tolist().index(dmin)
    closest_node = rrt[ind_min]

    return closest_node


def plot_point3D(p, color='blue'):
    ax.scatter3D(p[0], p[1], p[2], color=color)


def RRT(start,goal, ax, obstacles, animate, xbound, ybound, zbound):
    drawStartNGoal(start, goal, ax,animate)
    # RRT Initialization
    maxiters  = 750
    nearGoal = False # This will be set to true if goal has been reached
    minDistGoal = 0.05 # Convergence criterion: success when the tree reaches within 0.25 in distance from the goal.
    d = .5 # [m], Extension parameter: this controls how far the RRT extends in each step.
    # Initialize RRT. The RRT will be represented as a 2 x N list of points.
    # So each column represents a vertex of the tree.
    rrt = []
    start_node = Node3D()
    start_node.p = start
    start_node.i = 0
    start_node.iPrev = 0
    rrt.append(start_node)
    
    
    # RRT algorithm
    start_time = time.time()
    iters = 0
    while not nearGoal and iters < maxiters:
        # Sample point
        rnd = random()
        # With probability 0.05, sample the goal. This promotes movement to the goal.
        if rnd < 0.10:
            p = goal
        else:
            p = np.array([random()*xbound[1], random()*ybound[1], random()*zbound[1]]) # Should be a 3 x 1 vector
            
        # Check if sample is collision free
        collFree = isCollisionFreeVertex(obstacles, p)
        # If it's not collision free, continue with loop
        if not collFree:
            iters += 1
            continue
    
        # If it is collision free, find closest point in existing tree. 
        closest_node = closestNode3D(rrt, p)
        
        
        # Extend tree towards xy from closest_vert. Use the extension parameter
        # d defined above as your step size. In other words, the Euclidean
        # distance between new_vert and closest_vert should be d.
        new_node = Node3D()
        new_node.p = closest_node.p + d * (p - closest_node.p)
        new_node.i = len(rrt)
        new_node.iPrev = closest_node.i
    
        if animate:
            ax.plot([closest_node.p[0], new_node.p[0]], [closest_node.p[1], new_node.p[1]], [closest_node.p[2], new_node.p[2]],color = 'b', zorder=5)
            plt.pause(0.000001)
        
        # Check if new vertice is in collision
        collFree = isCollisionFreeEdge(obstacles, closest_node.p, new_node.p)
        # If it's not collision free, continue with loop
        if not collFree:
            iters += 1
            continue
        
        # If it is collision free, add it to tree    
        rrt.append(new_node)
    
        # Check if we have reached the goal
        if norm(np.array(goal) - np.array(new_node.p)) < minDistGoal:
            # Add last, goal node
            goal_node = Node3D()
            goal_node.p = goal
            goal_node.i = len(rrt)
            goal_node.iPrev = new_node.i
            if isCollisionFreeEdge(obstacles, new_node.p, goal_node.p):
                rrt.append(goal_node)
                P = [goal_node.p]
            else: P = []
            nearGoal = True

        iters += 1

    end_time = time.time()
    Ptime = end_time - start_time #planning time
    no_of_nodes = len(rrt)
    
    if iters >= maxiters:
        isSuccessful = 0
        P=None
        print("Planning Failed!!!")
        #raise RuntimeError("Maximum itteration reached")
        return P, Ptime,no_of_nodes, isSuccessful
        
    #print ('Number of iterations passed: %d / %d' %(iters, maxiters))
    print ('RRT nodes: ', len(rrt))
    print ('Planning time : %.2f seconds:' % (Ptime))
    
    
    # Path construction from RRT:
    #print ('Constructing the path...')
    i = len(rrt) - 1
    while True:
        i = rrt[i].iPrev
        P.append(rrt[i].p)
        if i == 0:
            #print ('Reached RRT goal node')
            break
    P = np.array(P)
    P  = np.flip(P, axis=0)
    
    isSuccessful = 1
    if animate:
        plt.show()
    return P, Ptime,no_of_nodes, isSuccessful  #returns rrt path and planning time