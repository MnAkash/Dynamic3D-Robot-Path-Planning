#!/usr/bin/env python
import numpy as np
import pandas as pd
from numpy.linalg import norm
from math import *
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from random import random
from scipy.spatial import ConvexHull
import time
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tools import init_fonts,drawStartNGoal, drawRRTpath, drawRRTSmoothpath
from path_shortening import isCollisionFreeVertex,isCollisionFreeEdge,uniPruning
from Obstacle import ObjecClass
from RRT import RRT
from drrt_connect import drrt_connect
from path_follower import follower
import  csv

no_of_objects = 6
object_size = 8
safety_radius= 10 #assuming it will be within this region after 1 second
xbound = (0,100)
ybound= (0,100)
zbound= (0,90)
stepSize = 2 #stepsize of subject moment that defines the speed of the robot(not rrt stepSize)
consciousness_dist = range(20,81,5) #(i,j,k) >> i to j-1, stepsize=k
min_safeApproach_dist = 10
waitTime = 10 #object waiting moments if obstructed by a obstacle inside min_safeApproach_dist

'''We can let Safe_waitTime be as a variable parameter using the speed of obstacle'''
Safe_waitTime = 10 #object waiting moments while moving slow between consciousness_dist and min_safeApproach_dist

experients = 15 #Number of experiments to conduct

RRTanimate = 0 #wiill node exploration animation be shown
subjectAnimate = 0 #wiill subject path following animation be shown

algorithm = 'drrt' #define which path planning algorithm to use 'rrt' and 'drrt'
#algorithm = 'drrt'



# Start and goal positions
start = np.array([20, 20, 0])
goal =  np.array([80, 85, 68])
    


def defineAxis():
    init_fonts()
    global xbound
    global ybound
    global zbound
    ax = plt.axes(projection='3d')
    ax.set_xlabel('X, [m]')
    ax.set_ylabel('Y, [m]')
    ax.set_zlabel('Z, [m]')
    ax.set_xlim([xbound[0], xbound[1]])
    ax.set_ylim([ybound[0], ybound[1]])
    ax.set_zlim([zbound[0], zbound[1]])
    #ax.view_init(elev=22, azim=0)
    return ax

def calc_pathLength(path):
    #calculate the distance of path
    #input parameter is whole path in list
    #Will return the path length
    pathLength=0
    for i in range(len(path)-1):
        pathLength += norm(path[i]-path[i+1])
    return pathLength

def isReached(present, goal):
    if present[0]==goal[0] and present[1]==goal[1] and present[2]==goal[2]:
        return True
    else:
        return False

#This functiong is used to plan a path using particular Algorithm(as parameter) 
def plan(algorithm,ax,start, goal, obstacles,RRTanimate,subjectAnimate, xbound, ybound, zbound):
    # obstacles = obj.give_objects()
    # obj.draw(ax)
    isSuccessful = 1
    if algorithm == 'rrt':
        P, planning_time,no_of_nodes,isSuccessful = RRT(start, goal, ax, obstacles,RRTanimate,xbound, ybound, zbound)
    elif algorithm == 'drrt':
        P, planning_time,no_of_nodes,isSuccessful = drrt_connect(start, goal, ax, obstacles,RRTanimate,xbound, ybound, zbound)
    
    if not isSuccessful:
        return P, 0, 0, planning_time, no_of_nodes, isSuccessful
    drawRRTpath(P,ax, subjectAnimate)

    sP = uniPruning(P, obstacles)
    drawRRTSmoothpath(sP, ax,subjectAnimate)

    interpolation_coeff = 4
    
    interpolated_points = sP[0]
    for i in range(len(sP)-1):
        p1 = sP[i]
        p2 = sP[i+1]
        dist = norm(p1-p2)
        #print(dist)
        
        #Task space interpolation operation through waypoints
        waypoint_interpolated = np.linspace(p1,p2, int(dist*interpolation_coeff))
        
        interpolated_points = np.vstack((interpolated_points, waypoint_interpolated)) #appending at one big list
    print("RRT path distance: ",calc_pathLength(P))
    #print("Short path distance: ",calc_pathLength(sP))
    #print(interpolated_points)
    
    return P, sP, interpolated_points, planning_time, no_of_nodes, isSuccessful



#This function will generate the values to calculate final result(in cDist fuction) after each experiment
#For example if experients=50, it will be called 50 time for each conciousness distance.
def main(RRTanimate, subjectAnimate, c_dist):
    no_of_collision = 0 #will be used to count nuber of collisions
    if subjectAnimate:
        #define plot window size
        fig = plt.figure(figsize=(11,6))
        #Define initial axis
        ax = defineAxis()
    else:
        ax=0
    
    
    
    #declare object of ostacles
    obj = ObjecClass(start, goal, no_of_objects ,object_size, safety_radius, xbound, ybound, zbound)
    
    
    
    obstacles = obj.give_objects()
    if subjectAnimate:
        obj.draw(ax)
    
    #Planning RRT
    print("========Initial Planning======")
    P ,sP ,interpolated_points,planning_time,no_of_nodes, isSuccessful = plan(algorithm,
                                                                              ax,
                                                                              start,
                                                                              goal,
                                                                              obstacles,
                                                                              RRTanimate,
                                                                              subjectAnimate,
                                                                              xbound,
                                                                              ybound,
                                                                              zbound)
    #taking measures if planning unsuccessful
    if not isSuccessful:
                print('Total Wasted planning time %.2f seconds' % (planning_time))
                return planning_time, no_of_nodes, 0, isSuccessful, no_of_collision
            
    total_planning_time = planning_time
    total_no_of_nodes = no_of_nodes
    subject = follower()
    currentPose = start
    previousPose = currentPose
    visited_points = [currentPose]
    i =0
    wait = 0
    safe_wait = 0
    while not isReached(currentPose, interpolated_points[-1]):
        #If stays slow(between i.e. 20-10cm)  and standstill(inside e.g 10cm) for waitTime
        if wait < waitTime and safe_wait<Safe_waitTime :
            obstacles = obj.give_objects()#fetching latest obstacle positions
            
            if subjectAnimate:
                #plt.cla()#claer plots to animate next frame
                #ax = defineAxis()
                ax.clear()
            drawStartNGoal(start,goal, ax, subjectAnimate)
            
            drawRRTpath(P, ax, subjectAnimate)# drawing path of RRT
            
            drawRRTSmoothpath(sP, ax,subjectAnimate)# drawing smoothed RRT generated path
            
            #change obstacle direction if it colides with subject
            #obj.change_dir_if_Collide_subject(currentPose, goal)
            
            
            #move function updates the object according to direction vector
            obj.move(ax,subjectAnimate)
            
            
            
            #Setting speed configuration
            closestObstacle_dist = subject.closestObstacleDist(obstacles, currentPose)
            if closestObstacle_dist> c_dist:
                stepSize = 2
                i = i+1
                wait = 0
                safe_wait =0
            elif min_safeApproach_dist < closestObstacle_dist < c_dist:
                stepSize = 1
                i = i+1
                wait = 0
                safe_wait +=1
            else:
                stepSize = 0
                if wait == 0:# to count collision one time 
                    no_of_collision += 1
                wait += 1
                safe_wait=0
                
            i = i+stepSize
            #hanlding when jump goes over the goal
            if len(interpolated_points)>i:
                jump = i
            else:
                jump = len(interpolated_points)-1
            
            # defining where to jump    
            currentPose = interpolated_points[jump]
            
            #saving all visited points
            if (currentPose!= previousPose).all():
                visited_points.append(currentPose)
                                
            #command to move subject to a point    
            subject.goto_point(currentPose, ax, subjectAnimate)
            
            previousPose = currentPose

            if subjectAnimate:
                plt.pause(0.01)
        else:
            print("=========Replanning=========")
            P ,sP ,interpolated_points,planning_time,no_of_nodes, isSuccessful = plan(algorithm,
                                                                                     ax,
                                                                                     currentPose, 
                                                                                     goal,
                                                                                     obstacles,
                                                                                     RRTanimate,
                                                                                     subjectAnimate,
                                                                                     xbound,
                                                                                     ybound,
                                                                                     zbound)
            i=0
            wait=0
            safe_wait =0
            total_planning_time +=planning_time
            total_no_of_nodes += no_of_nodes
            if not isSuccessful:
                print('Total Wasted planning time %.2f seconds' % (total_planning_time))
                return total_planning_time, total_no_of_nodes, 0, isSuccessful, no_of_collision
    
    if subjectAnimate:
        time.sleep(1)
    plt.close()
    path_length = calc_pathLength(visited_points)
    print("\nBatch result:")
    print('Total planning time %.2f seconds' % (total_planning_time))
    print('Total nodes explored :',total_no_of_nodes)
    print('Visited path length :',path_length)
    return total_planning_time, total_no_of_nodes, path_length, isSuccessful, no_of_collision



#This function will generate the final result after all experiment for a single conciousness distance
#Will be called single time for each consciousness distance
def simulate_for_each_cDist(c_dist):
    all_planning_timeList = []#list of all  total planning time
    all_no_of_nodesList = []#list of all total no_of_nodes
    all_path_lengthList = []#list of path length subject visited in each experiment
    successList = []
    Wasted_timeList = []
    no_of_collision_List = []
    
    
    for i in range(experients):
        print("\n=======================Experiment:",i+1,"======================")
        total_planning_time,total_no_of_nodes,path_length,isSuccessful,no_of_collision =main(RRTanimate, subjectAnimate, c_dist)

        if isSuccessful:
            all_planning_timeList.append(total_planning_time)
            all_no_of_nodesList.append(total_no_of_nodes)
            all_path_lengthList.append(path_length)
            successList.append(isSuccessful)
            no_of_collision_List.append(no_of_collision)
        else:
            successList.append(0)
            Wasted_timeList.append(total_planning_time)
    
    s_rate = (sum(successList)/experients)*100
    p_time = sum(all_planning_timeList)/(len(all_planning_timeList) if len(all_planning_timeList) else 1)
    node_no = sum(all_no_of_nodesList)/(len(all_no_of_nodesList) if len(all_no_of_nodesList) else 1)
    p_len = sum(all_path_lengthList)/(len(all_path_lengthList) if len(all_path_lengthList) else 1)
    #wasted = sum(Wasted_timeList)/(len(Wasted_timeList) if len(Wasted_timeList) else 1)
    avg_no_of_collision = sum(no_of_collision_List)/(len(no_of_collision_List) if len(no_of_collision_List) else 1)
    print("\n\nFinal Resuts:")
    print("\nSuccess Rate: ", s_rate, "%")
    print('Avg (successful)planning time %.2f seconds' % (p_time))
    print('Avg (successful)no of nodes explored:',node_no)
    print('Avg (successful)path length:',p_len)
    #print('Avg wasted time in unsuccessful planning %.2f seconds' % (wasted))
    print('Avg number of collision to generate a successful path:',avg_no_of_collision)

    return s_rate, p_time, node_no, p_len, avg_no_of_collision




with open("Results/"+algorithm+".csv", 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["consciousness Distance", "Success Rate", "Planning time","No of nodes", "Path length", "Collisions"])
    
    for cd in consciousness_dist:
        print("\n\nExperiment for consciousnedd distance: ", cd)
        s_rate, p_time, node_no, p_len, avg_no_of_collision = simulate_for_each_cDist(cd)
        writer.writerow([cd, s_rate, p_time, node_no, p_len, avg_no_of_collision])
        
    
    
    