from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import random as rn
import numpy as np
from path_shortening import isCollisionFreeVertex
from numpy.linalg import norm
"""
Object class cretaes cubic obstacles. A parameter for the obstacles is the safety_radius. 
safety_radius is necessary because without it, RRT can pick random nodes which are very near 
to the obstacles and thus some edges can be tangent to the obstacles, which we are considering 
a problem. As in real world there will be a window of time between each iteration of our calculation
within which the obstacles may move and invalidate some edge. In other words, safety radius can be 
considered as an invisible boundary around the obstacles which will considered while checking for 
collision. 

Each obstacle also has a predefined direction vector to move along. The direction is updated if and 
only if it intersects with the boundary. 

"""

# Class to manage all Obstacles cube variables(positions,dimensions and wireframe vertices)
class Parallelepiped:
    def __init__(self):
        self.dimensions = [0,0,0]
        self.pose = [0,0,0]
        self.poseList =[]
        self.currIndx = 0
        self.verts = self.vertixes()
        
    def vertixes(self):
        dx = self.dimensions[0]
        dy = self.dimensions[1]
        dz = self.dimensions[2]
        C = np.array(self.pose)

        Z = np.array([[-dx/2, -dy/2, -dz/2],
                      [dx/2, -dy/2, -dz/2 ],
                      [dx/2, dy/2, -dz/2],
                      [-dx/2, dy/2, -dz/2],
                      [-dx/2, -dy/2, dz/2],
                      [dx/2, -dy/2, dz/2 ],
                      [dx/2, dy/2, dz/2],
                      [-dx/2, dy/2, dz/2]])
        Z += C

        # list of sides' polygons of figure
        verts = [ [Z[0], Z[1], Z[2], Z[3]],
                  [Z[4], Z[5], Z[6], Z[7]], 
                  [Z[0], Z[1], Z[5], Z[4]], 
                  [Z[2], Z[3], Z[7], Z[6]], 
                  [Z[1], Z[2], Z[6], Z[5]],
                  [Z[4], Z[7], Z[3], Z[0]] ]

        return verts

    def vertixesToDraw(self,safety_radius):
        dx = self.dimensions[0] - safety_radius
        dy = self.dimensions[1] - safety_radius
        dz = self.dimensions[2] - safety_radius
        C = np.array(self.pose)

        Z = np.array([[-dx/2, -dy/2, -dz/2],
                      [dx/2, -dy/2, -dz/2 ],
                      [dx/2, dy/2, -dz/2],
                      [-dx/2, dy/2, -dz/2],
                      [-dx/2, -dy/2, dz/2],
                      [dx/2, -dy/2, dz/2 ],
                      [dx/2, dy/2, dz/2],
                      [-dx/2, dy/2, dz/2]])
        Z += C

        # list of sides' polygons of figure
        verts = [ [Z[0], Z[1], Z[2], Z[3]],
                  [Z[4], Z[5], Z[6], Z[7]], 
                  [Z[0], Z[1], Z[5], Z[4]], 
                  [Z[2], Z[3], Z[7], Z[6]], 
                  [Z[1], Z[2], Z[6], Z[5]],
                  [Z[4], Z[7], Z[3], Z[0]] ]

        return verts

    def draw(self, ax, safety_radius):
        ax.add_collection3d(Poly3DCollection(self.vertixesToDraw(safety_radius), facecolors='k', linewidths=1, edgecolors='k', alpha=.25))



class ObjecClass:
    def __init__(self, start, goal, Hands , object_size = 5, safety_radius=2, xbound = (0,50), ybound= (0,50), zbound= (0,30)):
        self.safety_radius = safety_radius #invisible boundary outside the object
        self.obstacles_poses = [] #obstacles positions
        self.obstacles_poses_list = []
        self.obstacles_dims = [] #obsatacles dimensions
        self.no_of_objects = 2 # number of object in workspace
        self.object_size = object_size #sides of the cubic object(considering all are same sized)
        #boundary of workspace(provided in class parameter)
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound

        self.obstacles = [] #all objects will be stored here
        self.start = start
        self.goal = goal
        self.Hands = Hands

        self.create_object() #command to create objects and update/store 'obstacles' positions

    def create_object(self):
        xbound = self.xbound
        ybound = self.ybound
        zbound = self.zbound
        size_with_safety = self.object_size + self.safety_radius
        for h in self.Hands:
            self.obstacles_poses.append(h[0])
            self.obstacles_poses_list.append(h)
            #appending the cube dimensions of all objects
            self.obstacles_dims.append([size_with_safety, size_with_safety, size_with_safety] )

        #Zip positions and dimensions respectively and send to add_obstacle() to create 'obstacels' object
        for pose, dim, poseList in zip(self.obstacles_poses, self.obstacles_dims, self.obstacles_poses_list):
            self.obstacles = self.add_obstacle(self.obstacles, pose, dim, poseList)


    def give_objects(self):#return obstacles to use in main script for path planning
        return self.obstacles
    

    def draw(self, ax):
        for obstacle in self.obstacles: obstacle.draw(ax,self.safety_radius)

    def add_obstacle(self, obstacles, pose, dim, poseList):
        obstacle = Parallelepiped()
        obstacle.dimensions = dim
        obstacle.pose = pose
        obstacle.poseList = poseList
        obstacles.append(obstacle)
        return obstacles

    
    def move(self, ax, animate):
        """
        This function moves the center of the cube one step along directon vector.
        More specifically, for each vertex, new_vertex = old_vertex + direction_vector (vector addition).
        """
        try:
            self.currIndx +=5;
        except :
            self.currIndx = 0
        
        for obstacle in self.obstacles:
            obstacle.pose = obstacle.poseList[self.currIndx]

        if animate:
            self.draw(ax)

    #change obstacle direction if it colides with subject or goal
    def change_dir_if_Collide_subject(self, currentPose, goal):
        if not isCollisionFreeVertex(self.obstacles, currentPose) or not isCollisionFreeVertex(self.obstacles, goal):
            for i in range(self.no_of_objects):
                self.obstacles[i].dir[0] = -self.obstacles[i].dir[0]
                self.obstacles[i].dir[1] = -self.obstacles[i].dir[1]
                self.obstacles[i].dir[2] = -self.obstacles[i].dir[2]
            #print("Collision about to be happened")
            for obstacle in self.obstacles:
                obstacle.pose[0] = obstacle.pose[0] + obstacle.dir[0] #add x direction vector
                obstacle.pose[1] = obstacle.pose[1] + obstacle.dir[1] #add y direction vector
                obstacle.pose[2] = obstacle.pose[2] + obstacle.dir[2] #add z direction vector