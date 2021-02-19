import time
import random
import numpy
import math
from Obstacle import ObjecClass
from path_shortening import isCollisionFreeVertex,isCollisionFreeEdge,uniPruning
import numpy as np

(ADVANCED, REACHED, TRAPPED) = ("ADVANCED", "REACHED", "TRAPPED")




###################### Implementation of Algorithm Section ######################
def calc_dist(x1, y1, z1, x2, y2, z2):
   
    d = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    return math.sqrt(d)


def scale(neighbour, point):
    """
    Parameters: neighbour (a node from RRT), point (some random point)
    this function returns point if the distance between point and neighbour is less than
    delta. Otherwise, it calculates the point at distance delta from neighbour on the line from
    neighbour to point (from parameter).
    """
    d = calc_dist(
        *neighbour, *point
    )  # distance between neighbour and point

    if d < delta:
        return point  

    dx = point[0] - neighbour[0]  
    dy = point[1] - neighbour[1]  
    dz = point[2] - neighbour[2]  
    dx = (dx / d) * delta  
    dy = (dy / d) * delta  
    dz = (dz / d) * delta  

    
    point = (neighbour[0] + int(dx), neighbour[1] + int(dy), neighbour[2] + int(dz))

    return point  




class RRT:
    def __init__(self, start, end, obstacle,gridX,gridY,gridZ):
        """
        Initializes the following data structures:
        - graph
        - active_node
        - edge_list
        - node_list
        - path_vectors
        """
        start = (int(start[0]),int(start[1]),int(start[2]))
        end = (int(end[0]),int(end[1]),int(end[2]))
        self.gridX = gridX
        self.gridY = gridY
        self.gridZ = gridZ
        
        
        # graph is the adjacency list for RRT
        # it is initialized as 2d array of list
        self.graph = [[[[] for J in range(self.gridZ)] for I in range(self.gridY)] for Y in range(self.gridX)]
        # active_node is used to mark the the points in the space which are also present in RRT.
        # it is initialized as 2d array
        self.active_node = [[[False for J in range(self.gridZ)] for I in range(self.gridY)] for Y in range(self.gridX)]

        self.edge_list = []  
        self.node_list = [] 
        
        self.obstacle = obstacle
        
        self.path_vectors = (
            []
        )  # path_vector maintains the list of nodes on the path from start to goal

        self.start_node = start
        self.goal_node = end

        self.add_node(*start)  # add the start node to RRT
        
    def random_point(self):
        """
        This function returns a random point.
        """
        p = random.random()  # pick a random number between range(0, 1)

        if p < goal_point_probability:
            return self.goal_node
        else:
            # a random point between (0, self.gridX - 1) x (0, self.gridY - 1)
            return (random.randint(0, self.gridX - 1), random.randint(0, self.gridY - 1), random.randint(0, self.gridZ - 1))

    def clear_ds(self):
        """
        This function clears the following data structures:
        - graph
        - node_list
        - active_node
        - edge_list
        """

        for node in self.node_list:  
            i, j, k = node 
            self.graph[i][j][k].clear()  
            self.active_node[i][j][k] = False  

        # clear edge_list and node_list.
        self.edge_list.clear()
        self.node_list.clear()

    def add_node(self, x, y, z):
        """
        Parameters: x, y (coordinates of a new node)
        given co-ordinates of a node, adds the node to our node_list, active_node
        """

        self.node_list.append((x, y, z))  
        self.active_node[x][y][z] = True  

    def add_edge(self, x1, y1, z1, x2, y2, z2):
        """
        Parameters: x1, y1, x2, y2
        This function adds an edge from (x1, y1) to (x2, y2) in RRT.
        """

        self.graph[x1][y1][z1].append(
            (x2, y2, z2)
        )  # append (x2, y2) to the adjacency list of (x1, y1)
        self.edge_list.append(
            ((x1, y1, z1), (x2, y2, z2))
        )  # append the edge ((x1, y1), (x2, y2)) to the edge_list
        self.add_node(x2, y2, z2) 



    def find_nearest_node(self, x, y, z):
        """
        Parameters: x, y (coordinates of a point)
        finds the nearest node of (x, y) from the RRT and returns it.
        """
        dist = 1e9  
        nearest_node = (-1, -1, -1)  

       
        for to in self.node_list:
            
            if calc_dist(*to, x, y, z) < dist:
                dist = calc_dist(*to, x, y, z)
                nearest_node = to

        assert nearest_node != (-1, -1, -1)  
        return nearest_node  


    def check_if_tree_in_goal_region(self):
        for i in range(-radius_around_goal, radius_around_goal + 1):
            for j in range(-radius_around_goal, radius_around_goal + 1):
                for k in range(-radius_around_goal, radius_around_goal + 1):
                    if calc_dist(i, j, k, 0, 0, 0) <= radius_around_goal:
                        if (
                            self.active_node[self.goal_node[0] + i][self.goal_node[1] + j][self.goal_node[2] + k]
                            == True
                        ):
                            return True

        return False

    def extend(self, point):
        if self.active_node[point[0]][point[1]][point[2]] == True:
            return (TRAPPED, None)  # if it is a node from RRT then we skip this node

        # find the nearest node to the random point and assign to neighbour
        neighbour = self.find_nearest_node(*point)
        temp = point
        point = scale(
            neighbour, point
        )  # the point is then scaled so that the distance is within delta
        if self.active_node[point[0]][point[1]][point[2]] == True:
            return (
                TRAPPED,
                None,
            )  # if the scaled point is a node on RRT then we skip this node

        # if the edge from neighbour to point collides with some obstacle, then the node is also skipped
        if isCollisionFreeEdge(self.obstacle, neighbour, point) == False:
            return (TRAPPED, None)

        # if no issue is found, add the edge from neighbour to point
        self.add_edge(*neighbour, *point)
        if point == temp:
            return (REACHED, point)

        return (ADVANCED, point)

    def grow_RRT(self):
        """
        This function continues to add new nodes and edges until a path to goal node is found.
        """

        while (
            not self.check_if_tree_in_goal_region()
        ):  # While goal_node is not included in RRT
            point = self.random_point()  # pick a random point

            self.extend(point)

        while not self.active_node[self.goal_node[0]][self.goal_node[1]][self.goal_node[2]]:
            self.extend(self.goal_node)

    def trim_RRT(self):
        """
        This function uses Depth First Search (DFS) to traverse the RRT until any edge collides with
        any obstacles, in which case the branch is not further traversed. Then the RRT is trimmed
        down to the subtree which is traversed by depth first search.
        """

        # declare a stack for DFS and initialized with start_node (root node of RRT).
        stack = [self.start_node]

        # new_edge_list maintains the list of edges which are traversed by DFS
        # without any collision with obstacle.
        new_edge_list = []

        while len(stack) > 0:  # DFS runs till the stack is not empty
            node = stack[-1]  # assign the top/last value of the stack to node.

            stack.pop()  # pop the top/last value from stack.

            x, y, z = node  # unpack (x, y) coordinates from the node.

            for to in self.graph[x][y][z]:  
                
                if isCollisionFreeEdge(self.obstacle, node, to) == False:
                    continue

                new_edge_list.append((node, to))
                stack.append(to)  # also node `to` is pushed/appended to the stack

        self.clear_ds()  # clear_ds is called for clearing previous RRT information.

        self.add_node(
            *self.start_node
        )  # start_node is added to the new RRT

        for edge in new_edge_list:  
            (x1, y1, z1), (x2, y2, z2) = edge  
            self.add_edge(x1, y1, z1, x2, y2, z2)  

    def generate_path(self):
        """
        This function uses a Depth First Search (DFS) to find the path from
        start_node to goal_node and stores the sequence of node in path_vector
        """
        self.path_vectors.clear() 
        stack = [self.start_node]  

        # initialize par as 2d array of tuple to store parent of each node.
        par = [[[(-1, -1, -1) for J in range(self.gridZ)] for I in range(self.gridY)] for Y in range(self.gridX)]

        while len(stack) > 0:  
            node = stack[-1]  
            stack.pop()  
            x, y, z = node  
            for to in self.graph[x][y][z]: 
                (
                    gox,
                    goy,
                    goz
                ) = to  
                par[gox][goy][goz] = node  
                stack.append(to)  

        x, y, z = self.goal_node  
        cur = self.goal_node  

        while cur != self.start_node:  
            self.path_vectors.append(cur) 
            cur = par[cur[0]][cur[1]][cur[2]]  
        self.path_vectors.append(cur)  

        self.path_vectors.reverse()  

    def optimal_path(self):
        """
        This function finds the polyline with minimum number of line segments, and
        draw those lines.
        """

        ptr = 0 
        while (
            ptr != len(self.path_vectors) - 1):  
            to = ptr - 1

            # iterate from last index of path_vectors to ptr
            for i in range(len(self.path_vectors) - 1, ptr, -1):
                # we check if the line segment from path_vector[ptr] to path_vectors[i]
                # collides with any obstacle or not.
                if isCollisionFreeEdge(self.obstacle, self.path_vectors[ptr], self.path_vectors[i]) == True:
                    to = i  # if it doesn't collide then we assign i to `to` and break
                    break

            # print(len(self.path_vectors), ptr, to)
            # here the line segment if drawn from path_vectors[ptr] to path_vectors[to]
            
            ptr = to  # since we have drawn a polyline upto `to`, ptr is updated to `to`


def connect(T, q):
    s = ADVANCED
    while s == ADVANCED:
        (s, _) = T.extend(q)

    return s


def merge_tree(start_node, goal_node,T,T1, T2, q):

    #T = RRT(start_node, goal_node, obstacle,gridX,gridY,gridZ)

    if T2.start_node == start_node:
        T1, T2 = T2, T1

    T1.goal_node = T2.goal_node = q

    T1.generate_path()
    T2.generate_path()

    T.path_vectors = T1.path_vectors
    assert T1.path_vectors[-1] == T2.path_vectors[-1]
    assert T1.path_vectors[-1] == q
    assert q == T2.path_vectors[-1]
    T.path_vectors += T2.path_vectors[::-1]

    return T


def RRT_connect_planner(start_node, goal_node, T, T1, T2):
    if (
        T1.active_node[T1.goal_node[0]][T1.goal_node[1]][T1.goal_node[2]]
        and T2.active_node[T2.goal_node[0]][T2.goal_node[1]][T2.goal_node[2]]
        and len(T.path_vectors) > 0
    ):
        return T

    for _ in range(max_iteration_per_planning):
        q_rand = T1.random_point()
        (status, q_new) = T1.extend(q_rand)
        if status != TRAPPED:
            if connect(T2, q_new) == REACHED:
                return merge_tree(start_node, goal_node,T,T1, T2, q_new)

        T1, T2 = T2, T1

    return None






########################## Start reading from here ##########################################



def drrt_connect(start_node, goal_node, ax, obstacle, animate, xbound, ybound, zbound):
    """
    Parameters
    start_node: robot arm's initial configuration.
    goal_node: robot arm's target configuration.
    delta: maximum length of any edge on the RRT.
    goal_point_probability: Probability of selecting goal node as a random node.
    xbound: X-axis boundary of the board.
    ybound: Y-axis boundary of the board.
    zbound: X-axis boundary of the board.
    """
    
    gridX,gridY,gridZ = xbound[1], ybound[1], zbound[1] #taking the maximum boundary
    start_node = tuple(start_node)
    goal_node = tuple(goal_node)
    T1 = RRT(start_node, goal_node, obstacle,gridX,gridY,gridZ)  # go to function definition for more details, also must read for starting
    T2 = RRT(goal_node, start_node, obstacle,gridX,gridY,gridZ)  # go to function definition for more details, also must read for starting
    T = RRT(start_node, goal_node, obstacle,gridX,gridY,gridZ)

    # analysis variables
    no_of_iteration = 0
    total_time = 0
    total_node = 0
    total_path_length = 0



    # st_time holds the current time.
    st_time = time.time()
    # This function invalidates all nodes for which if any obstacle intersects with that node or
    # any parent of that node.
    T1.trim_RRT()
    T2.trim_RRT()

    # this function grows RRT till a path is found from start to goal.
   
    T = RRT_connect_planner(start_node, goal_node,T, T1, T2)

    if not T:
        print("No path found: max iteration reached.")
        print("Planning Failed!!!")
        isSuccessful = 0
        Ptime = time.time() - st_time
        return None, Ptime, None, isSuccessful
    
    T.optimal_path()

    no_of_iteration += 1
    total_time += time.time() - st_time
    total_node += len(T1.node_list + T2.node_list)
    total_path_length += len(T.path_vectors)
    
    # time.time()-st_time results in time taken to grow_RRT, generate_path, and finding optimal_path.
    Ptime = time.time() - st_time
    
    #no_of_nodes in rrt tree
    no_of_nodes = len(T1.node_list + T2.node_list)
    
    #no_of_nodes in rrt optimal path
    no_of_nodes_path = len(T.path_vectors)
    
    print ('D-RRT nodes: ', no_of_nodes)
    print ('Planning time : %.2f seconds:' % (Ptime))
    
    if no_of_iteration % 50 == 0:
        print("""
            After {} iterations:
            Average Computational Time (s): {}
            Average Number of Nodes on Tree: {}
            Average Number of Nodes on Path: {}
        """.format(
            no_of_iteration,
            total_time / no_of_iteration,
            total_node / no_of_iteration,
            total_path_length / no_of_iteration
        ))
    path1 = []
    path2 = []
    for t in T1.edge_list:
        path1.append(list(t[0]))
    
    path2.append(list(T2.edge_list[0][0]))
    for t in T2.edge_list:
        path2.append(list(t[1]))
    
        
    path2.reverse()
    
    path = np.array(path1+path2)
    
    isSuccessful = 1
   
    return path, Ptime, no_of_nodes, isSuccessful



# Declare Constants
delta = 5  
goal_point_probability = 0  
radius_around_goal = 2
max_iteration_per_planning = 3000


###Algorithm Testing part
# xbound = (0,50)
# ybound= (0,50)
# zbound= (0,40)
# ax = 0
# animate = 0
# start_node = np.array([20, 20, 0])
# goal_node = np.array([40, 45, 28])
# obj = ObjecClass(start_node, goal_node)    
# obstacle = obj.give_objects()

#path, Ptime, no_of_nodes, isSuccessful= drrt_connect(start_node, goal_node, ax, obstacle, animate, xbound, ybound, zbound)
