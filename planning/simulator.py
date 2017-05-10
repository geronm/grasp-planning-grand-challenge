"""
simulator.py stores world and object information and allows simulating grasps.

"""


import numpy as np
from copy import deepcopy
from graphics import Grid

class SimObject:
    def __init__(self, x, y, occ_arr=None):
        r"""
        initializes a simulated object.

        Parameters
        ----------

        x : Object x location
        y : Object y location
        occ_array : An array encoding the object. The first dimension is Y and the second is X.

        """
        if occ_arr is not None:
            self.width = len(occ_arr)
            self.height = len(occ_arr[0])
            self.x = x
            self.y = y
            self.obj = deepcopy(occ_arr)
        else:
            sys.exit("Must pass in occ_arr");
        self.preprocess()

    def get_occ_arr(self):
        return np.array(self.obj)

    #Preprocessing to accelerate collision checking
    def preprocess(self):
        #create collision lookup table
        self.collide = [0] * 4
        self.collide[0] = [float("inf")] * self.height
        self.collide[2] = [-1] * self.height
        self.collide[1] = [float("inf")] * self.width
        self.collide[3] = [-1] * self.width

        #populate collision LUT
        for i in range(0, self.width):
            for j in range(0, self.height):
                if self.obj[i][j] == 1:
                    self.collide[0][j] = min(self.collide[0][j],i) #left collision
                    self.collide[2][j] = max(self.collide[2][j],i) #right collision
                    self.collide[1][i] = min(self.collide[1][i],j) #top collision
                    self.collide[3][i] = max(self.collide[3][i],j) #bottom collision

        for i in range(0,4):
            for j in range(0,len(self.collide[i])):
                if self.collide[i][j] == float("inf"):
                    self.collide[i][j] = -1

    def draw(self,grid):
        for i in range(0,len(self.obj)):
            for j in range(0,len(self.obj[0])):
                if self.obj[i][j] == 1:
                    grid.fill(self.x+i, self.y+j,0.5,'b',1)


class Simulator:
    def __init__(self, width, height, obj):
        r"""
        initializes the sim world.

        Parameters
        ----------

        width : World width
        height : World height
        object : an instance of SimObject

        """
        self.width = width
        self.height = height

        #Ensure valid object
        assert obj.x+obj.width <= width and obj.x >= 0, "object out of bounds in x"
        assert obj.y+obj.height <= height and obj.y >= 0, "object out of bounds in y"
        #store object
        self.obj = obj


    #Do not use externally.  grasp_action() is the public interface.
    def collision_check(self, loc, direc, obj_loc=None):
        if obj_loc == None:
            obj_x,obj_y = self.obj.x,self.obj.y
        else:
            obj_x, obj_y = obj_loc

        assert direc >= 0 and direc < 4, "Invalid direction"
        assert direc % 2 == 1 or loc > 0 and loc < self.height-1, "Grasp location out of bounds"
        assert direc % 2 == 0 or loc > 0 and loc < self.width-1, "Grasp location out of bounds"

        #print "=========="
        #print "obj location: {},{}".format(obj_x, obj_y)
        #print "gripper direction: {}. Gripper location: {}".format(direc,loc)
        #Check if gripper out of bounding box
        if direc == 0 or direc == 2: #horiz
            #print "horizontal grab case"
            if loc+1 < obj_y or loc > obj_y+self.obj.height:
                return (-1, direc, [False, False, False], loc)
            loc_ = loc - obj_y
            offset = obj_x
        else: #vert
            if loc+1 < obj_x or loc > obj_x+self.obj.width:
                return (-1, direc, [False, False, False], loc)
            loc_ = loc - obj_x
            offset = obj_y

        #print "within bounding box"
        #Check collisions... gripper within object bounding box.
        idx = (loc_-1, loc_, loc_+1)
        l = len(self.obj.collide[direc])

        if direc >= 2:
            col = [-float("inf"), -float("inf"), -float("inf")]
        else:
            col = [float("inf"), float("inf"), float("inf")]
        #check first row
        if idx[0] >= 0 and idx[0] < l:
            col[0] = self.obj.collide[direc][idx[0]]
            #print col[0]
        #check second row
        if idx[1] >= 0 and idx[1] < l:
            col[1] = self.obj.collide[direc][idx[1]]
        #check third row
        if idx[2] >= 0 and idx[2] < l:
            col[2] = self.obj.collide[direc][idx[2]]

        ofb=-1
        if direc >= 2:
            _stop_loc = max(max(col[0], col[1]-1), col[2])+1
            ofb=1
        else:
            _stop_loc = min(min(col[0], col[1]+1), col[2])-1

        col_world = np.add(col,offset)
        #print "col = {}".format(col_world)
        #print "stopped at: ",_stop_loc+offset

        col_logic = [False, False, False]
        if col[0]+ofb == _stop_loc:
            col_logic[0] = True
        if col[1] == _stop_loc:
            col_logic[1] = True
        if col[2]+ofb == _stop_loc:
            col_logic[2] = True

        stop_loc = _stop_loc + offset

        return (stop_loc, direc, col_logic, loc)

    def find_obj_rel_grasp_point(self, grasp_action_result, obj_loc=None):
        r""" Returns x,y in object reference frame of where the gripper stopped.

        Parameters
        ----------

        grasp_action_result : the return tuple from grasp_action()
        obj_loc : override the true object location.
                  format: (x,y) tuple

        Returns
        ----------
        Tuple of form (x,y)

        """
        if obj_loc == None:
            obj_x,obj_y = self.obj.x,self.obj.y
        else:
            obj_x, obj_y = obj_loc

        stop_loc, direc, col_logic, loc = grasp_action_result

        if direc == 0 or direc == 2: #horiz
            offset = obj_x
            x = stop_loc - obj_x
            y = loc - obj_y
        else: #vert
            x = loc - obj_x
            y = stop_loc - obj_y

        return (x,y)

    def grasp_action(self, loc, gripper_dir, obj_loc=None, animate=False):
        #Direction:
        #Return Value: (gripper stop location, dir, colision array, loc=gripper position)

        r"""
        initializes the sim world.

        Parameters
        ----------

        loc : Where should the gripper grab from.
              This is the lateral axis in relation to chosen direction
        gripper_dir : 0: left, 1: top, 2: right, 3: bottom
        obj_loc : override the true object location.
                  format: (x,y) tuple
        animate : Add this grasp action to the animation queue
                  ***TODO: implement this

        Returns
        ----------

        Tuple of form (stop_loc, direc, col_logic, loc) where:

        stop_loc : the cell the gripper stopped in (axis determined by direction)
        direc : the direction the gripper moved in (0: left, 1: top, 2: right, 3: bottom)
        col_logic : a 3-element boolean array describing which parts of the gripper
                    made contact with an object. element 1 is the center, and
                    the other two elements are the two sides.
        loc : Where the gripper grabbed from.

        """
        result = self.collision_check(loc,gripper_dir, obj_loc=obj_loc)
        # Animation related stuff here
        return result

    def draw_grasp(self, grid, grasp_action_result):
        grid.clear()
        self.obj.draw(grid)

        if grasp_action_result[0] == -1:
            #print  "No gripper contact"
            return
        #horizontal case

        c = 'r' if grasp_action_result[2][1] == False else 'g'
        if grasp_action_result[1] == 0 or grasp_action_result[1] == 2:
            #print "drawing horizontal gripper"
            x = grasp_action_result[0]
            y = grasp_action_result[3]
            grid.fill(x, y+1, 1, c, 1)
            grid.fill(x, y-1, 1, c, 1)
            i = -1 if grasp_action_result[1] == 0 else 1
            grid.fill(x+i, y+1, 1, c, 1)
            grid.fill(x+i, y, 1, c, 1)
            grid.fill(x+i, y-1, 1, c, 1)
        else:
            #print "drawing vertical gripper"
            x = grasp_action_result[3]
            y = grasp_action_result[0]
            grid.fill(x+1, y, 1, c, 1)
            grid.fill(x-1, y, 1, c, 1)
            i = -1 if grasp_action_result[1] == 1 else 1
            grid.fill(x+1, y+i, 1, c, 1)
            grid.fill(x, y+i, 1, c, 1)
            grid.fill(x-1, y+i, 1, c, 1)




if __name__ == "__main__":
    #Object in array form
    obj1 =             [[0,0,1,0,0],
                     [1,1,1,1,1],
                     [1,1,1,1,1]]

    obj2 =             [[0,0,0,0,0],
                     [0,1,1,1,0],
                     [0,0,0,1,0],
                     [0,0,0,0,0]]
    print "----"
    check_col_proc = SimObject(0, 0, obj2)
    print check_col_proc.collide
    print "----"

    obj_simple = [[1]]

    #Create the object object <_<
    test_obj = SimObject(1, 1, obj_simple)

    #Create sim instance
    sim = Simulator(5, 5, test_obj)

    print "Collision tables:"
    print test_obj.collide

    obj_arr =         [[1,2,3,4,5],
                     [6,7,8,9,10],
                     [11,12,13,14,15]]

    print "Testing Grabs.."
    print "Grabbing to left at location 3. Results:"
    print sim.grasp_action(1,3)
    print sim.grasp_action(2,3)
    print sim.grasp_action(3,3)
