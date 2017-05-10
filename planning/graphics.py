import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import matplotlib.ticker as ticker
from IPython.display import HTML
import math 

# Set the default video engine to html5
rc('animation', html='html5')

from enum import Enum

class Direction(Enum):
    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3

class Gripper:
    # Directions: LEFT, TOP, RIGHT, BOTTOM
    # i: which cell the arm starts movin
    
    def __init__(self, i, direction, color, w, h):
        # Discrete points that compose the arm
        points = []
        
        if direction == 0 or direction == 2:
            # print "drawing horizontal gripper"
            x = 1 if direction == 0 else (w - 2)
            y = i 
            points.append(plt.Rectangle((x, y+1), 1, 1, fc=color))
            points.append(plt.Rectangle((x, y-1), 1, 1, fc=color))
            
            i = -1 if direction == 0 else 1
            points.append(plt.Rectangle((x+i, y+1), 1, 1, fc=color))
            points.append(plt.Rectangle((x+i, y), 1, 1, fc=color))
            points.append(plt.Rectangle((x+i, y-1), 1, 1, fc=color))
            
        else:
            # print "drawing vertical gripper"
            x = i
            y = 1 if direction == 1 else (h - 2)
            points.append(plt.Rectangle((x+1, y), 1, 1, fc=color))
            points.append(plt.Rectangle((x-1, y), 1, 1, fc=color))
            
            i = -1 if direction == 1 else 1
            points.append(plt.Rectangle((x+i, y+i), 1, 1, fc=color)) # Error y is shifted down: Tell REO
            points.append(plt.Rectangle((x, y+i), 1, 1, fc=color))
            points.append(plt.Rectangle((x-i, y+i), 1, 1, fc=color))
        
        self.points = points 
        self.x = x
        self.y = y
    
    # Takes plt shape and laods the objects
    def load_gripper(self, gca):
        for p in self.points:
            p.zorder = 1000
            gca.add_patch(p)
    
    # Clears this arm from the board 
    def clear(self):
        for p in self.points:
            # print("Point removed", p)
            p.set_visible(False)
            p.remove()
            
            
    
    # Generates a coordinate travel path from 
    def get_frames(self, x, y, duration, time_per_frame):
        # print("Self x, y", self.x, self.y)
        paths = []
        num_frames = int(duration/time_per_frame)
        x_inc = (x - self.x)*1.0 / num_frames
        y_inc = (y - self.y)*1.0 / num_frames
        for i in xrange(num_frames + 1):
            paths.append((self.x + x_inc*i, self.y + y_inc*i))
        return paths
    
    # Moves the gripper to specified coordinate
    # Center coordinate
    def move(self, x, y):
        x_inc = (x - self.x)
        y_inc = (y - self.y)
        for p in self.points:
            p_x, p_y = p.xy
            p.xy = (p_x + x_inc, p_y + y_inc)
        self.x = x
        self.y = y
         

class Grid(object):
    def __init__(self, w, h):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], lw=2)
        self.w = w
        self.h = h
        
        self.gca = plt.gca()

        # Represents the gripper on the board
        self.gripper = None
        
        # Holds shapes that is associated with object
        self.obstacles = []
        
        # Number of frames currently needed to be presented
        self.scenes = [] # holds the objects for each scene
        self.cur_scene = None
        self.num_frames = 0
        
        self.time_per_frame = 1e-1 # defined in seconds 
        
        # self.arm.body.set_clip_on(False) # The arm is always on board
        
    
    
    # initialization function: plot the background of each frame
    def init(self):
        self.ax.set_xlim(( 0, self.w))
        self.ax.set_ylim((self.h, 0))
        loc = ticker.MultipleLocator(1)
        self.ax.xaxis.set_major_locator(loc)
        self.ax.yaxis.set_major_locator(loc)
        return (self.line,)
    
    
    ####################################
    # Methods to render on the board
    ####################################
    
    # Visualizing single prob 
    # generates frames and add it to the 
    # sim - simulator 
    def visualize_grasp_sequence(self, sim, belief, action):
        r"""Visualizes current string.
           
            Parameters
            ----------
            sim : simulator
                Array_like means all those objects -- lists, nested lists, etc. --
                that can be converted to an array.  We can also refer to
                variables like `var1`.
            belief : int
                The type above can either refer to an actual Python type
                (e.g. ``int``), or describe the type of the variable in more
                detail, e.g. ``(N,) ndarray`` or ``array_like``.
            action : (coor, direction)
                Choices in brackets, default first when optional.
                
            Returns
            -------
            type
                Explanation of anonymous return value of type ``type``.
            describe : type
                Explanation of return value named `describe`.
            out : type
                Explanation of `out`.
        """

        
        # Create arm
        i, direction = action
        gripper = Gripper(i, direction, 'r', self.w, self.h)
        (stop_loc, direc, col_logic, loc) = sim.grasp_action(i, direction)
        
        if sum(col_logic) > 0:
            if direction == 0 or direction == 2:
                #print "drawing horizontal gripper"
                final_x = stop_loc
                final_y = loc
            else:
                final_x = loc
                final_y = stop_loc
        else:
            # No grip was found 
            # Animate all the way through
            if direction == 0 or direction == 2:
                x = self.w-1 if direction == 0 else 0
                final_x = x
                final_y = i
            else:
                y = self.h-1 if direction == 1 else 0
                final_x = i
                final_y = y
            
        # self.gripper.clear() # Removes all previous     
        # self.gripper.load_gripper(self.gca)
        #print("Final_x_y", final_x, final_y)
        paths = gripper.get_frames(final_x, final_y, 5.0, self.time_per_frame)
        #print("Total path", paths)
        
        interval = [self.num_frames]
        self.num_frames += len(paths)
        interval.append(self.num_frames)
        
        # (index, (scene_info))
        scene = (len(self.scenes), (interval, gripper, paths, (belief, sim.obj)))
        self.scenes.append(scene)
        
        
    
    # Visulaizing multiple probs 
    def visualize_grasp_sequences(self, sim, beliefs, actions):
        for i in range(len(actions)):
            self.visualize_grasp_sequence(sim, beliefs[i], actions[i])
        
    def add_keyframe(time,x,y,theta):
        # add a keyframe at t=time to move the gripper to x,y with orientation theta.
        # We can assume the keyframes are added in order such that time is strictly increasing.
        pass
    
    ####################################
    # Methods to render on the board
    ####################################
    
    #fill cell 
    def fill(self, x,y,alpha, color, zorder):
        #fill cell x,y with color 
        rect = plt.Rectangle((x, y), 1, 1, fc=color)
        rect.set_alpha(alpha)
        # rect.zorder = zorder 
        plt.gca().add_patch(rect)
        self.obstacles.append(rect)

    def clear(self):
        #clear the canvas (ie: undo all fill commands)
        for obs in self.obstacles:
            obs.remove()
            obs.set_visible(False)
        self.obstacles = []
    
    
    def render_obj(self, obj, x, y, alpha, color, zorder):
        for i in range(0,len(obj.obj)):
            for j in range(0,len(obj.obj[0])):
                if obj.obj[i][j] == 1:
                    self.fill(x+i, y+j, alpha, color, zorder)
                    
    # Rendering the beilief on the board 
    def render_belief(self, belief, obj):
        # Rendering the actual object
        # print("Rendering object")
        self.render_obj(obj, obj.x, obj.y, 0.8, 'g', 1)
        
        # print("Placing rendered cells")
        # Placing objects on the belief state
        for i in range(len(belief)):
            for j in range(len(belief[0])):
                alpha = belief[i][j]
                if alpha != 0:
                    self.render_obj(obj, j, i, min(0.9, 0.3+ alpha*0.5), 'c', 1)
        
        
    def load_scene(self):
        (index, (interval, gripper, paths, belief)) = self.cur_scene
        self.clear()
        # Load new scene objects here
        gripper.load_gripper(self.gca)
        # actual belief and object
        self.render_belief(belief[0], belief[1])
        # print("Loading new scene", index)
        
    
    def step(self):
        # Change to next scene
        if self.cur_scene == None:
            try:
                self.cur_scene = self.scenes[0]
            except IndexError:
                self.cur_scene = None # keep looping at this state 
        else:
            (index, (interval, gripper, paths, belief)) = self.cur_scene
            # Loading next scene, remove all old state remnants here
            gripper.clear()
            # print("Removing old scene", index)
            self.cur_scene = self.scenes[index + 1]
            
        self.load_scene()


    def plot_step(self, i):
        if self.cur_scene == None:
            self.step()
        else:
            (index, (interval, gripper, paths, belief)) = self.cur_scene
            # Within the current scene frames
            if interval[0] <= i and interval[1] > i:
                x, y = paths[i-interval[0]]
                gripper.move(x, y)
            # load next scene
            else:
                self.step()
                
        return (self.line,)

    def play(self):
        #anim = animation.FuncAnimation(self.fig, self.plot_step, init_func=self.init, frames=100, interval=100) 
        self.cur_scene = None # make sure animation starts from beginning
        anim = animation.FuncAnimation(self.fig, self.plot_step, init_func=self.init, frames=self.num_frames, interval=self.time_per_frame*1000.0)
        
        #anim.writer = animation.AVConvWriter()

        #print help(animation.FuncAnimation)
        #print help(anim.to_html5_video)
        
        self.anim = anim
        plt.grid(True)
        html = HTML(self.anim.to_html5_video())
        plt.close() # make sure there is not more than one plot at a time
        
        return html
        # plt.show()


