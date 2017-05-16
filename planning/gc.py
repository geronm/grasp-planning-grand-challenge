from __future__ import division
from pomdp import POMDP
import numpy as np
from polytope import *
from beliefensemble import BeliefEnsemble
from multibeliefensemble import MultiBeliefEnsemble, LayeredBeliefEnsemble
from gc_pomdp import GrandChallengePOMDP
import random
import math
import time

class GrandChallengeGraspPlanInstance(object):
    """ OOP-style singleton for the grand challenge.

        This instance will contain all the API methods
        required by our planner. Its parameters will
        be specifically tuned to our particular grasp
        planning task. """

    FINGER_LENGTH = 0.16 # 16 cm
    FINGER_OFFSET = 0.03 # 3 cm
    BLOCK_WIDTH_UNIT = 0.10 # 10 cm
    BLOCK_THICKNESS_UNIT = 0.0475 # 19/4 cm
    LONG_BLOCK_LENGTH = 0.28 # 28 cm


    BLOCK_TRIPLE_POKE = 0
    BLOCK_LONG = 1
    BLOCK_HEX = 2
    def __init__(self):
        """ Initializes this instance for grand challenge-ing. """

        # First, make the block types
        self.canon_blocks = [None, None, None]
        self.canon_grasp_pose = [None, None, None]


        # The following parameters are common to all blocks.
        # They define the pose space and discretization.
        x_lim = (0.450 + self.FINGER_LENGTH, 0.875 + self.FINGER_LENGTH)
        y_lim = (-0.350, 0.350)
        theta_lim = (0, 2*np.pi)
        nx = 25
        ny = 25
        ntheta = 20
        
        # Triple-poke block. One square atop two squares.
        belief_ensembles = []

        FINGER_LENGTH = 0.16 # 16 cm
        FINGER_OFFSET = 0.03 # 3 cm
        BLOCK_WIDTH_UNIT = 0.10 # 10 cm
        BLOCK_THICKNESS_UNIT = 0.0475 # 19/4 cm
        LONG_BLOCK_LENGTH = 0.28 # 28 cm

        poly1 = Square(BLOCK_WIDTH_UNIT/2, BLOCK_WIDTH_UNIT/2)
        start = time.time()
        be_square_norot1 = BeliefEnsemble(poly1, x_lim, y_lim, theta_lim, nx, ny, ntheta)

        poly2 = poly1.translated_by_2D_vector( \
                        np.array([[BLOCK_WIDTH_UNIT],[0.0]]))
        be_square_norot2 = BeliefEnsemble(poly2, x_lim, y_lim, theta_lim, nx, ny, ntheta)

        poly3 = poly1.translated_by_2D_vector( \
                        np.array([[0.0],[BLOCK_WIDTH_UNIT]]))
        be_square_norot3 = BeliefEnsemble(poly3, x_lim, y_lim, theta_lim, nx, ny, ntheta)
        
        be_triple = MultiBeliefEnsemble([be_square_norot1, be_square_norot2, be_square_norot3])

        belief_ensembles.append(be_triple)
        
        poly1 = Square(BLOCK_WIDTH_UNIT/2, BLOCK_WIDTH_UNIT/2)
        start = time.time()
        be_square_norot1 = BeliefEnsemble(poly1, x_lim, y_lim, theta_lim, nx, ny, ntheta)
        be_topper = MultiBeliefEnsemble([be_square_norot1])

        belief_ensembles.append(be_topper)

        self.canon_blocks[self.BLOCK_TRIPLE_POKE] = \
                    LayeredBeliefEnsemble(belief_ensembles, (4*BLOCK_THICKNESS_UNIT,0.0))

        self.canon_grasp_pose[self.BLOCK_TRIPLE_POKE] = (0.,0.,0.)


        # Note the (x,y)-offset of fingertips
##        self.finger_positions = [np.array([[fx],[fy]]) for (fx, fy) in \
##                                 [(FINGER_LENGTH,FINGER_OFFSET),
##                                  (FINGER_LENGTH,-FINGER_OFFSET),
##                                  (-FINGER_LENGTH,0)]]
##        self.finger_positions = [np.array([[fx],[fy]]) for (fx, fy) in \
##                                         [(0,FINGER_OFFSET),
##                                          (0,-FINGER_OFFSET),
##                                          (-2*FINGER_LENGTH,0)]]
        self.finger_positions = [np.array([[fx],[fy]]) for (fx, fy) in \
                                         [(0,FINGER_OFFSET),
                                          (0,-FINGER_OFFSET)]]

    def reset(self, block_type):
        self.block_type = block_type
        self.gc_pomdp = GrandChallengePOMDP(self.canon_blocks[block_type],
                                            self.canon_grasp_pose[block_type],
                                            self.finger_positions)
        self.b_s = self.gc_pomdp.get_uniform_belief()

    def query_action(self):
        score, a = self.gc_pomdp.solve(self.b_s, depth=1)
        return a, score

    def step_update_bs(self, a, o):
        self.b_s = self.gc_pomdp.update_belief(self.b_s, a, o)
        return self.b_s

    def query_grasp_loc_confidence(self):
        # Confident if we can be within a sixth of a block width away
        MARGIN_METERS = float(self.BLOCK_WIDTH_UNIT) / 3.0
        MARGIN_RADIANS = np.pi # 0.1

        be = self.gc_pomdp.be
        
        x0 = be.x0
        x1 = be.x1
        nx = be.nx
        y0 = be.y0
        y1 = be.y1
        ny = be.ny
        theta0 = be.theta0
        theta1 = be.theta1
        ntheta = be.ntheta
        x_states_margin = int(math.ceil(MARGIN_METERS * nx / float(x1 - x0)))
        y_states_margin = int(math.ceil(MARGIN_METERS * ny / float(y1 - y0)))
        theta_states_margin = int(math.ceil(MARGIN_RADIANS * ntheta / float(theta1 - theta0)))

        # get guess
        (gx, gy, gtheta) = be.big_index_to_indices(np.argmax(self.b_s))

        b_s_rect = self.b_s.reshape((nx, ny, ntheta))

        # sum up belief to get probability that state is in that max
        # belief region
        print x_states_margin, y_states_margin, theta_states_margin
        prob_grasp = 0.0
        for ix in range(gx-x_states_margin, gx+x_states_margin+1):
            for iy in range(gy-y_states_margin, gy+y_states_margin+1):
                for itheta in range(gtheta-theta_states_margin, gtheta+theta_states_margin+1):
                    if ix >= 0 and ix < nx and \
                       iy >= 0 and iy < ny:
                        itheta_tru = max([itheta, itheta-ntheta, itheta+ntheta], \
                                         key=lambda x: float(0 <= x < ntheta))
                        prob_grasp += b_s_rect[ix,iy,itheta_tru]

        return (gx, gy, gtheta), prob_grasp

    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    gc = GrandChallengeGraspPlanInstance()
    gc.reset(gc.BLOCK_TRIPLE_POKE)
    #gc.gc_pomdp.be.render_belief_xy(plt, gc.b_s)

    print gc.gc_pomdp.get_possible_actions(gc.b_s)
    print gc.gc_pomdp.get_possible_actions(gc.b_s)
    print gc.gc_pomdp.get_possible_actions(gc.b_s)

    res = gc.query_grasp_loc_confidence()

    print (res)

    #action, score = gc.query_action()

##    print str(action)
##    print str(score)

    beliefs = []
    actions = []
    max_itr = 15
    a, o = None, None

    real_object_indices = (10, 10, 3)
    real_object_big_index = gc.gc_pomdp.be.indices_to_big_index(real_object_indices)
    print 'Real object pose: %s' % str(gc.gc_pomdp.be.indices_to_continuous_pose(real_object_indices))
    print gc.gc_pomdp.be.x0
    print gc.gc_pomdp.be.x1

    # gc.gc_pomdp.be.belief_ensembles[0].belief_ensembles[0].polytopes[real_object_big_index].render_polytope(plt, xmax=1.0)


    gc.gc_pomdp.be.render_belief_xy(plt, gc.b_s)
    
    beliefs.append(gc.b_s)
    while max_itr > 0 and not gc.gc_pomdp.is_terminal_belief(gc.b_s, a, o):
        gp = gc.gc_pomdp

        print 'Real object pose: %s' % str(gc.gc_pomdp.be.indices_to_continuous_pose(real_object_indices))

        print 'About to solve...'
        a, score = gc.query_action()
        print 'score: %s action: %s' % (str(score), str(a))
        
        actions.append(a)


        #  Ideal belief distribution:
        fingers_recentered = [a + f for f in gc.gc_pomdp.finger_positions]
        ideal_contact_given_s, ideal_iz_given_s = gc.gc_pomdp.be.get_ideal_obs(fingers_recentered)
        simulated_contact = ideal_contact_given_s[gc.gc_pomdp.be.indices_to_big_index(real_object_indices)]
        simulated_iz = ideal_iz_given_s[real_object_big_index][0]
        o = (tuple(simulated_contact), simulated_iz)
        print 'obs: %s' % str(o)
        gc.step_update_bs(a, o)

        #sim_obs = sim.grasp_action(a[0],a[1])
        #o = gp.sim_obs_to_pomdp_obs(sim_obs)
        beliefs.append(gc.b_s)
        print 'is_terminal_belief(b_s): %s' % str(gp.is_terminal_belief(gc.b_s,a,o))

        print 'np.argmax b_s: %s' % str(np.argmax(gc.b_s))
        print 'np.argmax b_s in continuous: %s' % str(
            gc.gc_pomdp.be.indices_to_continuous_pose(
              gc.gc_pomdp.be.big_index_to_indices(np.argmax(gc.b_s))))

        loc,confidence = gc.query_grasp_loc_confidence()
        print 'grasp and graspability: %s %s' % (str(loc), str(confidence))
        max_itr -= 1

        gc.gc_pomdp.be.render_belief_xy(plt, gc.b_s, fingers_recentered)

