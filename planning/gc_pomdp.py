from pomdp import POMDP
import numpy as np
from polytope import *
from beliefensemble import BeliefEnsemble
from multibeliefensemble import MultiBeliefEnsemble, LayeredBeliefEnsemble
import random

class GrandChallengePOMDP(POMDP):
    """ POMDP for the grand challenge.

        Capable of determining continuous-space reach
        actions that are heuristically optimal via
        lookahead search, and updating belief based
        on that.

        beliefs - 2D Numpy array representing the
        belief distribution over discretized
        pose <x,y,theta> (exact parameters determined
        by the layered_belief_ensemble passed in).

        actions - continuous 2D Numpy vector giving the
        (x,y) location of the fingers as they are pressed
        onto the table. Pose is assumed vertical

        observations - contact results from the fingers
        (as a tuple of 1s and 0s (1=contact)) and the iz
        value for the plane at which the hand stopped.
        eg: obs = ((1,0,1), 2)

        """
    def __init__(self, layered_belief_ensemble,
                        desired_grasp_pose,
                        finger_positions):
        """ Makes a POMDP instance

            Parameters:
            ---
            layered_belief_ensemble: model for object pose-space
            
            desired_grasp_pose: object-relative grasp pose for hand.
            Should be a tuple (x, y, theta).

            finger_positions: hand-relative finger positions, as
            a python list of 2D Numpy vectors. """
        self.be = layered_belief_ensemble
        self.desired_grasp_pose = desired_grasp_pose
        self.finger_positions = finger_positions

    def prob_obs_given_bs_a(self, b_s, a, o):
        p_s = b_s
        fingers_center = a
        obs_contact, iz = o

        fingers_recentered = [fingers_center + f for f in self.finger_positions]

        # Delta_dist over iz:
        b_z = np.zeros((len(self.be.belief_ensembles),1))
        b_z[iz][0] = 1.0

        # For a given observation o, get the probability of that observation
        # under the belief p_s.
        total_prob_obs = np.sum(
            np.multiply(p_s,
                        self.be.prob_obs(fingers_recentered, obs_contact, b_z)))

        return total_prob_obs

    def update_belief(self, b_s, a, o):
        p_s = b_s
        fingers_center = a
        obs_contact, iz = o

        fingers_recentered = [fingers_center + f for f in self.finger_positions]

        # Delta_dist over iz:
        b_z = np.zeros((len(self.be.belief_ensembles),1))
        b_z[iz][0] = 1.0
        
        prob_obs_given_s = \
            np.multiply(p_s,
                        self.be.prob_obs(fingers_recentered, obs_contact, b_z))

        p_s_new = prob_obs_given_s

        p_s_new_total = np.sum(p_s_new)

        assert p_s_new_total != 0.0, "Error: Performed Bayesian update on an observation of 0 probability. a = %s, o = %s, belief:\n%s\n%s" % (str(a),str(o),str(b_s), str(self.prob_obs_given_bs_a(b_s,a,o)))

        p_s_new = p_s_new / p_s_new_total

        return p_s_new

    def cost(self, b_s, actions, observations):
        cost = 0

        # cost-so-far; number of actions taken
        cost += len(actions)

        # if terminal belief, then cost-so-far is the cost
        if self.is_terminal_belief(b_s, actions[-1], observations[-1]):
            return cost

        # if not terminal belief, need to include heuristic cost
        cost += self.heuristic(b_s)

        return cost

    def is_terminal_belief(self, b_s, a, o):
        if a is None or o is None:
            return False   # Need to have grasped for this to be a terminal belief

        # There are no terminal beliefs. An outer loop should
        # assess grasp confidence to handle that decision.
        # We're flying the the seat of our heuristic.
        return False

    def heuristic(self, b_s, disbelief_threshold=None):
        # heuristic in Battleship is min of
        #         number of rows with nonzero belief
        #          + number of cols with nonzero belief
        #  and
        #         num with nonzero belief.

        if disbelief_threshold is None:
            disbelief_threshold = 0.5 / (self.be.nx*self.be.ny)  # 1/(nx*ny) is prob of everything under uniform
        
        p_s = b_s

        p_s = p_s.reshape((self.be.nx,self.be.ny,self.be.ntheta))
        p_s = np.sum(p_s,2)
        p_s = p_s.reshape((self.be.nx,self.be.ny))

        p_s_nonzero = (p_s > disbelief_threshold) + 0.0

        row_sums = np.sum(p_s, 1)
        row_count_nonzero = np.sum( (row_sums > self.be.ny*disbelief_threshold) + 0.0 )

        col_sums = np.sum(p_s, 0)
        col_count_nonzero = np.sum( (col_sums > self.be.nx*disbelief_threshold) + 0.0 )

        return min( np.sum(p_s_nonzero) , np.sum(row_count_nonzero) + np.sum(col_count_nonzero) )

    def get_possible_actions(self, b_s, N=4, seed=None):
        p_s = b_s

        possible_actions = []

        # Consider a random set of actions sampled within the domain
        random.seed(seed)
        for i in range(N):
            x = self.be.x0 + random.random()*(self.be.x1 - self.be.x0)
            y = self.be.y0 + random.random()*(self.be.y1 - self.be.y0)
            theta = 0
            possible_actions.append(np.array([[x],[y]]))

        return possible_actions

    def get_possible_observations(self, b_s, a):
        p_s = b_s
        fingers_center = a

##        fingers_recentered = [fingers_center + f for f in self.finger_positions]
##
##        # For each in s, get the observation under this a
##        obs_ideals = self.be.get_ideal_obs(fingers_recentered, obs_contact)
##
##        # Zero-out observations corresponding to probability zero
##        obs_prob_scaled = np.multiply(np.kron(p_s,np.ones((len(self.finger_positions),1))),obs_ideals)
##
##        # Test these obs_ideals for whether they contain each of the possible observations
##        

        possible_observations = []
        
        for i in range(2**len(self.finger_positions)):
            obs = []
            k = i
            for j in range(len(self.finger_positions)):
                obs.insert(0, [k % 2])
                k //= 2

            for k in range(len(self.be.belief_ensembles)):
                possible_observations.append((np.array(obs), k))
        
        return [o for o in possible_observations if self.prob_obs_given_bs_a(b_s, a, o) > 0.0]

    def get_uniform_belief(self):
        return self.be.get_uniform_belief()


if __name__ == '__main__':
    print 'I am GrandChallengePOMDP'
    pass # TODO: Test this POMDP
