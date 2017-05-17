from pomdp import POMDP
import numpy as np
import scipy.stats
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
                 finger_positions,
                 accuracy = 0.8,
                 num_actions_max = 10,
                 seed = None):
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
        self.num_actions_max = num_actions_max
        self.accuracy = accuracy
        self.randgen = random.Random(seed)

    def prob_obs_given_bs_a(self, b_s, a, o):
        p_s = b_s
        fingers_center = a
        obs_contact, obs_iz = o

        fingers_recentered = [fingers_center + f for f in self.finger_positions]

        # For a given observation o, get the probability of that observation
        # under the belief p_s.
        total_prob_obs = np.sum(
            np.multiply(p_s,
                        self.be.prob_obs(fingers_recentered, (obs_contact, obs_iz), self.accuracy)))
    
        return total_prob_obs

    def update_belief(self, b_s, a, o):
        p_s = b_s
        fingers_center = a
        obs_contact, obs_iz = o

        fingers_recentered = [fingers_center + f for f in self.finger_positions]
        
        prob_obs_given_s = \
            np.multiply(p_s,
                        self.be.prob_obs(fingers_recentered, (obs_contact, obs_iz), self.accuracy))

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
        return scipy.stats.entropy(b_s)
        

##        if disbelief_threshold is None:
##            disbelief_threshold = 0.5 / (self.be.nx*self.be.ny)  # 1/(nx*ny) is prob of everything under uniform
##        
##        p_s = b_s
##
##        p_s = p_s.reshape((self.be.nx,self.be.ny,self.be.ntheta))
##        p_s = np.sum(p_s,2)
##        p_s = p_s.reshape((self.be.nx,self.be.ny))
##
##        p_s_nonzero = (p_s > disbelief_threshold) + 0.0
##
##        row_sums = np.sum(p_s, 1)
##        row_count_nonzero = np.sum( (row_sums > self.be.ny*disbelief_threshold) + 0.0 )
##
##        col_sums = np.sum(p_s, 0)
##        col_count_nonzero = np.sum( (col_sums > self.be.nx*disbelief_threshold) + 0.0 )
##
##        return min( np.sum(p_s_nonzero) , np.sum(row_count_nonzero) + np.sum(col_count_nonzero) )

    def get_possible_actions(self, b_s):
        p_s = b_s

        possible_actions = []

        # Consider a random set of actions sampled within the domain
        for i in range(self.num_actions_max):
            x = self.be.x0 + self.randgen.random()*(self.be.x1 - self.be.x0)*0.65
            y = self.be.y0 + self.randgen.random()*(self.be.y1 - self.be.y0)*0.9
            theta = 0
            possible_actions.append(np.array([[x],[y]]))

        return possible_actions

    def get_possible_observations(self, b_s, a):
        # This method will cheat, use max probability
        # observation as the only expected observation
        # (this is reasonable, since our observation model
        # is pretty dumb anyway: ideal with probability
        # accuracy, random with probability 1-accuracy).
        
        p_s = b_s
        fingers_center = a

        fingers_recentered = [fingers_center + f for f in self.finger_positions]

        # For each in s, get the observation under this a
        obs_ideals = self.be.get_ideal_obs(fingers_recentered)

        # Take only ideal observations
        contact_obs, contact_iz = obs_ideals

        # Encode observations as integers.
        # {self.nz} x {0,1}^|f|
        #
        # Formula:
        #  N = nz * |f|_2 + iz
        encoded_obs = np.transpose(contact_iz)
        for f in range(contact_obs.shape[1]):
            encoded_obs += (2**f) * np.transpose(contact_obs)[f] * self.be.nz

        # Uniquify! Turn S nonunique observations into 1-16 unique observations.
        encoded_obs = np.unique(encoded_obs)

        # Decode the unique observations
        decoded_obs = []
        for c in encoded_obs:
            iz = c % self.be.nz
            c = c // self.be.nz
            contacts = []
            for f in range(contact_obs.shape[1]):
                B = (2**f)
                contacts.append(int((c // B) % 2))
            decoded_obs.append((contacts,iz))

        reencoded_obs = []
        for o in decoded_obs:
            c = o[1]
            for f in range(contact_obs.shape[1]):
               c += (2**f) * o[0][f] * self.be.nz
            reencoded_obs.append(c)

        ret = decoded_obs
        
##        possible_observations = []
##        
##        for i in range(2**len(self.finger_positions)):
##            obs = []
##            k = i
##            for j in range(len(self.finger_positions)):
##                obs.insert(0, k % 2)
##                k //= 2
##
##            for k in range(len(self.be.belief_ensembles)):
##                possible_observations.append((obs, k))
##        
##        ret = [o for o in possible_observations if self.prob_obs_given_bs_a(b_s, a, o) > 0.0]
##        print ret
        return ret

    def get_uniform_belief(self):
        return self.be.get_uniform_belief()


if __name__ == '__main__':
    print 'I am GrandChallengePOMDP. Run gc.py to test me.'
    pass # TODO: Test this POMDP. Currently, run gc.py to test.
