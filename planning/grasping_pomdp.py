from pomdp import POMDP
from simulator import SimObject, Simulator
import numpy as np

class GraspingProblem(POMDP):
    # a = (loc, dir) source of gripper eg. (5, GraspingProblem.TOP)
    # o = (stop_loc, sensors) eg. ((4,5), [True,True,True])
    #
    # b_s = p_s , grid of probabilities for center of object
    LEFT = 0
    TOP = 1
    RIGHT = 2
    BOTTOM = 3
    BAD_OBS = (-1, [False,False,False])
    def __init__(self, simulator, sim_object, desired_grasp_loc=None):
        self.simulator = simulator
        self.sim_object = sim_object
        self.desired_grasp_loc = desired_grasp_loc
        self.possible_actions = []
        for x in range(1,self.simulator.width-1):
            self.possible_actions += [ (x, GraspingProblem.TOP) ]
        for y in range(1,self.simulator.height-1):
            self.possible_actions += [ (y, GraspingProblem.LEFT) ]
        for x in range(1,self.simulator.width-1):
            self.possible_actions += [ (x, GraspingProblem.BOTTOM) ]
        for y in range(1,self.simulator.height-1):
            self.possible_actions += [ (y, GraspingProblem.RIGHT) ]

    def sim_obs_to_pomdp_obs(self,sim_obs):
        stop_loc, direc, col_logic, loc = sim_obs
        return (stop_loc, col_logic)

    def obs_given_s_a(self, s, a):
        i, j = s
        # Put top-left corner of object at ith row, jth col
        top_left_x = j
        top_left_y = i
        if top_left_x >= 0 and top_left_y >= 0:
            # Run grasp action
            sim_obs = self.simulator.grasp_action(a[0], a[1], (top_left_x, top_left_y))
            # Check if the observation  = o
            return self.sim_obs_to_pomdp_obs(sim_obs)

        return GraspingProblem.BAD_OBS

    def prob_obs_given_s_a(self, s, a, o):
        if self.obs_given_s_a(s, a) == o:
            return 1.0
        return 0.0

    def prob_obs_given_bs_a(self, b_s, a, o):
        p_s = b_s

        # For a given observation o, get the probability of that observation
        # under the belief p_s.
        prob_o = 0
        for i in xrange(p_s.shape[0]):
            for j in xrange(p_s.shape[1]):
                if p_s[i, j] > 0:
                    prob_o += self.prob_obs_given_s_a((i, j), a, o) * p_s[i, j]

        return prob_o / np.sum(p_s)

    def update_belief(self, b_s, a, o):
        p_s = b_s

        # For each nonzero entry in p_s
        p_s_new = np.zeros(p_s.shape)
        for i in xrange(p_s.shape[0]):
            for j in xrange(p_s.shape[1]):
                if p_s[i, j] > 0:
                    # Update probability based on observation
                    p_s_new[i, j] = self.prob_obs_given_s_a((i, j), a, o) * p_s[i, j]

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
        p_s = b_s
        (loc, direction) = a
        (stop_loc, sensors) = o

        certain = np.sum( (p_s == 1) + 0.0 ) == 1

        # NOTE: ONLY APPLIES if we care about where we grasp the object!
        # make sure we know 100% where the object is
        if not certain:
            return False

        # make sure latest reach was a grasp
        s1, s2, s3 = sensors
        if not s2: # we can't grasp anything without s2
            return False
        
        # if self.desired_grasp_loc defined,
        # check for it
        if self.desired_grasp_loc is None:
            return True # success if can grasp anywhere!
        else:
            ij = np.argmax(b_s)
            assert ij < np.size(b_s)
            i = ij // len(b_s[0])
            j = ij % len(b_s[0])
            # Put top-left corner of object at ith row, jth col
            top_left_x = j
            top_left_y = i
            
            grasp_action_result = self.simulator.grasp_action(a[0],a[1], (top_left_x, top_left_y))
            grasp_loc = self.simulator.find_obj_rel_grasp_point(grasp_action_result, (top_left_x, top_left_y))
            print "grasped: {}. Wanted: {}".format(grasp_loc, self.desired_grasp_loc)
            if self.desired_grasp_loc == grasp_loc:
                print "Success."
                return True

        return False

    def heuristic(self, b_s):
        # heuristic in Battleship is min of
        #         number of rows with nonzero belief
        #          + number of cols with nonzero belief
        #  and
        #         num with nonzero belief.
        p_s = b_s

        p_s_nonzero = (p_s > 0) + 0.0

        row_sums = np.sum(p_s, 1)
        row_count_nonzero = np.sum( (row_sums > 0) + 0.0 )

        col_sums = np.sum(p_s, 0)
        col_count_nonzero = np.sum( (col_sums > 0) + 0.0 )

        return min( np.sum(p_s_nonzero) , np.sum(row_count_nonzero) + np.sum(col_count_nonzero) )

    def get_possible_actions(self, b_s):
        # this is a slight cheat, but for performance reasons,
        # we don't consider actions that offer literally zero
        # information, ie. we don't consider actions for which
        # the gripper is swept through a bunch of cells that have
        # probability 0 of containing the manipuland.
        p_s = b_s

        possible_actions = []

        p_cols = np.sum(p_s, 0)
        p_rows = np.sum(p_s, 1)

        obj_width, obj_height = self.sim_object.get_occ_arr().shape

        reach_cols = range(1, self.simulator.width-1)
        possible_actions.extend([(reach_col,GraspingProblem.TOP) for reach_col in reach_cols])
        possible_actions.extend([(reach_col,GraspingProblem.BOTTOM) for reach_col in reach_cols])

        reach_rows = range(1, self.simulator.height-1)
        possible_actions.extend([(reach_row,GraspingProblem.LEFT) for reach_row in reach_rows])
        possible_actions.extend([(reach_row,GraspingProblem.RIGHT) for reach_row in reach_rows])

        return possible_actions

    def get_possible_observations(self, b_s, a):
        p_s = b_s

        possible_o = []

        # For each nonzero entry in p_s, get the observation
        p_s_new = np.zeros(p_s.shape)
        for i in xrange(p_s.shape[0]):
            for j in xrange(p_s.shape[1]):
                if p_s[i, j] > 0:
                    possible_o.append(self.obs_given_s_a((i,j),a))

        return possible_o

    def get_uniform_belief(self):
        width = self.simulator.width - self.sim_object.width + 1
        height = self.simulator.height - self.sim_object.height + 1
        return np.ones((height, width)) / np.sum(np.ones((height, width)))
