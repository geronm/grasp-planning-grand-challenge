import IPython
from pomdp import POMDP
from grasping_pomdp import GraspingProblem
from simulator import SimObject, Simulator
import numpy as np

def test_ok():
    """ If execution gets to this point, print out a happy message """
    try:
        from IPython.display import display_html
        display_html("""<div class="alert alert-success">
        <strong>Tests passed!!</strong>
        </div>""", raw=True)
    except:
        print("Tests passed!!")

# ----- get_uniform_belief tests -----
def test_uniform_belief_1(student_get_uniform_belief):
    objarr =[[1]]
    obj = SimObject(2, 3, objarr)
    sim = Simulator(5, 5, obj)
    ans = np.array([[0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04],
                    [0.04, 0.04, 0.04, 0.04, 0.04]])
    assert(np.array_equal(ans, student_get_uniform_belief(sim)))

def test_uniform_belief_2(student_get_uniform_belief):
    objarr =[[0,1], [1,1], [1,1]]
    obj = SimObject(1, 3, objarr)
    sim = Simulator(7, 8, obj)
    ans = np.array([[0.02857142857142857, 0.02857142857142857, 0.02857142857142857, \
            0.02857142857142857, 0.02857142857142857],
            [0.02857142857142857, 0.02857142857142857, 0.02857142857142857, \
            0.02857142857142857, 0.02857142857142857],
            [0.02857142857142857, 0.02857142857142857, 0.02857142857142857, \
            0.02857142857142857, 0.02857142857142857],
            [0.02857142857142857, 0.02857142857142857, 0.02857142857142857, \
            0.02857142857142857, 0.02857142857142857],
            [0.02857142857142857, 0.02857142857142857, 0.02857142857142857, \
            0.02857142857142857, 0.02857142857142857],
            [0.02857142857142857, 0.02857142857142857, 0.02857142857142857, \
            0.02857142857142857, 0.02857142857142857],
            [0.02857142857142857, 0.02857142857142857, 0.02857142857142857, \
            0.02857142857142857, 0.02857142857142857]])
    assert(np.array_equal(ans, student_get_uniform_belief(sim)))

def test_uniform_belief_3(student_get_uniform_belief):
    objarr =[[0,1,0], [1,1,1], [1,1,1], [0,1,0]]
    obj = SimObject(4, 3, objarr)
    sim = Simulator(11, 9, obj)
    ans = np.array([[0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856, \
                    0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856],
                    [0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856, \
                    0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856],
                    [0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856, \
                    0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856],
                    [0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856, \
                    0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856],
                    [0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856, \
                    0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856],
                    [0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856, \
                    0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856],
                    [0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856, \
                    0.017857142857142856, 0.017857142857142856, 0.017857142857142856, 0.017857142857142856]])
    assert(np.array_equal(ans, student_get_uniform_belief(sim)))

# ----- obs_given_s_a tests ------
def test_obs_given_s_a_1(student_obs_given_s_a):
    # grasp from above, grasp object successfully
    objarr =[[1]]
    obj = SimObject(2, 3, objarr)
    sim = Simulator(5, 5, obj)
    ans = (3, [False, True, False])
    assert(ans == student_obs_given_s_a(sim, (3, 2), (2, 1)))

def test_obs_given_s_a_2(student_obs_given_s_a):
    # grasp from the left, miss object
    objarr =[[0,1], [1,1], [1,1]]
    obj = SimObject(1, 3, objarr)
    sim = Simulator(7, 8, obj)
    ans = (-1, [False, False, False])
    assert(ans == student_obs_given_s_a(sim, (3, 1), (6, 0)))

def test_obs_given_s_a_3(student_obs_given_s_a):
    # grasp from the right, hits but doesn't grasp
    objarr =[[0,1,0], [1,1,1], [1,1,1], [0,1,0]]
    obj = SimObject(4, 3, objarr)
    sim = Simulator(11, 9, obj)
    ans = (5, [False, False, True])
    assert(ans == student_obs_given_s_a(sim, (3, 2), (2, 2)))

# ----- update_belief tests ------
def update_belief_test_case_1(update_belief_to_test):
    objarr =[[0,1],[1,1],[0,1]]
    obj = SimObject(7,1,objarr)
    sim = Simulator(10, 10, obj)
    b_s_0 = []
    b_s_0 = np.array([[ 0.01388889,  0.01388889,  0.01388889,  0.01388889,  0.01388889,
         0.01388889,  0.01388889,  0.01388889],
       [ 0.01388889,  0.01388889,  0.01388889,  0.01388889,  0.01388889,
         0.01388889,  0.01388889,  0.01388889],
       [ 0.01388889,  0.01388889,  0.01388889,  0.01388889,  0.01388889,
         0.01388889,  0.01388889,  0.01388889],
       [ 0.01388889,  0.01388889,  0.01388889,  0.01388889,  0.01388889,
         0.01388889,  0.01388889,  0.01388889],
       [ 0.01388889,  0.01388889,  0.01388889,  0.01388889,  0.01388889,
         0.01388889,  0.01388889,  0.01388889],
       [ 0.01388889,  0.01388889,  0.01388889,  0.01388889,  0.01388889,
         0.01388889,  0.01388889,  0.01388889],
       [ 0.01388889,  0.01388889,  0.01388889,  0.01388889,  0.01388889,
         0.01388889,  0.01388889,  0.01388889],
       [ 0.01388889,  0.01388889,  0.01388889,  0.01388889,  0.01388889,
         0.01388889,  0.01388889,  0.01388889],
       [ 0.01388889,  0.01388889,  0.01388889,  0.01388889,  0.01388889,
         0.01388889,  0.01388889,  0.01388889]])

    a = (4, 1)
    o = (-1, [False, False, False])

    b_s_1_act = update_belief_to_test(sim, b_s_0, a, o)
    b_s_1_exp = np.array([[ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704]])

    assert np.sum(np.abs(b_s_1_act - b_s_1_exp) < 0.001) == np.size(b_s_1_exp)

def update_belief_test_case_2(update_belief_to_test):
    objarr =[[0,1],[1,1],[0,1]]
    obj = SimObject(7,1,objarr)
    sim = Simulator(10, 10, obj)
    b_s_0 = []
    b_s_0 = np.array([[ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704]])

    a = (7, 1)
    o = (0, [False, False, True])

    b_s_1_act = update_belief_to_test(sim, b_s_0, a, o)
    b_s_1_exp = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    assert np.sum(np.abs(b_s_1_act - b_s_1_exp) < 0.001) == np.size(b_s_1_exp)

def update_belief_test_case_3(update_belief_to_test):
    objarr =[[0,1],[1,1],[0,1]]
    obj = SimObject(7,1,objarr)
    sim = Simulator(10, 10, obj)
    b_s_0 = []
    b_s_0 = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    a = (8, 1)
    o = (1, [True, True, True])

    b_s_1_act = update_belief_to_test(sim, b_s_0, a, o)
    b_s_1_exp = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    assert np.sum(np.abs(b_s_1_act - b_s_1_exp) < 0.001) == np.size(b_s_1_exp)

# ----- is_terminal_belief tests -------
def is_terminal_belief_test_case_1(is_terminal_belief_to_test):
    objarr =[[0,1],[1,1],[1,1]]
    obj = SimObject(7,1,objarr)
    sim = Simulator(10, 10, obj)
    b_s = np.array([[ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.03703704,  0.03703704],
           [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.03703704,  0.03703704],
           [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.03703704,  0.03703704],
           [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.03703704,  0.03703704],
           [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.03703704,  0.03703704],
           [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.03703704,  0.03703704],
           [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.03703704,  0.03703704],
           [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.03703704,  0.03703704],
           [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
             0.        ,  0.03703704,  0.03703704]])

    a = (4, 3)
    o = (-1, [False, False, False])
    desired_loc = (0, 1)

    actual = is_terminal_belief_to_test(sim, b_s, a, o, desired_loc)
    expected = False

    assert actual == expected

def is_terminal_belief_test_case_2(is_terminal_belief_to_test):
    objarr =[[0,1],[1,1],[1,1]]
    obj = SimObject(7,1,objarr)
    sim = Simulator(10, 10, obj)
    b_s = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    a = (8, 1)
    o = (0, [False, False, True])
    desired_loc = (0, 1)

    actual = is_terminal_belief_to_test(sim, b_s, a, o, desired_loc)
    expected = False

    assert actual == expected

def is_terminal_belief_test_case_3(is_terminal_belief_to_test):
    objarr =[[0,1],[1,1],[1,1]]
    obj = SimObject(7,1,objarr)
    sim = Simulator(10, 10, obj)
    b_s = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    a = (2, 0)
    o = (7, [True, True, False])
    desired_loc = (0, 1)

    actual = is_terminal_belief_to_test(sim, b_s, a, o, desired_loc)
    expected = True

    assert actual == expected

# ----- cost tests ------
def cost_test_case_1(cost_to_test):
    objarr = np.transpose([[1]])
    obj = SimObject(1,1,objarr)
    sim = Simulator(10, 10, obj)
    b_s = np.array([[ 0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571,
         0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571],
       [ 0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571,
         0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571],
       [ 0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571,
         0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.        ,  0.        ,  0.        ,  0.        ],
       [ 0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571,
         0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571],
       [ 0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571,
         0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571],
       [ 0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571,
         0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571],
       [ 0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571,
         0.01428571,  0.01428571,  0.01428571,  0.01428571,  0.01428571]])
    actions = [(4, 0)]
    observations = [(-1, [False, False, False])]

    actual_cost = cost_to_test(sim, b_s, actions, observations, desired_loc=(0,0))

    exp_cost = heuristic(sim, b_s) + len(actions)

    assert actual_cost == exp_cost

def cost_test_case_2(cost_to_test):
    objarr = np.transpose([[1]])
    obj = SimObject(1,1,objarr)
    sim = Simulator(10, 10, obj)
    b_s = np.zeros((10,10))
    b_s[1][1] = 1.0 # certain object location, with successful grasp; cost will be length of actions
    actions = [(4, 0), (7, 0), (1, 0)]
    observations = [(-1, [False, False, False]),
                    (-1, [False, False, False]),
                    (1, [False, True, False])]

    actual_cost = cost_to_test(sim, b_s, actions, observations, desired_loc=(0,0))
    exp_cost = len(actions)

    assert actual_cost == exp_cost

def cost_test_case_3(cost_to_test):
    objarr =[[0,1],[1,1],[1,1]]
    obj = SimObject(7,1,objarr)
    sim = Simulator(10, 10, obj)

    b_s = np.array([[ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704],
       [ 0.03703704,  0.        ,  0.        ,  0.        ,  0.        ,
         0.        ,  0.03703704,  0.03703704]])
    actions = [(4, 3)]
    observations = [(-1, [False, False, False])]

    actual_cost = cost_to_test(sim, b_s, actions, observations, desired_loc=(0,0))
    exp_cost = heuristic(sim, b_s) + len(actions)

    assert actual_cost == exp_cost


# ------ additional functions ------

def heuristic(sim, b_s):
    # heuristic value of the belief
    p_s = b_s

    p_s_nonzero = (p_s > 0) + 0.0

    row_sums = np.sum(p_s, 1)
    row_count_nonzero = np.sum( (row_sums > 0) + 0.0 )

    col_sums = np.sum(p_s, 0)
    col_count_nonzero = np.sum( (col_sums > 0) + 0.0 )

    return min( np.sum(p_s_nonzero) , np.sum(row_count_nonzero) + np.sum(col_count_nonzero) )

def make_grasping_pomdp_class(get_uniform_belief_in,
                                obs_given_s_a_in,
                                update_belief_in,
                                is_terminal_belief_in,
                                cost_in):
    # Synthesize the GraspingPOMDP class
    class GraspingPOMDP(POMDP):
        # a = (loc, dir) source of gripper eg. (5, GraspingProblem.TOP)
        # o = (stop_loc, sensors) eg. ((4,5), [True,True,True])
        #
        # b_s = p_s , grid of probabilities for center of object
        LEFT = 0
        TOP = 1
        RIGHT = 2
        BOTTOM = 3
        BAD_OBS = (-1, [False,False,False])
        def __init__(self, simulator, desired_grasp_loc):
            self.simulator = simulator
            self.sim_object = simulator.obj
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
            return obs_given_s_a_in(self.simulator,s,a)

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
            return update_belief_in(self.simulator, b_s, a, o)

        def cost(self, b_s, actions, observations):
            return cost_in(self.simulator, b_s, actions, observations, self.desired_grasp_loc)

        def is_terminal_belief(self, b_s, a, o):
            return is_terminal_belief_in(self.simulator, b_s, a, o, self.desired_grasp_loc)

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

            reach_cols = range(1,self.simulator.width-1)
            possible_actions.extend([(reach_col,GraspingProblem.TOP) for reach_col in reach_cols])
            possible_actions.extend([(reach_col,GraspingProblem.BOTTOM) for reach_col in reach_cols])

            reach_rows = range(1,self.simulator.height-1)
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
            return get_uniform_belief_in(self.simulator)


    return GraspingPOMDP
