from hw_interface import ArmControl
from planning.gc import GrandChallengeGraspPlanInstance
import rospy
import sys
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion, Twist, PoseStamped
from tf.transformations import quaternion_from_euler
from std_msgs.msg import Float64MultiArray, Header
from std_srvs.srv import Empty
from math import pi
import numpy as np

""" Leverages python hardware interface and POMDP planner to
   execute grasp planning. """
import matplotlib.pyplot as plt

Z_LIMIT = 0.602


## LOAD HARDWARE INTERFACE
rospy.init_node('arm_control_test', anonymous=True)
arm_control = ArmControl(live=True)
print "moving to calibration point..."
arm_control.calibrate_touch_sensor()
print "done. Calibration = {}".format(arm_control.touch_thresh)



## PREPARE PLANNERS
gc = GrandChallengeGraspPlanInstance()
gc.reset(gc.BLOCK_TRIPLE_POKE)

beliefs = []
actions = []
max_itr = 5
a, o = None, None

beliefs.append(gc.b_s)
while max_itr > 0 and not gc.gc_pomdp.is_terminal_belief(gc.b_s, a, o):

  ## DO SOME PLANNING, CHOOSE NEXT GRASP
  # gc.gc_pomdp.be.render_belief_xy(plt, np.array(gc.b_s))
  gp = gc.gc_pomdp
  print 'About to solve...'
  a, score = gc.query_action()
  print 'score: %s action: %s' % (str(score), str(a))
  actions.append(a)

  ## REACH IN HARDWARE
  LOC_NOMINAL = [0.735, 0.0, 0.8]
  ORIENT_NOMINAL = [-3.116, 0.047, -0.052]
  my_loc = list(LOC_NOMINAL)
  my_ori = list(ORIENT_NOMINAL)
  my_loc[0], my_loc[1] = a[0][0], a[1][0]
  assert 0.45 <= my_loc[0] <= 0.875
  assert -.35 <= my_loc[1] <= .35
  assert 0.6 <= my_loc[2] <= 1.0
  print 'REACHING FOR LOCATION ' + str(my_loc)
  probe_loc, probe_ori, probe_touch = arm_control.probe_at( \
          my_loc, my_ori,Z_LIMIT)
  
  touch = [(1 if i else 0) for i in probe_touch]
  iz = int( max(0, min(2, gc.gc_pomdp.be.z_continuous_to_index(probe_loc[2] - Z_LIMIT))))
  o = (touch, iz)

  ## (TODO!!!) REACH STRAIGHT BACK UP TO AVOID HITTING


  ## PROCESS UPDATE
  gc.step_update_bs(a, o)
  
  #sim_obs = sim.grasp_action(a[0],a[1])
  #o = gp.sim_obs_to_pomdp_obs(sim_obs)
  print 'obs: %s' % str(o)
  beliefs.append(gc.b_s)
  print 'is_terminal_belief(b_s): %s' % str(gp.is_terminal_belief(gc.b_s,a,o))

  loc,confidence = gc.query_grasp_loc_confidence()
  print 'grasp and graspability: %s %s' % (str(loc), str(confidence))
  max_itr -= 1


