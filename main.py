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
run_live = False
do_viz = True

## LOAD HARDWARE INTERFACE
rospy.init_node('arm_control_test', anonymous=True)
arm_control = ArmControl(live=run_live)
print "moving to calibration point..."
arm_control.calibrate_touch_sensor()
print "done. Calibration = {}".format(arm_control.touch_thresh)

## OPEN HAND
print "opening hand.."
arm_control.open_hand()


## PREPARE PLANNERS
gc = GrandChallengeGraspPlanInstance()
gc.reset(gc.BLOCK_TRIPLE_POKE)

if do_viz:
  gc.gc_pomdp.be.render_belief_xy(plt, np.array(gc.b_s))

beliefs = []
actions = []
max_itr = 2
a, o = None, None

beliefs.append(gc.b_s)
confidence = 0.01
while confidence < 0.6 and max_itr > 0 and not gc.gc_pomdp.is_terminal_belief(gc.b_s, a, o):

  ## DO SOME PLANNING, CHOOSE NEXT GRASP
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
  my_loc[0], my_loc[1] = max(a[0][0] - gc.FINGER_LENGTH, .45)   ,    a[1][0]
  assert 0.45 <= my_loc[0] <= 0.875, str(my_loc[0])
  assert -.35 <= my_loc[1] <= .35, str(my_loc[1])
  assert 0.6 <= my_loc[2] <= 1.0, str(my_loc[2])
  print 'REACHING FOR LOCATION ' + str(my_loc)
  probe_loc, probe_ori, probe_touch = arm_control.probe_at( \
          my_loc, my_ori,Z_LIMIT)
  
  ## PROCESS UPDATE
  touch = [(1 if i else 0) for i in probe_touch[:2]] # ONLY FIRST TWO FINGERS, AND MUST SHIFT
  iz = int( max(0, min(2, gc.gc_pomdp.be.z_continuous_to_index(probe_loc[2] - Z_LIMIT))))
  o = (touch, iz)
  gc.step_update_bs(a, o)
  
  #sim_obs = sim.grasp_action(a[0],a[1])
  #o = gp.sim_obs_to_pomdp_obs(sim_obs)
  print 'obs: %s' % str(o)
  beliefs.append(gc.b_s)
  print 'is_terminal_belief(b_s): %s' % str(gp.is_terminal_belief(gc.b_s,a,o))

  print 'grasp guess b_s: %s' % str(np.argmax(gc.b_s))
  print 'grasp guess b_s in continuous: %s' % str(
      gc.gc_pomdp.be.indices_to_continuous_pose(
        gc.gc_pomdp.be.big_index_to_indices(np.argmax(gc.b_s))))

  loc,confidence = gc.query_grasp_loc_confidence()
  print 'grasp and graspability: %s %s' % (str(loc), str(confidence))
  max_itr -= 1

  fingers_recentered = [a + f for f in gc.gc_pomdp.finger_positions]
  #if do_viz:
  #  gc.gc_pomdp.be.render_belief_xy(plt, np.array(gc.b_s), fingers_recentered)


## GRASP AT GUESSED LOCATION:
guessed_obj_pose = gc.gc_pomdp.be.indices_to_continuous_pose(
        gc.gc_pomdp.be.big_index_to_indices(np.argmax(gc.b_s)))
ox, oy, otheta = guessed_obj_pose
print "Grasping object at " + str((ox, oy, otheta))
# CONVERT TO A HAND POSE
hx = ox # reach palm to where object really is
hy = oy
hyaw = otheta
grasp_pre_loc = [hx, hy, LOC_NOMINAL[2] + 0.10]
grasp_pre_ori = [ORIENT_NOMINAL[0], ORIENT_NOMINAL[1], hyaw]

grasp_dur_loc = [hx, hy, Z_LIMIT + gc.gc_pomdp.be.z0]
grasp_dur_ori = [ORIENT_NOMINAL[0], ORIENT_NOMINAL[1], hyaw]

print "grasp_pre_loc: " + str(grasp_pre_loc)
print "grasp_pre_ori: " + str(grasp_pre_ori)
print "grasp_dur_loc: " + str(grasp_dur_loc)
print "grasp_dur_ori: " + str(grasp_dur_ori)

print "Reaching..."
arm_control.goto_pt(tuple(grasp_pre_loc), tuple(grasp_pre_ori)) #REACH UP
print "Moving down..."
arm_control.goto_pt(tuple(grasp_dur_loc), tuple(grasp_dur_ori)) #REACH DOWN
print "Closing hand..."
arm_control.close_hand() #CLOSE HAND
print "Reaching back up..."
arm_control.goto_pt(tuple(grasp_pre_loc), tuple(grasp_pre_ori)) #REACH UP



