#! /usr/bin/env python

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
from mpex_activity_manager import MpexClient

""" Leverages python hardware interface and POMDP planner to
   execute grasp planning. """
import matplotlib.pyplot as plt

def pick_up(lb,ub,obj_type):
  """ Pick up object using tactile exploration.

      Parameters
      ---
      lb: temporal lb

      ub: temporal ub

      obj_type: integer. 0 for triple-poke object,
      1 for long object. """
  ## OPEN HAND
  print "opening hand.."
  arm_control.open_hand()

  ## TEST MOVE
  # grasp_pre_loc = tuple([0.78, -0.3, arm_control.Z_GRASP[0]])
  # ORIENT_GRASP = (3.14, 0, 0)
  # ORIENT_NOMINAL = (0, 1.57, 0)
  # my_ori = list([ORIENT_NOMINAL[0], ORIENT_NOMINAL[1], ORIENT_NOMINAL[2]])
  # arm_control.goto_pt(tuple(grasp_pre_loc), tuple(grasp_pre_ori))

  ## PREPARE PLANNERS
  gc = GrandChallengeGraspPlanInstance()
  #gc.reset(gc.BLOCK_TRIPLE_POKE) == 0
  #gc.reset(gc.BLOCK_LONG) == 1
  gc.reset(obj_type)

  # if do_viz:
  #   gc.gc_pomdp.be.render_belief_xy(plt, np.array(gc.b_s))


  LOC_NOMINAL = (0.735, 0.0, arm_control.Z_PROBE[0])
  ORIENT_NOMINAL = (3.14, 0, 0) #(0, 1.57, 0)
  ORIENT_GRASP = (3.14, 0, 0)


  beliefs = []
  actions = []
  max_itr = 5
  a, o = None, None

  ## INJECT ARTIFICIAL KNOWLEDGE OF BELIEF
  real_object_indices = (1, 10, 0)
  real_object_big_index = gc.gc_pomdp.be.indices_to_big_index(real_object_indices)
  print 'Real object pose: %s' % str(gc.gc_pomdp.be.indices_to_continuous_pose(real_object_indices))
  print gc.gc_pomdp.be.x0
  print gc.gc_pomdp.be.x1
  # gc.b_s[real_object_big_index] += 1000.0
  # gc.b_s /= np.sum(gc.b_s)

  beliefs.append(gc.b_s)
  confidence = 0.01
  while confidence < 0.6 and max_itr > 0 and not gc.gc_pomdp.is_terminal_belief(gc.b_s, a, o):

    ## DO SOME PLANNING, CHOOSE NEXT GRASP
    gp = gc.gc_pomdp
    print 'About to solve...'
    a, score = gc.query_action()
    print 'score: %s action: %s' % (str(score), str(a))
    actions.append(a)

    print 'belief first few: %s' % str(gc.b_s[:10])

    ## REACH IN HARDWARE
    my_loc = list(LOC_NOMINAL)
    my_ori = list(ORIENT_NOMINAL)
    my_loc[0], my_loc[1] = max(a[0][0] - gc.FINGER_LENGTH, .45)   ,    a[1][0]
    #my_loc[0], my_loc[1] = max(a[0][0] - .138, .45)   ,    a[1][0]
    assert 0.45 <= my_loc[0] <= 0.875, str(my_loc[0])
    assert -.35 <= my_loc[1] <= .35, str(my_loc[1])
    assert 0.6 <= my_loc[2] <= 1.0, str(my_loc[2])
    print 'REACHING FOR LOCATION ' + str(my_loc)
    probe_loc, probe_ori, probe_touch, probe_Z_idx = arm_control.probe_at( \
            my_loc, my_ori,Z_LIMIT)
    
    ## PROCESS UPDATE
    touch = [(1 if i else 0) for i in probe_touch[:2]] # ONLY FIRST TWO FINGERS, AND MUST SHIFT
    iz = int( max(0, min(2, gc.gc_pomdp.be.z_continuous_to_index(probe_loc[2] - Z_LIMIT))))
    o = (touch, probe_Z_idx)
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
    if do_viz:
     gc.gc_pomdp.be.render_belief_xy(plt, np.array(gc.b_s), fingers_recentered)


  ## GRASP AT GUESSED LOCATION:
  #guessed_obj_pose = gc.gc_pomdp.be.indices_to_continuous_pose(
  #        gc.gc_pomdp.be.big_index_to_indices(np.argmax(gc.b_s)))
  guessed_obj_pose_inds, confidence = gc.query_grasp_loc_confidence()
  guessed_obj_pose = gc.gc_pomdp.be.indices_to_continuous_pose(guessed_obj_pose_inds)
  ox, oy, otheta = guessed_obj_pose
  print "Grasping object at " + str((ox, oy, otheta))
  # CONVERT TO A HAND POSE
  hx = ox+0.07 # reach palm to where object really is
  hy = oy
  hyaw = otheta
  grasp_pre_loc = tuple([hx, hy, arm_control.Z_GRASP[0]])
  grasp_pre_ori = tuple([ORIENT_GRASP[0], ORIENT_GRASP[1], hyaw])

  if gc.block_type == gc.BLOCK_TRIPLE_POKE:
    grasp_dur_loc = tuple([hx, hy, arm_control.Z_GRASP[1]])
  else: # gc.block_type == gc.BLOCK_LONG:
    grasp_dur_loc = tuple([hx, hy, arm_control.Z_GRASP[2]])


  grasp_dur_ori = tuple([ORIENT_GRASP[0], ORIENT_GRASP[1], hyaw])

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
  arm_control.goto_pt((0.151,0,1.1), (0,0.79,0)) #TRIUMPHANT POSE





Z_LIMIT = 0.60
run_live = True
do_viz = False

## LOAD HARDWARE INTERFACE
rospy.init_node('grasp_planning', anonymous=True)
arm_control = ArmControl(live=run_live)
print "moving to calibration point..."
arm_control.open_hand()
arm_control.calibrate_touch_sensor()
print "done. Calibration = {}".format(arm_control.touch_thresh)

print "Creating MPEX Client"
client = MpexClient()
client.add_listener('grasp',pick_up)
client.run()
#pick_up(1,1,1)
print "Ready to accept commands."
rospy.spin()




