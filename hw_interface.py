#! /usr/bin/env python

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

TOUCH_HYSTERESIS = 200 #gripper feedback 0-4048
TOUCH_HYSTERESIS = 80

class ArmControl:

  UNSAFE_Z_LIMIT_METERS = 0.6 # 0.490 # 0.62
  SAFE_Z_LIMIT_METERS = 1.0
  def __init__(self, live=False):
    #Should it actually physically move?
    self.live = live

    #ROS-related init
    #TODO - FIX
    rospy.Subscriber("/wam/fingertip_torque_sensors", Float64MultiArray, self.callback_touch_sensor)
    self.touch = (2000,2000,2000)
    self.touch_thresh = (2000,2000,2000)

    #moveit init
    moveit_commander.roscpp_initialize(sys.argv)
    self.robot = moveit_commander.RobotCommander()
    self.scene = moveit_commander.PlanningSceneInterface()
    self.arm = moveit_commander.MoveGroupCommander("arm")

    #for viewing in RVIZ
    display_trajectory_publisher = rospy.Publisher(
                                      '/move_group/display_planned_path',
                                      moveit_msgs.msg.DisplayTrajectory,
                                      queue_size=20)

  def print_state(sellf):
    print "============ Printing robot state"
    print self.robot.get_current_state()
    print "============"

  def goto_pt(self,location,orientation):
    target = PoseStamped(Header(frame_id='base_footprint'),
			Pose(Point(*location), Quaternion(*list(quaternion_from_euler(*orientation)))))
    #posestamped instead of pose
    #set frame to base_link within header
    print target
    assert location[2] >= self.UNSAFE_Z_LIMIT_METERS
    self.goto_pt_internal(target)

  def goto_pt_internal(self, target):
    self.arm.set_pose_target(target) # TODO : switch to pose_stamped
    plan = self.arm.plan()
    if self.live:
      self.arm.go(wait=True)

  def callback_touch_sensor(self, data):
    self.touch = (data.data[0],data.data[1],data.data[2])

  def calibrate_touch_sensor(self):
    #self.goto_pt((0.55, 0, 0.82),(pi, 0, 0))   # 0.200, -0.547, 1.059
    self.goto_pt((0.800, -0.0, 1.059-.13),(pi, 0, 0))  # Almost touching: (0.800, -0.0, 1.059-.20)
    end = rospy.get_time()+2
    i = 0
    total = [0,0,0]
    while(rospy.get_time() < end):
      for j in range(0,3):
        total[j]+=self.touch[j]
      i+=1
      rospy.sleep(0.05)
    self.touch_thresh = tuple(map(lambda x: (float(x)/i) + TOUCH_HYSTERESIS, total))

  def probe_at(self,location,orientation, Z_LIMIT):
    """ Probes starting from the given point and
        orientation and moves downward in world z
        until Z_LIMIT is reached.
        
        Parameters:
        ---
        location: triple containing xyz
        base_footprint->wam_bhand_base_link

        orientation: triple containing rpy
        base_footprint->wam_bhand_base_link

        Z_LIMIT: (!!!) very important argument for
        safety. Gives the stopping point for
        the hand in meters above base_footprint

        Returns:
        ---
        (loc_final, ori_final, touch):
        loc_final is final xyz,
        ori_final is final rpy,
        touch is numpy array shape (3,) containing
        three booleans: [left forefinger, right
        forefinger, middle finger] """
    my_loc = list(location)
    my_ori = list(orientation)
    
    self.goto_pt(tuple(my_loc), tuple(my_ori))
    
    while my_loc[2] > Z_LIMIT:
      print 'MOVING WITH Z_LIMIT (meters from floor): %f' % Z_LIMIT
      if any(np.less(self.touch_thresh, self.touch)):
        print ('Contact detected!: %s' % str(np.less(self.touch_thresh, self.touch)))
        break
      my_loc[2] -= .02
      if my_loc[2] > Z_LIMIT:
        self.goto_pt(tuple(my_loc), tuple(my_ori))
      print (self.touch)

    if my_loc[2] <= Z_LIMIT:
      print ('Z_LIMIT reached with z value: %f' % my_loc[2])

    # Finally, move back to original location, orientation
    self.goto_pt(tuple(location), tuple(orientation))

    return tuple(my_loc), tuple(my_ori), np.less(self.touch_thresh, self.touch)


    #target = Pose(Point(*location), Quaternion(*list(quaternion_from_euler(*orientation))))
    #self.goto_pt_internal(target)
    #
    #while(all(less(self.touch, self.touch_neutral))):
    #  rospy.sleep(100.0/1000)
    #  target.position.z-= 0.01
    #  self.goto_pt_internal(target)
    #
    #return (arm.getCurrentPose(), greater(self.touch, self.touch_neutral))

  def grasp_at(self,location_approach,orientation_approach,
                    location_pre,orientation_pre,Z_LIMIT):
    my_loc = list(location_approach)
    my_ori = list(orientation_approach)


    return 

    # Approach location (both pre and post)
    self.goto_pt(tuple(my_loc), tuple(my_ori))

    # Go to grasp location
    self.goto_pt(tuple(my_loc), tuple(my_ori))

    # Close hand
    self.goto_pt(tuple(my_loc), tuple(my_ori))

    # Open hand
    
    end = rospy.get_time()+2
    while(rospy.get_time() < end):
      rospy.sleep(0.05)

  def close_hand(self):
    if self.live:
      rospy.wait_for_service('/wam/close')
      f = rospy.ServiceProxy('/wam/close', Empty)
      resp1 = f()
    else:
      print ('Would close hand if live')

  def open_hand(self):
    if self.live:
      rospy.wait_for_service('/wam/open')
      f = rospy.ServiceProxy('/wam/open', Empty)
      resp1 = f()
    else:
      print ('Would open hand if live')

if __name__=='__main__':
  rospy.init_node('arm_control_test', anonymous=True)
  arm_control = ArmControl(live=True)
  print "moving to calibration point..."
  arm_control.calibrate_touch_sensor()
  print "done. Calibration = {}".format(arm_control.touch_thresh)


  # POSE_NOMINAL = (0.800, -0.0, 1.059-.20)


  # SOME FRAMES  base_footprint -> wam_bhand_base_link: 

  # LOWEST ADMISSIBLE Z
  """At time 1494890726.965
    - Translation: [0.904, 0.104, 0.696]
    - Rotation: in Quaternion [0.992, 0.085, -0.057, -0.074]
                in RPY (radian) [-2.984, 0.101, 0.179]
                in RPY (degree) [-170.949, 5.787, 10.246] """

  ## EXTREME RIGHT:
  """At time 1494890828.964
    - Translation: [0.833, -0.338, 0.668]
    - Rotation: in Quaternion [0.998, -0.051, -0.022, -0.044]
                in RPY (radian) [-3.056, 0.048, -0.100]
                in RPY (degree) [-175.107, 2.751, -5.731]"""

  ## CLOSE RIGHT:
  """ At time 1494890996.982
    - Translation: [0.531, -0.331, 0.667]
    - Rotation: in Quaternion [0.999, -0.005, -0.012, -0.041]
                in RPY (radian) [-3.060, 0.024, -0.009]
                in RPY (degree) [-175.339, 1.354, -0.491]"""

  ## FURTHEST REACH FORWARD:
  """ At time 1494891145.983
    - Translation: [0.875, -0.064, 0.708]
    - Rotation: in Quaternion [0.999, -0.008, -0.021, -0.031]
                in RPY (radian) [-3.079, 0.042, -0.014]
                in RPY (degree) [-176.439, 2.415, -0.805]"""

  ## NOMINAL POSE:
  """ At time 1494891237.002
    - Translation: [0.735, -0.076, 0.915]
    - Rotation: in Quaternion [0.999, -0.026, -0.023, -0.013]
                in RPY (radian) [-3.116, 0.047, -0.052]
                in RPY (degree) [-178.523, 2.672, -2.967] """

  ## REALCLOSE RIGHT:
  """ At time 1494893367.562
    - Translation: [0.434, -0.350, 0.681]
    - Rotation: in Quaternion [0.996, -0.022, -0.072, -0.053]
                in RPY (radian) [-3.037, 0.146, -0.036]
                in RPY (degree) [-174.023, 8.352, -2.070] """

  ## REALCLOSE IN FRONT:
  """ At time 1494893474.961
    - Translation: [0.450, 0.042, 0.701]
    - Rotation: in Quaternion [0.998, 0.003, -0.052, 0.020]
                in RPY (radian) [3.101, 0.104, 0.004]
                in RPY (degree) [177.682, 5.977, 0.249] """


  ## NEW ZLIMIT:
  """ At time 1494895421.342
    - Translation: [0.617, -0.255, 0.535]
    - Rotation: in Quaternion [0.997, 0.021, -0.044, -0.054]
                in RPY (radian) [-3.031, 0.087, 0.047]
                in RPY (degree) [-173.645, 4.956, 2.676] """

  ## NEW GRASP PLANE HI
  """ At time 1494895838.689
    - Translation: [0.728, 0.014, 0.670]
    - Rotation: in Quaternion [0.999, 0.039, -0.027, -0.017]
                in RPY (radian) [-3.106, 0.053, 0.079]
                in RPY (degree) [-177.957, 3.011, 4.545] """

  ## NEW GRASP PLANE LO
  """ At time 1494895922.702
    - Translation: [0.742, 0.012, 0.564]
    - Rotation: in Quaternion [0.998, 0.042, -0.040, -0.009]
                in RPY (radian) [-3.120, 0.079, 0.084]
                in RPY (degree) [-178.786, 4.502, 4.837] """

  # LOC_VERY_CLOSE_DONTUSE = [0.450, 0.0, 0.775]
  LOC_NOMINAL = [0.735, 0.0, 0.8] # [0.735, 0.0, 0.69] # [0.735, 0.0, 0.775]
  ORIENT_NOMINAL = [-3.116, 0.047, -0.052]

  my_loc = LOC_NOMINAL
  my_ori = ORIENT_NOMINAL
  
  arm_control.probe_at(my_loc, my_ori, arm_control.UNSAFE_Z_LIMIT_METERS)
  # arm_control.close_hand()
  

  # pose, contact_arr = arm_control.probe_at((0,0,0),(0,0,0))
  #print "Arm collided at: {} with collisions on: {}".format(pose,contact_arr)
