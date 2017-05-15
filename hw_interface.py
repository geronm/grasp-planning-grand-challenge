import rospy
import moveit_commander
import moveit_msgs.msg
from geometry_msgs.msg import Pose, Point, Quaternion, Twist
from tf.transformations import quaternion_from_euler

TOUCH_HYSTERESIS = 200 #gripper feedback 0-4048

class ArmControl:

  def __init__(self, live=False):
    #Should it actually physically move?
    self.live = live

    #ROS-related init
    #TODO - FIX
    rospy.Subscriber("TOUCH_SENSORS_HERE", MSG_TYPE, self.callback_touch_sensor)
    self.touch = (2000,2000,2000)
    self.touch_thresh = (2000,2000,2000)

    #moveit init
    moveit_commander.roscpp_initialize()
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
    x,y,z = location
    r,p,y = orientation
    target = Pose(Point(x,y,z), quaternion_from_euler(r, p, y, axes='sxyz'))
    goto_pt_internal(target)

  def goto_pt_internal(self, target):
    self.arm.set_pose_target(target)
    plan = self.arm.plan()
    if self.live:
      self.arm.go(wait=True)

  def callback_touch_sensor(self, data):
    self.touch = (data.a, data.b, data.c)

  def calibrate_touch_sensor(self):
    self.goto_pt((0,0,0),(0,0,0))
    end = rospy.get_time()+2
    i = 0
    total = [0,0,0]
    while(rospy.get_time() < end):
      for j in range(0,3):
        total[j]+=self.touch[j]
      i+=1
      rospy.sleep(0.05)
    self.touch_thresh = tuple(map(lambda x: (float(x)/i) + TOUCH_HYSTERESIS, total))
      
  def probe_at(self,location,orientation):
    target = Pose(Point(x,y,z), quaternion_from_euler(r, p, y, axes='sxyz'))
    self.goto_pt_internal(target)

    while(all(less(self.touch, self.touch_neutral))):
      rospy.sleep(100.0/1000)
      target.position.z-= 0.01
      self.goto_pt_internal(target)

    return (arm.getCurrentPose(), greater(self.touch, self.touch_neutral))


if __name__=='__main__':
  rospy.init_node('arm_control_test', anonymous=True)
  arm_control = ArmControl()
  arm_control.calibrate_touch_sensor()
  pose, contact_arr = arm_control.probe_at((0,0,0),(0,0,0))
  print "Arm collided at: {} with collisions on: {}".format(pose,contact_arr)
