# Grasp Planning Team - Grand Challenge Overview

The plan for our task is to localize and grasp one of several objects, given an initial Gaussian belief distribution over the object pose.

We will write modules to handle the task: "go over and pick up object, which is of type X and is at location Y, off of the floor". Executing this grasp will first require us to position our base so that the object is positioned in the arm's field of manipulation (in front of the base). Next, we will use probing actions to tightly-localize the object in <x,y,theta>. Finally, we will use an object-type- and pose- dependent grasping maneuver to pick up the object.

System modules:

* ROS-based motion API which provides the necessary primitive actions. This may include:
    * move the robot base so that the object at Y is in front of it
    * position the arm in the crane-like "ready position" for probing, with fingers splayed
    * move the arm to (dx,dy) and move down until contact is made, and report results
    * grasp an object by positioning the hand over (dx, dy, theta, z), moving the arm down to (dx, dy, theta, z'), closing fingers tightly, then lifting arm back up to (dx, dy, theta, z).

* Planning library capable of performing finite lookahead search over the grasping domain, complete with:
    * different object representations
    * belief distribution over x, y, theta
    * proper probing actions and observation model
    * possibly a transition model
    * expectimax search to determine next probing action
    * ability to sense, for a given belief, whether confident enough to grasp

* ROS node which utilizes the planning and motion libraries to accomplish our task

* Datatype that will be sent from object recognition team to our node to tell us object properties


We will work with the object recognition team on this task. Their job will be to identify the type of object and its rough pose in the world. Our job will then be to grasp the known object.


# Hardware Notes



Arm:
MoveIt!

Robot: Wam
Group name: arm


Drivetrain:
Move_Base

Ros master 192.168.2.100:11311
Mersxram/cogrob2017 

