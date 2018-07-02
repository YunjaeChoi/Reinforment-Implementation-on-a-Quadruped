#! /usr/bin/env python
from __future__ import print_function
#from builtins import range

import rospy
import time
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryActionGoal, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_srvs.srv import Empty
from sensor_msgs.msg import JointState
from sensor_msgs.msg import Imu

import numpy as np

class AllJoints:
    def __init__(self,joint_name_lst):
        rospy.loginfo('Waiting for joint trajectory Publisher')
        self.jtp = rospy.Publisher('/quadruped/joint_trajectory_controller/command',JointTrajectory,queue_size=1)
        rospy.loginfo('Found joint trajectory Publisher!')
        self.joint_name_lst = joint_name_lst
        self.jtp_zeros = np.zeros(len(joint_name_lst))

    def move_jtp(self, pos):
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        point.accelerations = self.jtp_zeros
        point.effort = self.jtp_zeros
        point.time_from_start = rospy.Duration(1.0/60.0)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

    def reset_move_jtp(self, pos):
        jtp_msg = JointTrajectory()
        self.jtp.publish(jtp_msg)
        jtp_msg = JointTrajectory()
        jtp_msg.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = pos
        point.velocities = self.jtp_zeros
        point.accelerations = self.jtp_zeros
        point.effort = self.jtp_zeros
        point.time_from_start = rospy.Duration(0.0001)
        jtp_msg.points.append(point)
        self.jtp.publish(jtp_msg)

class QuadrupedEnvironment:
    def __init__(self):
        rospy.init_node('joint_position_node')
        self.nb_joints = 12
        self.nb_links = 13
        self.state_shape = (self.nb_joints * 2 + 4 + 3 + 3,) #joint states + orientation
        self.action_shape = (self.nb_joints,)
        self.link_name_lst = ['quadruped::base_link',
                             'quadruped::front_right_leg1', 'quadruped::front_right_leg2', 'quadruped::front_right_leg3',
                             'quadruped::front_left_leg1', 'quadruped::front_left_leg2', 'quadruped::front_left_leg3',
                             'quadruped::back_right_leg1', 'quadruped::back_right_leg2', 'quadruped::back_right_leg3',
                             'quadruped::back_left_leg1', 'quadruped::back_left_leg2', 'quadruped::back_left_leg3']
        self.leg_link_name_lst = self.link_name_lst[1:]
        self.joint_name_lst = ['front_right_leg1_joint', 'front_right_leg2_joint', 'front_right_leg3_joint',
                               'front_left_leg1_joint', 'front_left_leg2_joint', 'front_left_leg3_joint',
                               'back_right_leg1_joint', 'back_right_leg2_joint','back_right_leg3_joint',
                               'back_left_leg1_joint', 'back_left_leg2_joint', 'back_left_leg3_joint']
        self.all_joints = AllJoints(self.joint_name_lst)
        self.starting_pos = np.array([-0.01, 0.01, 0.01,
                                     -0.01, 0.01, -0.01,
                                     -0.01, 0.01, -0.01,
                                     -0.01, 0.01, 0.01])

        self.last_pos = np.zeros(3)
        self.last_ori = np.zeros(4)

        self.joint_pos_high = np.array([1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0,
                                     1.0, 1.0, 1.0])
        self.joint_pos_low = np.array([-1.0, -1.0, -1.0,
                                    -1.0, -1.0, -1.0,
                                    -1.0, -1.0, -1.0,
                                    -1.0, -1.0, -1.0])
        self.joint_pos_coeff = np.array([2.0, 3.0, 2.0,
                                        2.0, 3.0, 2.0,
                                        2.0, 3.0, 2.0,
                                        2.0, 3.0, 2.0])
        self.joint_pos_range = self.joint_pos_high - self.joint_pos_low
        self.joint_pos_mid = (self.joint_pos_high + self.joint_pos_low) / 2.0
        self.joint_pos = self.starting_pos
        self.joint_state = np.zeros(self.nb_joints)
        # -1 ~ 1 to -pi*2/18 ~ pi*2/18
        self.joint_pos_to_state_factor = np.pi*2.0/18.0

        self.orientation = np.zeros(4)
        self.angular_vel = np.zeros(3)
        self.linear_acc = np.zeros(3)
        self.imu_subscriber = rospy.Subscriber('/quadruped/imu',Imu,
                                                       self.imu_subscriber_callback)
        self.normed_sp = self.normalize_joint_state(self.starting_pos)
        self.state = np.zeros(self.state_shape)
        self.diff_state_coeff = 3.0
        self.action_coeff = 1.0
        self.linear_acc_coeff = 0.1
        self.last_action = np.zeros(self.nb_joints)

    def normalize_joint_state(self,joint_pos):
        return joint_pos * self.joint_pos_coeff

    def imu_subscriber_callback(self,imu):
        self.orientation = np.array([imu.orientation.x,imu.orientation.y,imu.orientation.z,imu.orientation.w])
        self.angular_vel = np.array([imu.angular_velocity.x,imu.angular_velocity.y,imu.angular_velocity.z])
        self.linear_acc = np.array([imu.linear_acceleration.x,imu.linear_acceleration.y,imu.linear_acceleration.z])

    def reset(self):
        self.joint_pos = self.starting_pos
        self.all_joints.reset_move_jtp(self.starting_pos)
        rospy.sleep(0.5)
        self.state = np.zeros(self.state_shape)
        self.last_joint = self.joint_state
        diff_joint = np.zeros(self.nb_joints)
        normed_js = self.normalize_joint_state(self.joint_state)
        self.state = np.concatenate((normed_js,diff_joint,self.orientation,self.angular_vel,self.linear_acc_coeff*self.linear_acc)).reshape(1,-1)
        self.last_action = np.zeros(self.nb_joints)

        return self.state

    def step(self, action):
        print('action:',action)
        action = action * self.joint_pos_range * self.action_coeff
        self.joint_pos = np.clip(self.joint_pos + action,a_min=self.joint_pos_low,a_max=self.joint_pos_high)
        self.all_joints.move_jtp(self.joint_pos)
        print('joint pos:',self.joint_pos)

        rospy.sleep(15.0/60.0)

        #normed_js = self.normalize_joint_state(self.joint_state)
        normed_js = self.normalize_joint_state(self.joint_pos * self.joint_pos_to_state_factor)

        diff_joint = self.diff_state_coeff * (normed_js - self.last_joint)

        self.state = np.concatenate((normed_js,diff_joint,self.orientation,self.angular_vel,self.linear_acc_coeff*self.linear_acc)).reshape(1,-1)

        self.last_joint = normed_js
        self.last_action = action

        print('state',self.state)
        return self.state
