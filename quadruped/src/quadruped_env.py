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
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest
from gazebo_msgs.srv import GetModelState, GetModelStateRequest
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Imu

import numpy as np

class AllJoints:
    def __init__(self,joint_name_lst):
        self.jta = actionlib.SimpleActionClient('/quadruped/joint_trajectory_controller/follow_joint_trajectory',
                                                FollowJointTrajectoryAction)
        rospy.loginfo('Waiting for joint trajectory action')
        self.jta.wait_for_server()
        rospy.loginfo('Found joint trajectory action!')
        self.jtp = rospy.Publisher('/quadruped/joint_trajectory_controller/command',JointTrajectory,queue_size=1)
        self.joint_name_lst = joint_name_lst
        self.jtp_zeros = np.zeros(len(joint_name_lst))

    def move(self, pos):
        msg = FollowJointTrajectoryActionGoal()
        msg.goal.trajectory.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start = rospy.Duration(1.0/60.0)
        msg.goal.trajectory.points.append(point)
        self.jta.send_goal_and_wait(msg.goal)

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

    def reset_move(self, pos):
        jtp_msg = JointTrajectory()
        self.jtp.publish(jtp_msg)
        msg = FollowJointTrajectoryActionGoal()
        msg.goal.trajectory.joint_names = self.joint_name_lst
        point = JointTrajectoryPoint()
        point.positions = pos
        point.time_from_start = rospy.Duration(0.0001)
        msg.goal.trajectory.points.append(point)
        self.jta.send_goal(msg.goal)

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

        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics',Empty)
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics',Empty)
        self.model_config_proxy = rospy.ServiceProxy('/gazebo/set_model_configuration',SetModelConfiguration)
        self.model_config_req = SetModelConfigurationRequest()
        self.model_config_req.model_name = 'quadruped'
        self.model_config_req.urdf_param_name = 'robot_description'
        self.model_config_req.joint_names = self.joint_name_lst
        self.model_config_req.joint_positions = self.starting_pos
        self.model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        self.model_state_req = SetModelStateRequest()
        self.model_state_req.model_state = ModelState()
        self.model_state_req.model_state.model_name = 'quadruped'
        self.model_state_req.model_state.pose.position.x = 0.0
        self.model_state_req.model_state.pose.position.y = 0.0
        self.model_state_req.model_state.pose.position.z = 0.25
        self.model_state_req.model_state.pose.orientation.x = 0.0
        self.model_state_req.model_state.pose.orientation.y = 0.0
        self.model_state_req.model_state.pose.orientation.z = 0.0
        self.model_state_req.model_state.pose.orientation.w = 0.0
        self.model_state_req.model_state.twist.linear.x = 0.0
        self.model_state_req.model_state.twist.linear.y = 0.0
        self.model_state_req.model_state.twist.linear.z = 0.0
        self.model_state_req.model_state.twist.angular.x = 0.0
        self.model_state_req.model_state.twist.angular.y = 0.0
        self.model_state_req.model_state.twist.angular.z = 0.0
        self.model_state_req.model_state.reference_frame = 'world'

        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)
        self.get_model_state_req = GetModelStateRequest()
        self.get_model_state_req.model_name = 'quadruped'
        self.get_model_state_req.relative_entity_name = 'world'
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
        self.joint_pos_coeff = 3.0
        self.joint_pos_range = self.joint_pos_high - self.joint_pos_low
        self.joint_pos_mid = (self.joint_pos_high + self.joint_pos_low) / 2.0
        self.joint_pos = self.starting_pos
        self.joint_state = np.zeros(self.nb_joints)
        self.joint_state_subscriber = rospy.Subscriber('/quadruped/joint_states',JointState,
                                                       self.joint_state_subscriber_callback)
        self.orientation = np.zeros(4)
        self.angular_vel = np.zeros(3)
        self.linear_acc = np.zeros(3)
        self.imu_subscriber = rospy.Subscriber('/quadruped/imu',Imu,
                                                       self.imu_subscriber_callback)
        self.normed_sp = self.normalize_joint_state(self.starting_pos)
        self.state = np.zeros(self.state_shape)
        self.diff_state_coeff = 4.0
        self.reward_coeff = 20.0
        self.reward = 0.0
        self.done = False
        self.episode_start_time = 0.0
        self.max_sim_time = 15.0
        self.pos_z_limit = 0.18
        self.action_coeff = 1.0
        self.linear_acc_coeff = 0.1
        self.last_action = np.zeros(self.nb_joints)

    def normalize_joint_state(self,joint_pos):
        return joint_pos * self.joint_pos_coeff

    def joint_state_subscriber_callback(self,joint_state):
        self.joint_state = np.array(joint_state.position)

    def imu_subscriber_callback(self,imu):
        self.orientation = np.array([imu.orientation.x,imu.orientation.y,imu.orientation.z,imu.orientation.w])
        self.angular_vel = np.array([imu.angular_velocity.x,imu.angular_velocity.y,imu.angular_velocity.z])
        self.linear_acc = np.array([imu.linear_acceleration.x,imu.linear_acceleration.y,imu.linear_acceleration.z])

    def reset(self):
        #pause physics
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except rospy.ServiceException, e:
            print('/gazebo/pause_physics service call failed')
        #set models pos from world
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.model_state_proxy(self.model_state_req)
        except rospy.ServiceException, e:
            print('/gazebo/set_model_state call failed')
        #set model's joint config
        rospy.wait_for_service('/gazebo/set_model_configuration')
        try:
            self.model_config_proxy(self.model_config_req)
        except rospy.ServiceException, e:
            print('/gazebo/set_model_configuration call failed')

        self.joint_pos = self.starting_pos
        self.all_joints.reset_move_jtp(self.starting_pos)
        #unpause physics
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except rospy.ServiceException, e:
            print('/gazebo/unpause_physics service call failed')

        rospy.sleep(0.5)
        self.reward = 0.0
        self.state = np.zeros(self.state_shape)
        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)
        pos = np.array([model_state.pose.position.x, model_state.pose.position.y, model_state.pose.position.z])
        done = False
        self.last_joint = self.joint_state
        self.last_pos = pos
        diff_joint = np.zeros(self.nb_joints)
        normed_js = self.normalize_joint_state(self.joint_state)
        self.state = np.concatenate((normed_js,diff_joint,self.orientation,self.angular_vel,self.linear_acc_coeff*self.linear_acc)).reshape(1,-1)
        self.episode_start_time = rospy.get_time()
        self.last_action = np.zeros(self.nb_joints)
        return self.state, done

    def step(self, action):
        print('action:',action)
        action = action * self.joint_pos_range * self.action_coeff
        self.joint_pos = np.clip(self.joint_pos + action,a_min=self.joint_pos_low,a_max=self.joint_pos_high)
        self.all_joints.move_jtp(self.joint_pos)
        print('joint pos:',self.joint_pos)

        rospy.sleep(15.0/60.0)

        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)
        pos = np.array([model_state.pose.position.x, model_state.pose.position.y, model_state.pose.position.z])

        self.reward = self.reward_coeff * (pos[1] - self.last_pos[1] - np.sqrt((pos[0]-self.last_pos[0])**2))
        print('pos reward:', self.reward)
        self.reward -=  0.75 * np.sqrt(np.sum((self.orientation)**2))

        normed_js = self.normalize_joint_state(self.joint_state)
        #self.reward -= 0.25 * np.sqrt(np.sum((self.normed_sp - normed_js)**2))

        diff_joint = self.diff_state_coeff * (normed_js - self.last_joint)

        self.state = np.concatenate((normed_js,diff_joint,self.orientation,self.angular_vel,self.linear_acc_coeff*self.linear_acc)).reshape(1,-1)

        self.last_joint = normed_js
        self.last_pos = pos
        self.last_action = action

        curr_time = rospy.get_time()
        print('time:',curr_time - self.episode_start_time)
        if (curr_time - self.episode_start_time) > self.max_sim_time:
            done = True
            self.reset()
        elif(model_state.pose.position.z < self.pos_z_limit):
            done = False
            self.reward += -1.0
        else:
            done = False
        print('state',self.state)

        self.reward = np.clip(self.reward,a_min=-10.0,a_max=10.0)
        return self.state, self.reward, done
