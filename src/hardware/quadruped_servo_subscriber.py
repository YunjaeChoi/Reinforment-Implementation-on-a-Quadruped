#! /usr/bin/env python
from __future__ import division
from __future__ import print_function

import rospy
import numpy as np
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import time
# Import the PCA9685 module.
import Adafruit_PCA9685

# Initialise the PCA9685 using the default address (0x40).
pwm = Adafruit_PCA9685.PCA9685()

# Set frequency to 60hz, good for servos.
pwm.set_pwm_freq(60)

servo_low = 150
servo_high = 600
servo_mid = (servo_high + servo_low)/2.0
servo_half_range = (servo_high - servo_low)/2.0
# 90 degrees to 20 degrees
model_joint_range_coeff = 2.0/9.0
model_joint_zero_adjust = np.array([0.0, -1.0, 0.0,
									0.0, -1.0, 0.0,
									0.0, -1.0, 0.0,
									0.0, -1.0, 0.0])
model_joint_mirror_adjust = np.array([1.0, -1.0, 1.0,
									-1.0, 1.0, 1.0,
									1.0, -1.0, 1.0,
									-1.0, 1.0, 1.0])
print('Moving servo, press Ctrl-C to quit...(last servo position will be maintained)')

def callback(jtp_msg):
	if len(jtp_msg.points)>0:
		pos = np.array(jtp_msg.points[-1].positions)
		servo_pos = model_joint_range_coeff * (pos + model_joint_zero_adjust) * model_joint_mirror_adjust
		servo_pos = servo_half_range * servo_pos + servo_mid
		servo_pos = servo_pos.astype(int)
		# Move servo on channel (0~15)
		for channel in xrange(12):
			pwm.set_pwm(channel, 0, servo_pos[channel])



rospy.init_node('servo_subscriber')
sub = rospy.Subscriber('/quadruped/joint_trajectory_controller/command', JointTrajectory, callback)

rospy.spin()
