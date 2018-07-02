#! /usr/bin/env python
from __future__ import print_function
import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Quaternion, Vector3
from Adafruit_BNO055 import BNO055

rospy.init_node('imu_publisher')
pub = rospy.Publisher('/quadruped/imu', Imu, queue_size=1)
bno = BNO055.BNO055(serial_port='/dev/serial0', rst=18)
rate = rospy.Rate(60)
imu_val = Imu()

# Initialize the BNO055 and stop if something went wrong.
if not bno.begin():
	raise RuntimeError('Failed to initialize BNO055! Is the sensor connected?')

# Print system status and self test result.
status, self_test, error = bno.get_system_status()
print('System status: {0}'.format(status))
print('Self test result (0x0F is normal): 0x{0:02X}'.format(self_test))
# Print out an error if system status is in error mode.
if status == 0x01:
	print('System error: {0}'.format(error))
	print('See datasheet section 4.3.59 for the meaning.')

# Print BNO055 software revision and other diagnostic data.
sw, bl, accel, mag, gyro = bno.get_revision()
print('Software version:   {0}'.format(sw))
print('Bootloader version: {0}'.format(bl))
print('Accelerometer ID:   0x{0:02X}'.format(accel))
print('Magnetometer ID:    0x{0:02X}'.format(mag))
print('Gyroscope ID:       0x{0:02X}\n'.format(gyro))

#setting axis
#bno.set_axis_remap(0,1,2,z_sign=1)
print('Axis:',bno.get_axis_remap())
print('Reading BNO055 data, press Ctrl-C to quit...')
while not rospy.is_shutdown():
	quat = bno.read_quaternion()
	imu_val.orientation = Quaternion(quat[0],quat[1],quat[2],quat[3])
	ang = bno.read_gyroscope()
	imu_val.angular_velocity = Vector3(ang[0],ang[1],ang[2])
	#lin = bno.read_linear_acceleration()
	lin = bno.read_accelerometer()
	imu_val.linear_acceleration = Vector3(lin[0],lin[1],lin[2])
	pub.publish(imu_val)
	rate.sleep()
