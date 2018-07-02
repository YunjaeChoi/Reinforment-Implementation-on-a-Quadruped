# Reinforcement Learning Implementation on a Quadruped
My project in Korea University. 

# Overview
This project is a reinforment learning environment for a quadruped. It uses python 2.7, ROS (robot operating system) Kinetic Kame, Tensorflow, Gazebo simulation. The robot interacts with both Gazebo simulation and the real world using ROS. It is trained to walk forward in Gazebo simulation and can be deployed in a real robot. Deep Deterministic Policy Gradient(DDPG) is used for the robot.

# Files
#### quadruped folder
This folder contains files for reinforment learning environment.

- config/quadruped_control.yaml : joint controller config for Gazebo
- launch/quadruped_control.launch : Spawn quadruped model with controller in Gazebo
- launch/quadruped_gazebo.launch : Spawn quadruped model only in Gazebo
- launch/rvis_display.launch : display quadruped model in Rvis
- rvis/urdf.rvis : rvis config
- src/model/ : tensorflow model files of learned agent
- src/agent_mem.p : agent Replay buffer memory
- src/eps_rewards.npy : agent episode rewards
- src/ddpg.py : Deep Deterministic Policy Gradient agent written in tensorflow
- src/quadruped_env.py : quadruped environment for gazebo simulation
- src/quadruped_learn.py : learn a ddpg agent
- src/quadruped_gazebo_act.py : learned agent takes actions in Gazebo
- src/quadruped_model.py : quadruped environment for real world (no learning, only actions)
- src/quadruped_act.py : learned agent takes actions in a real quadruped
- src/plot_eps_rewards.py : plot episode rewards from eps_rewards.npy
- urdf/materials.xacro : colors etc. for quadruped urdf
- urdf/quadruped_model.xacro : xacro file for quadruped urdf
- worlds/quadruped.world : Gazebo world file
- CMakeLists.txt : ROS CMakeLists file
- package.xml : ROS package file


#### quadruped_imu_and_servo folder
This folder contains files for a real robot implementation. ROS should be installed on the real robot. (Running confirmed on Raspberry pi model B + Ubuntu MATE)


#### quadruped_imu_publisher
- launch/quadruped_imu_publisher_launch.launch : publish IMU readings from adafruit BNO055 9DOF IMU
- src/quadruped_imu_publisher.py : publisher src file
- CMakeLists.txt : ROS CMakeLists file
- package.xml : ROS package file


#### quadruped_servo_subscriber
- launch/quadruped_servo_subscriber_launch.launch : control servo position(PWM) from published servo position messages
- src/quadruped_servo_subscriber.py : subscriber src file
- CMakeLists.txt : ROS CMakeLists file
- package.xml : ROS package file


# Prerequisites 
- python 2.7
- ROS Kinetic Kame
- Tensorflow >= 1.4
- Gazebo simulation

### additional installations for simulation
- ROS control
```
sudo apt-get install ros-kinetic-ros-control
sudo apt-get install ros-kinetic-ros-controllers
sudo apt-get install ros-kinetic-gazebo-ros-control
```

- Gazebo simulation update -> update to 7.x (7.0 causes error)

### additional installations for real robot
-

# How to
- config ROS packages (quadruped, quadruped_imu_publisher, quadruped_servo_subscriber)

#### Learning an agent
```
roslaunch quadruped quadruped_control.launch
python quadruped_learn.py
```

#### simulate learned agent
```
roslaunch quadruped quadruped_control.launch
python quadruped_gazebo_act.py
```

#### learned agent on real quadruped
- setup networking for ROS if needed (run roscore on master)

- on quadruped
```
roslaunch quadruped_imu_publisher quadruped_imu_publisher_launch.launch
roslaunch quadruped_servo_subscriber quadruped_servo_subscriber_launch.launch
```

- on computer running tensorflow
```
python quadruped_act.py
```


# Tips
#### always
```
source ~/catkin_ws/devel/setup.bash
```
#### ROS core
roscore

#### ROS via network (local LAN)
- MASTER Setting -> get master_ip_addr via ifconfig (master runs roscore)
```
export ROS_MASTER_URI=http://master_ip_addr:11311
export ROS_IP=master_ip_addr
export ROS_MASTER_URI=http://192.168.0.5:11311
```

- SECOND Setting (from other computer)
```
ssh second_ip_addr -p(port_num)
export ROS_MASTER_URI=http://master_ip_addr:11311
export ROS_IP=second_ip_addr
```

- exiting ssh
```
exit
```

#### xacro to URDF file
```
rosrun xacro xacro --inorder quadruped_model.xacro > model1.urdf
```

# Videos
- simulation
https://youtu.be/jo47bkJQrjU

- real quadruped
https://youtu.be/T0lQ6aDYJkE
