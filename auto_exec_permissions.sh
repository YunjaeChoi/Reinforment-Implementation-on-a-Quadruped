#!/usr/bin/bash
find /home/rohan/catkin_ws/src/arthrobots_git/quadruped/src/ddpg/ -type f -iname "*.py" -exec chmod +x {} \;
find /home/rohan/catkin_ws/src/arthrobots_git/quadruped/src/dreamerv2/ -type f -iname "*.py" -exec chmod +x {} \;
find /home/rohan/catkin_ws/src/arthrobots_git/quadruped/src/dreamerv3/ -type f -iname "*.py" -exec chmod +x {} \;
find /home/rohan/catkin_ws/src/arthrobots_git/quadruped/src/qlearning/ -type f -iname "*.py" -exec chmod +x {} \;
find /home/rohan/catkin_ws/src/arthrobots_git/quadruped/src/hardware/ -type f -iname "*.py" -exec chmod +x {} \;
