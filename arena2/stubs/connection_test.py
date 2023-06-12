import robomaster
from robomaster import robot
if __name__ == '__main__':
	ep_robot = robot.Robot()
	# Specify the connection mode
	# ap corresponds to Wi-Fi direct connection
	ep_robot.initialize(conn_type='ap')
	version = ep_robot.get_version()
	# Robot version will be printed upon successful connection
	print("Robot version: {0}".format(version))
	ep_robot.close()
