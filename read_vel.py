import rospy
from geometry_msgs.msg import Point, Twist, PoseStamped, Pose, Vector3Stamped

class gps_vel():
    def __init__(self):
        rospy.init_node('cmd_vel_publisher')
        self.r = rospy.Rate(10)#10Hz

        self.gps_vel_x = 0
        self.gps_vel_y = 0
        self.gps_vel_z = 0

    def listener(self):
        self.sub = rospy.Subscriber("/navsat/vel", Vector3Stamped, self.callback)


    def callback(self, msg):
        self.gps_vel_x = msg.vector.x
        self.gps_vel_y = msg.vector.y
        self.gps_vel_z = msg.vector.z


if __name__ == '__main__':

    velocity = gps_vel()
    velocity.listener()

# Prints the same two times
    while not rospy.is_shutdown():
        velocity.r.sleep()
        print("Velocity from GPS:")
        print("X: ", velocity.gps_vel_x)
        print("Y: ", velocity.gps_vel_y)
        print("Z: ", velocity.gps_vel_y)



