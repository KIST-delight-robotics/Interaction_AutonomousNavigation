
import rclpy
from rclpy.node import Node
from unitree_go.msg import SportModeState

class SportModeStateListener(Node):
    def __init__(self):
        super().__init__('sport_mode_state_listener')

        self.subscription_sport = self.create_subscription(
            SportModeState,
            '/sportmodestate',
            self.listener_callback_sport,
            10)

    def listener_callback_sport(self, msg):
        x0 = msg.position[0]
        y0 = msg.position[1]
        yaw = msg.yaw_speed
        self.get_logger().info(f'/sportmodestate position: x={x0}, y={y0}, yaw={yaw}')




def main(args=None):
    rclpy.init(args=args)
    node = SportModeStateListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()