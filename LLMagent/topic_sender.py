#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class UserQuestionPublisher(Node):
    def __init__(self):
        super().__init__("user_question_publisher")
        self.pub = self.create_publisher(String, "/user_question", 10)
        self.get_logger().info("✅ /user_question 퍼블리셔 노드 시작")

    def publish_text(self, text: str):
        msg = String()
        msg.data = text
        self.pub.publish(msg)
        self.get_logger().info(f"📤 /user_question 발행: {text}")


def main(args=None):
    rclpy.init(args=args)
    node = UserQuestionPublisher()

    try:
        while rclpy.ok():
            text = input("🗣 보낼 문장 입력 (q 입력 시 종료): ").strip()
            if text.lower() == "q":
                break
            if not text:
                continue
            node.publish_text(text)
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt로 종료")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
