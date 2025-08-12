# VRX내 WAM-V 카메라로 보이는 화면을 YOLO를 통해 추론
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from rclpy.qos import qos_profile_sensor_data

class YoloCamNode(Node):
    def __init__(self):
        super().__init__('yolo_cam_node')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/wamv/sensors/cameras/front_left_camera_sensor/optical/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )
        self.model = YOLO("yolo11n.pt") 
        self.get_logger().info("YOLO node initialized!")

    def image_callback(self, msg):
        self.get_logger().info("Image received.")
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            results = self.model.predict(source=frame, conf=0.5, save=False)
            annotated = results[0].plot()
            cv2.imshow("YOLO Detection", annotated)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloCamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
