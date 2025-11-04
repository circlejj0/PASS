#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# [!!] 수정됨 (DockingGuid)
# [!!] 3개 도형 감지 및 순서 판단 로직 추가

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from ament_index_python.packages import get_package_share_directory
import os

class DockingGuid(Node):
    def __init__(self):
        super().__init__('docking_Guid')

        # [!!] 감지할 타겟 3개 (사용자님 스크린샷 기준 'rectangle' 사용)
        self.declare_parameter('camera_topic', '/wamv/sensors/cameras/front_left_camera_sensor/optical/image_raw')
        self.declare_parameter('show', True)
        self.declare_parameter('targets', ['circle', 'triangle', 'rectangle']) 
        
        gp = self.get_parameter
        self.camera_topic = gp('camera_topic').value
        self.show = bool(gp('show').value)
        self.targets = [t.lower() for t in gp('targets').value]
        
        # [!!] Navi가 "ARRIVED_SCAN_P1"을 보내면 True가 됨
        self.arrived = False 
        # [!!] 순서 판단을 한 번만 수행하기 위한 래치(latch)
        self.order_determined = False

        self.create_subscription(String, '/nav/status', self.status_cb, 10)
        self.create_subscription(Image, self.camera_topic, self.image_cb, 10)

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.errors_pub = self.create_publisher(Float64MultiArray, 'visual_errors', qos_profile)
        # [!!] 감지된 도형의 순서를 발행할 퍼블리셔
        self.order_pub = self.create_publisher(String, '/docking/target_order', qos_profile)
        
        self.bridge = CvBridge()

        # [!!] (패키지 이름이 'pass_docking'이 아닐 경우 수정 필요)
        pkg_share = get_package_share_directory('pass_docking') 
        model_path = os.path.join(pkg_share, 'weights', 'vrx_dock.pt')
        self.model = YOLO(model_path)

        self.get_logger().info(f"YOLO model loaded from: {model_path}")
        self.get_logger().info(f"Looking for targets: {self.targets}")
        if len(self.targets) != 3:
            self.get_logger().warn(f"Warning: Expected 3 targets, but got {len(self.targets)}. Order logic may fail.")

    def status_cb(self, msg: String):
        # [!!] Navi가 P1 회전 완료 후 "ARRIVED_SCAN_P1"을 발행할 것임
        if msg.data == "ARRIVED_SCAN_P1" and not self.arrived:
            self.arrived = True
            self.get_logger().info("[Guidance] ARRIVED_SCAN_P1 -> Start YOLO detection and error publishing")

    def image_cb(self, msg: Image):
        if not self.arrived:
            return # [!!] Navi 신호 오기 전까지 반환됨

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w, _ = frame.shape
        center_x = w // 2

        results = self.model(frame, verbose=False)
        
        # [!!] self.targets 리스트에 기반하여 errors와 centers 딕셔너리 초기화
        errors = {target: 0.0 for target in self.targets}
        target_centers = {} # (cx, cy) 저장용

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                cls_name = self.model.names[cls_id].lower()
                conf = float(box.conf.item())

                if cls_name in self.targets and conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # (간단한 처리: conf가 더 높은 것을 덮어쓰거나, 그냥 둠)
                    errors[cls_name] = float(cx - center_x)
                    target_centers[cls_name] = (cx, cy) # 중심 좌표 저장
                    
                    if self.show:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                        text = f"{cls_name} ({conf:.2f})"
                        cv2.putText(frame, text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # --- [!!] 순서 판단 로직 (핵심) ---
        # 1. 3개의 타겟이 모두 감지되었는지 확인
        all_found = all(t in target_centers for t in self.targets)
        
        # 2. 모두 감지되었고, 아직 순서 판단(order_determined)을 안했다면
        if all_found and not self.order_determined:
            self.order_determined = True # [!!] 래치(latch) 잠금!
            
            # 3. target_centers 딕셔너리를 x좌표(cx) 기준으로 정렬
            # item[1][0]은 cx 값을 의미
            sorted_targets = sorted(target_centers.items(), key=lambda item: item[1][0])
            
            # 4. 정렬된 순서대로 이름만 추출
            ordered_names = [name for name, center in sorted_targets]
            
            # 5. 쉼표(,)로 구분된 문자열 생성 (예: "circle,rectangle,triangle")
            order_str = ",".join(ordered_names)
            
            self.get_logger().info(f"--- [!!] All targets detected! Order: {order_str} [!!] ---")
            
            # 6. 순서 정보 발행
            self.order_pub.publish(String(data=order_str))

        # --- [!!] 에러 메시지 발행 (순서와 관계없이 계속 발행) ---
        errors_msg = Float64MultiArray()
        # [!!] self.targets에 정의된 순서대로 에러 값을 리스트에 담아 발행
        # [!!] (예: ['circle', 'triangle', 'rectangle'] 순서였다면 [circle_err, tri_err, rect_err])
        errors_msg.data = [errors.get(t, 0.0) for t in self.targets]
        self.errors_pub.publish(errors_msg)

        if self.show:
            cv2.line(frame, (center_x, 0), (center_x, h), (0, 255, 255), 1)
            # 감지된 모든 타겟에 대해 라인 그리기
            for name, center in target_centers.items():
                if name == 'circle': color = (255, 0, 0)
                elif name == 'triangle': color = (0, 0, 255)
                elif name == 'rectangle': color = (255, 0, 255) # 마젠타
                else: color = (255, 255, 0)
                cv2.line(frame, (center_x, h), center, color, 2)
            
            cv2.imshow("YOLO Detection", frame)
            cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DockingGuid()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
