#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
import numpy as np
import open3d as o3d
import sensor_msgs_py.point_cloud2 as pc2
from scipy.spatial.distance import cdist

from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

class DockingTargetDetector(Node):
    def __init__(self):
        super().__init__('docking_target_detector')
        self.create_subscription(PointCloud2, "/preprocessed_points", self.point_callback, 10)
        self.marker_publisher = self.create_publisher(MarkerArray, "/docking_visualization", 10)

        # --- 안정화를 위한 추적 및 필터링 변수 ---
        self.tracked_pillars = {} # {persistent_id: {'center': ..., 'corners': ..., 'last_seen': ...}}
        self.next_persistent_id = 0
        self.ASSOCIATION_THRESHOLD = 0.5 # 같은 기둥으로 판단할 최대 거리 (m)
        
        # 지수 이동 평균 필터의 스무딩 계수 (alpha). 낮을수록 부드러워짐 (0~1)
        self.SMOOTHING_FACTOR = 0.3
        
        self.get_logger().info("도킹 목표 검출 노드가 시작되었습니다. (안정화 필터 적용)")

    def point_callback(self, msg: PointCloud2):
        current_time = self.get_clock().now()
        points_struct = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points_np = np.array([[p[0], p[1], p[2]] for p in points_struct], dtype=np.float32)
        
        if points_np.shape[0] < 10: return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        labels = np.array(pcd.cluster_dbscan(eps=0.25, min_points=10, print_progress=False))
        unique_labels = np.unique(labels)
        unique_labels = unique_labels[unique_labels != -1]
        
        # 1. 현재 프레임에서 유효한 기둥 후보 찾기
        current_pillars = []
        for label in unique_labels:
            indices = np.where(labels == label)[0]
            cluster_pcd = pcd.select_by_index(indices)
            if len(cluster_pcd.points) < 10: continue

            try:
                box = cluster_pcd.get_oriented_bounding_box(robust=True)
                box_extent = box.extent
                width = min(box_extent[0], box_extent[1])
                length = max(box_extent[0], box_extent[1])
                if width < 1e-4: continue
                aspect_ratio = length / width
                if aspect_ratio > 2.5:
                    current_pillars.append({'label': label, 'box': box})
            except Exception:
                continue

        # 2. 추적: 현재 기둥 후보들을 이전 프레임의 기둥들과 매칭
        unmatched_pillars = list(range(len(current_pillars)))
        persistent_id_map = {} # {current_pillar_index: persistent_id}

        if self.tracked_pillars and current_pillars:
            old_centroids = np.array([p['center'] for p in self.tracked_pillars.values()])
            new_centroids = np.array([p['box'].get_center() for p in current_pillars])
            distances = cdist(old_centroids, new_centroids)

            old_ids = list(self.tracked_pillars.keys())
            
            for _ in range(min(len(old_ids), len(current_pillars))):
                if distances.size == 0 or np.min(distances) > self.ASSOCIATION_THRESHOLD: break
                
                old_idx, new_idx = np.unravel_index(np.argmin(distances), distances.shape)
                pid = old_ids[old_idx]
                
                if new_idx in unmatched_pillars:
                    persistent_id_map[new_idx] = pid
                    unmatched_pillars.remove(new_idx)
                    
                    # --- 3. 지수 이동 평균 필터 적용 ---
                    current_corners = np.asarray(current_pillars[new_idx]['box'].get_box_points())
                    prev_corners = self.tracked_pillars[pid]['corners']
                    
                    smoothed_corners = (self.SMOOTHING_FACTOR * current_corners) + \
                                       ((1 - self.SMOOTHING_FACTOR) * prev_corners)
                    
                    self.tracked_pillars[pid]['corners'] = smoothed_corners
                    self.tracked_pillars[pid]['center'] = np.mean(smoothed_corners, axis=0)
                    self.tracked_pillars[pid]['last_seen'] = current_time

                distances[old_idx, :] = np.inf
                distances[:, new_idx] = np.inf
        
        # 4. 새로운 기둥 등록
        for new_idx in unmatched_pillars:
            pid = self.next_persistent_id
            persistent_id_map[new_idx] = pid
            new_corners = np.asarray(current_pillars[new_idx]['box'].get_box_points())
            self.tracked_pillars[pid] = {
                'corners': new_corners,
                'center': np.mean(new_corners, axis=0),
                'last_seen': current_time
            }
            self.next_persistent_id += 1

        # 5. 오래된 기둥 제거
        self.prune_old_tracks(current_time)
        
        # 6. 안정화된 데이터를 이용해 시각화 및 중점 계산
        marker_array = MarkerArray()
        delete_marker = Marker(action=Marker.DELETEALL)
        marker_array.markers.append(delete_marker)

        # 추적 중인 기둥들을 y좌표 기준으로 정렬
        sorted_tracked_pillars = sorted(self.tracked_pillars.values(), key=lambda p: p['center'][1])
        
        if len(sorted_tracked_pillars) >= 2:
            # 모든 기둥 상자 그리기
            for i, p_data in enumerate(sorted_tracked_pillars):
                marker_array.markers.append(self.create_box_marker(msg.header, p_data['corners'], i))

            # 모든 이웃 쌍에 대해 중점 계산
            for i in range(len(sorted_tracked_pillars) - 1):
                p1_center = sorted_tracked_pillars[i]['center']
                p2_center = sorted_tracked_pillars[i+1]['center']
                p1_corners = sorted_tracked_pillars[i]['corners']
                p2_corners = sorted_tracked_pillars[i+1]['corners']

                p1_inner_center = self.get_inner_face_center(p1_corners, p2_center)
                p2_inner_center = self.get_inner_face_center(p2_corners, p1_center)
                target_point = (p1_inner_center + p2_inner_center) / 2.0
                marker_array.markers.append(self.create_target_marker(msg.header, target_point, i))
        
        self.marker_publisher.publish(marker_array)

    def get_inner_face_center(self, corners, other_pillar_center):
        distances = cdist(corners, [other_pillar_center]).flatten()
        inner_indices = np.argsort(distances)[:4]
        return np.mean(corners[inner_indices], axis=0)
        
    def prune_old_tracks(self, current_time):
        timeout = Duration(seconds=1.0)
        ids_to_remove = [pid for pid, obj in self.tracked_pillars.items() if (current_time - obj['last_seen']) > timeout]
        for pid in ids_to_remove: del self.tracked_pillars[pid]

    def create_box_marker(self, header, corners, marker_id):
        # (이전과 동일, ID 부분을 color_index 대신 marker_id로 명확화)
        marker_box = Marker()
        marker_box.header = header
        marker_box.ns = f"pillar_box"
        marker_box.id = marker_id
        marker_box.type = Marker.LINE_LIST
        marker_box.action = Marker.ADD
        marker_box.scale.x = 0.01
        marker_box.color = self.get_color_by_id(marker_id)
        c = corners
        lines = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7], [7,4], [0,4], [1,5], [2,6], [3,7]]
        for line in lines:
            marker_box.points.append(Point(x=c[line[0]][0], y=c[line[0]][1], z=c[line[0]][2]))
            marker_box.points.append(Point(x=c[line[1]][0], y=c[line[1]][1], z=c[line[1]][2]))
        return marker_box

    def create_target_marker(self, header, target_point, marker_id):
        # (이전과 동일)
        marker_target = Marker()
        marker_target.header = header
        marker_target.ns = "docking_targets"
        marker_target.id = marker_id
        marker_target.type = Marker.SPHERE
        marker_target.action = Marker.ADD
        marker_target.scale.x, marker_target.scale.y, marker_target.scale.z = 0.1, 0.1, 0.1
        marker_target.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        marker_target.pose.position = Point(x=target_point[0], y=target_point[1], z=target_point[2])
        return marker_target

    def get_color_by_id(self, index):
        colors = [
            ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0), ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0),
            ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0), ColorRGBA(r=1.0, g=0.0, b=1.0, a=1.0),
            ColorRGBA(r=0.0, g=1.0, b=1.0, a=1.0), ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0),
        ]
        return colors[index % len(colors)]

def main(args=None):
    # NameError를 피하기 위해 Duration import 추가
    from rclpy.duration import Duration
    rclpy.init(args=args)
    node = DockingTargetDetector()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
