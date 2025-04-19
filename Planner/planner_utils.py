import scipy
import os, math
import torch
import numpy as np
import random
import logging
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely import Point, LineString
from shapely.geometry.base import CAP_STYLE
import matplotlib.pyplot as plt
from common_utils import *
from .refinement import RefinementPlanner
from .refinement_sl import RefinementPlanner_sl
from .refinement_sl_noobs import RefinementPlanner_sl_noobs
from .occupancy_adapter import occupancy_grid
from .bezier_path import compute_control_points_and_trajectory
from .observation_addref import *
from .cubic_spline import *
from shapely.geometry import Polygon

from nuplan.planning.simulation.path.path import AbstractPath
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from nuplan.common.maps.abstract_map_objects import RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData, TrafficLightStatusType
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.metrics.utils.expert_comparisons import principal_value


class TrajectoryPlanner:
    def __init__(self, device='cpu'):
        self.N = int(T/DT)
        self.ts = DT
        self._device = device
        self.planner = RefinementPlanner(device)
        self.planner_sl = RefinementPlanner_sl(device)
        self.planner_sl_noobs = RefinementPlanner_sl_noobs(device)
        self.save_path = './plots'
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def select_best_ref_path(self, plan, ref_paths):
        min_avg_distance = float('inf')
        best_ref_path = None

        # 遍历每一条ref_path
        for ref_path in ref_paths:
            # 计算plan每个点到ref_path的所有点的距离矩阵,选了前30个
            distance_to_ref = scipy.spatial.distance.cdist(plan[:30, :2], ref_path[:, :2])
            
            # 找到距离最近的ref_path点的索引
            i = np.argmin(distance_to_ref, axis=1)
            
            # 提取与plan匹配的ref_path上的点
            #closest_ref_points = ref_path[i, :3]
            
            # 计算平均距离
            avg_distance = np.mean(distance_to_ref[np.arange(30), i])
            
            # 更新最优ref_path
            if avg_distance < min_avg_distance:
                min_avg_distance = avg_distance
                best_ref_path = ref_path
        
        return best_ref_path
    
    def plan(self, ego_state, ego_state_transformed, neighbors_state_transformed, 
             predictions, plan, scores, ref_paths, observation, features, iteration, map_api, route_roadblock_ids, traffic_light_data, debug=False):
        # Get the plan from the prediction model
        plan = plan[0].cpu().numpy()

        # Get the plan in the reference path
        if ref_paths is not None:
            best_ref_path = self.select_best_ref_path(plan, ref_paths)

            frenet_paths = np.array([transform_to_Frenet(traj, best_ref_path) for traj in ref_paths])

            left_bound, right_bound = get_sampling_boundaries(frenet_paths)


            ref_path_frenet = transform_to_Frenet(best_ref_path, best_ref_path)


            lateral_sampling_frenet=lateral_sampling_in_frenet(ref_path_frenet, left_bound, right_bound, num_samples=20)


            ego_state_frenet = ref_path_frenet[0]

            bezier_path = generate_combined_paths(ego_state_frenet , lateral_sampling_frenet)


            lateral_sampling = transform_all_paths_to_Cartesian(bezier_path, best_ref_path)

            lateral_sampling_world_frame = transform_all_paths_to_world_frame(lateral_sampling, ego_state)
            select_path_world_frame = find_top_k_paths(lateral_sampling_world_frame,  traffic_light_data, ego_state, map_api, route_roadblock_ids, observation)
            select_path = transform_all_paths_to_ego_frame(select_path_world_frame, ego_state)


            best_lateral_sampling = self.select_best_ref_path(plan, select_path)
            
            
            dx = np.diff(best_lateral_sampling[:, 0])
            dy = np.diff(best_lateral_sampling[:, 1])
            yaw = np.arctan2(dy, dx)

            yaw = np.append(yaw, yaw[-1])

            best_lateral_sampling_frenet = transform_to_Frenet(best_lateral_sampling, best_ref_path)

            all_obstacles_coordinates = extract_polygon_coordinates(ego_state, observation, best_ref_path)


            if all_obstacles_coordinates is not None:

                occ_map_obstacles = occupancy_from_obstacles(all_obstacles_coordinates)

                ego_state_transformed_sl = ego_state_transformed.to(self._device)
                best_lateral_sampling_frenet_sl = torch.from_numpy(best_lateral_sampling_frenet).unsqueeze(0).to(self._device)
                occ_map_obstacles = torch.from_numpy(occ_map_obstacles).unsqueeze(0).to(self._device)
                ref_path_sl = torch.from_numpy(ref_path_frenet).unsqueeze(0).to(self._device)
                left_bound = torch.from_numpy(left_bound).unsqueeze(0).to(self._device)
                right_bound = torch.from_numpy(right_bound).unsqueeze(0).to(self._device)
                yaw = torch.from_numpy(yaw).unsqueeze(0).to(self._device)

                
                sl= self.planner_sl.plan(ego_state_transformed_sl, best_lateral_sampling_frenet_sl, occ_map_obstacles, ref_path_sl, left_bound, right_bound, yaw)

            else:
                ego_state_transformed_sl = ego_state_transformed.to(self._device)
                best_lateral_sampling_frenet_sl = torch.from_numpy(best_lateral_sampling_frenet).unsqueeze(0).to(self._device)
                ref_path_sl = torch.from_numpy(ref_path_frenet).unsqueeze(0).to(self._device)
                left_bound = torch.from_numpy(left_bound).unsqueeze(0).to(self._device)
                right_bound = torch.from_numpy(right_bound).unsqueeze(0).to(self._device)
                yaw = torch.from_numpy(yaw).unsqueeze(0).to(self._device)
                sl = self.planner_sl_noobs.plan(ego_state_transformed_sl, best_lateral_sampling_frenet_sl, ref_path_sl, left_bound, right_bound, yaw)

            sl = sl.squeeze(0).cpu().numpy()

            path_optimized = transform_to_Cartesian_path(sl, best_ref_path)
            
            best_lateral_sampling= post_process(path_optimized,ego_state, map_api, route_roadblock_ids, traffic_light_data)



            distance_to_ref = scipy.spatial.distance.cdist(plan[:, :2], best_lateral_sampling[:, :2])
            i = np.argmin(distance_to_ref, axis=1)
            plan = best_lateral_sampling[i, :3]
            s = np.concatenate([[0], i]) * 0.1
            speed = np.diff(s) / DT

        else:
            speed = np.diff(plan[:, :2], axis=0) / DT
            speed = np.linalg.norm(speed, axis=-1)
            speed = np.concatenate([speed, [speed[-1]]])
            
        # Refine planning
        if ref_paths is None:
            print("no ref")
            pass
        else:

            occupancy = occupancy_grid(predictions[0], scores[0, 1:], neighbors_state_transformed[0], best_ref_path)
            ego_plan_ds = torch.from_numpy(speed).float().unsqueeze(0).to(self._device)
            ego_plan_s = torch.from_numpy(s).float().unsqueeze(0).to(self._device)
            ego_state_transformed = ego_state_transformed.to(self._device)
            best_ref_path = torch.from_numpy(best_ref_path).unsqueeze(0).to(self._device)
            occupancy = torch.from_numpy(occupancy).unsqueeze(0).to(self._device)

            s, speed = self.planner.plan(ego_state_transformed, ego_plan_ds, ego_plan_s, occupancy, best_ref_path)
            s = s.squeeze(0).cpu().numpy()
            speed = speed.squeeze(0).cpu().numpy()

            # Convert to Cartesian trajectory
            best_ref_path = best_ref_path.squeeze(0).cpu().numpy()
            i = (s * 10).astype(np.int32).clip(0, len(best_ref_path)-1)
            plan = best_ref_path[i, :3]
            

        return plan



    @staticmethod
    def transform_to_Cartesian_path(path, ref_path):
        frenet_idx = np.array(path[:, 0] * 10, dtype=np.int32)
        frenet_idx = np.clip(frenet_idx, 0, len(ref_path)-1)
        ref_points = ref_path[frenet_idx]
        l = path[frenet_idx, 1]

        cartesian_x = ref_points[:, 0] - l * np.sin(ref_points[:, 2])
        cartesian_y = ref_points[:, 1] + l * np.cos(ref_points[:, 2])
        cartesian_path = np.column_stack([cartesian_x, cartesian_y])

        return cartesian_path

def transform_to_Frenet(traj, ref_path, only_l = False):
    distance_to_ref_path = scipy.spatial.distance.cdist(traj[:, :2], ref_path[:, :2])
    frenet_idx = np.argmin(distance_to_ref_path, axis=-1)
    ref_points = ref_path[frenet_idx]
    interval = 0.1

    frenet_s = interval * frenet_idx
    e = np.sign((traj[:, 1] - ref_points[:, 1]) * np.cos(ref_points[:, 2]) - (traj[:, 0] - ref_points[:, 0]) * np.sin(ref_points[:, 2]))
    frenet_l = np.linalg.norm(traj[:, :2] - ref_points[:, :2], axis=-1) * e 

    if only_l is True:
        return frenet_l

    if traj.shape[-1] == 6:
        frenet_h = principal_value(ref_points[:, 2] - traj[:, 2])
        frenet_traj = np.column_stack([frenet_s, frenet_l, frenet_h])
    else:
        frenet_traj = np.column_stack([frenet_s, frenet_l])

    return frenet_traj

def transform_obstacles_to_frenet(all_obstacles_coordinates, ref_path):

    all_frenet_obstacles = []
    
    for obstacle_coords in all_obstacles_coordinates:
        
        frenet_coords = transform_to_Frenet(obstacle_coords, ref_path)
        
        
        if np.all(frenet_coords[:, 0] == 0):
            continue
        
        
        if np.all(frenet_coords[:, 0] > 100):
            continue
        
       
        if np.any(np.abs(frenet_coords[:, 1]) > 10):
            continue

        
        all_frenet_obstacles.append(frenet_coords)
    
   
    if len(all_frenet_obstacles) > 0:
        all_frenet_obstacles = np.array(all_frenet_obstacles)
    else:
        all_frenet_obstacles = None  
    
    return all_frenet_obstacles


def transform_to_Cartesian_path(path, ref_path):

    l = path[:, 1]

    cartesian_x = ref_path[:, 0] - l * np.sin(ref_path[:, 2])
    cartesian_y = ref_path[:, 1] + l * np.cos(ref_path[:, 2])
    cartesian_path = np.column_stack([cartesian_x, cartesian_y])

    return cartesian_path


def transform_all_paths_to_Cartesian(sampled_paths, ref_path):

    cartesian_paths = []
    
    for path in sampled_paths:
        cartesian_path = transform_to_Cartesian_path(path, ref_path)
        cartesian_paths.append(cartesian_path)
    
    return np.array(cartesian_paths)

def get_sampling_boundaries(frenet_paths):

    lateral_offsets = frenet_paths[:, :, 1]
    
    
    left_bound = np.max(lateral_offsets, axis=(0)) 
    right_bound = np.min(lateral_offsets, axis=(0))  
    return left_bound, right_bound

def lateral_sampling_in_frenet(frenet_ref_path, left_bound, right_bound, num_samples=20, lateral_offset=0.1):
    
    sampled_paths = []

    # Include the original reference path first
    sampled_paths.append(frenet_ref_path.copy())

    # Perform sampling on both sides of the reference path
    for i in range(1, num_samples + 1):
        # Positive lateral offset (to the left of the reference path)
        sampled_left = frenet_ref_path.copy()
        sampled_left[:, 1] += i * lateral_offset
        sampled_left[:, 1] = np.clip(sampled_left[:, 1], right_bound, left_bound)  
        sampled_paths.append(sampled_left)

        # Negative lateral offset (to the right of the reference path)
        sampled_right = frenet_ref_path.copy()
        sampled_right[:, 1] -= i * lateral_offset
        sampled_right[:, 1] = np.clip(sampled_right[:, 1], right_bound, left_bound) 
        sampled_paths.append(sampled_right)
    sampled_paths_array = np.array(sampled_paths)

    return sampled_paths_array

def generate_combined_paths(ego_state, sampled_paths):
 
    combined_paths = []
    
    for path in sampled_paths:
        for idx in [100, 150, 200, 250]:
            ex, ey = path[idx][:2]
            eyaw = path[idx][2]  
            bezier_path = compute_control_points_and_trajectory(ego_state[0], ego_state[1], ego_state[2], ex, ey, eyaw, 3, idx)[0]

            combined_path = np.concatenate([bezier_path, path[idx:, :2]])
            combined_paths.append(combined_path)
    combined_paths = np.array(combined_paths)


    return combined_paths

def annotate_speed1(ref_path, speed_limit):

    speed = np.ones(len(ref_path)) * speed_limit
    

    turning_idx = np.argmax(np.abs(ref_path[:, 3]) > 1/10)

    
    if turning_idx > 0:
       
        end_turning_idx = turning_idx + np.argmax(np.abs(ref_path[turning_idx:, 3]) < 1/10)

       
        if end_turning_idx == turning_idx:
            end_turning_idx = len(ref_path)

        
        speed[turning_idx:end_turning_idx] = 3

   
    return speed[:, None]

def post_process(best_lateral_sampling, ego_state, map_api, route_roadblock_ids, traffic_light_data):


    x = best_lateral_sampling[:, 0]
    y = best_lateral_sampling[:, 1]
    ryaw, rk = calculate_yaw_and_curvature(x, y)
   
    spline_path = np.stack([x, y, ryaw, rk], axis=1)

    if spline_path.shape[0] < MAX_LEN * 10:
        spline_path = np.append(spline_path, np.repeat(spline_path[np.newaxis, -1], MAX_LEN*10-len(spline_path), axis=0), axis=0)
          
    elif spline_path.shape[0] > MAX_LEN * 10:
        spline_path = spline_path[:MAX_LEN * 10]

    starting_block = None
    min_target_speed = 3
    max_target_speed = 15
    closest_distance = math.inf
    cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
    candidate_lane_edge_ids,route_roadblocks = initialize_route_plan(map_api, route_roadblock_ids)
    for block in route_roadblocks:
        for edge in block.interior_edges:
            distance = edge.polygon.distance(Point(cur_point))
            if distance < closest_distance:
                starting_block = block

    occupancy = np.zeros(shape=(spline_path.shape[0], 1))
    for data in traffic_light_data:
        id_ = str(data.lane_connector_id)
        if data.status == TrafficLightStatusType.RED and id_ in candidate_lane_edge_ids:
            lane_conn = map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
            conn_path = lane_conn.baseline_path.discrete_path
            conn_path = np.array([[p.x, p.y] for p in conn_path])
            red_light_lane = transform_to_ego_frame(conn_path, ego_state)
            
            occupancy = annotate_occupancy(occupancy, spline_path, red_light_lane)

    target_speed = 13.0 
    target_speed = starting_block.interior_edges[0].speed_limit_mps or target_speed
    target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
    
    max_speed = annotate_speed1(spline_path, target_speed)
    best_lateral_sampling = np.concatenate([spline_path, max_speed, occupancy], axis=-1) 

    


    return best_lateral_sampling  

def calc_spline_course_refpath(x, y, ds=0.1):
    if len(x) < 2 or len(y) < 2:
        return None, None, None, None

    xy = np.column_stack((x, y))

    _, unique_indices = np.unique(xy, axis=0, return_index=True)


    xy_unique = xy[np.sort(unique_indices)]


    x_unique = xy_unique[:, 0]
    y_unique = xy_unique[:, 1]
    
    sp = Spline2D(x_unique, y_unique)

    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        rk.append(sp.calc_curvature(i_s))

    return rx, ry, ryaw, rk


def transform_all_paths_to_world_frame(paths, ego_state):
    world_paths = []
    for path in paths:
        
        world_path = transform_to_world_frame(path, ego_state)
        world_paths.append(world_path)

    return np.array(world_paths)


def transform_to_world_frame(path, ego_state):
    
    ego_x, ego_y, ego_h = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading


    ego_path_x, ego_path_y = path[:, 0], path[:, 1]


    world_path_x = np.cos(ego_h) * ego_path_x - np.sin(ego_h) * ego_path_y + ego_x
    world_path_y = np.sin(ego_h) * ego_path_x + np.cos(ego_h) * ego_path_y + ego_y


    world_path = np.column_stack((world_path_x, world_path_y))

    return world_path


def transform_all_paths_to_ego_frame(paths, ego_state):
    ego_paths = []
    for path in paths:
        
        ego_path = transform_to_ego_frame(path, ego_state)
        ego_paths.append(ego_path)
    

    return np.array(ego_paths)


def calculate_path_curvature(path):
    dx = np.gradient(path[:, 0])
    dy = np.gradient(path[:, 1])
    d2x = np.gradient(dx)
    d2y = np.gradient(dy)
    curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**(3/2)

    return curvature

def check_target_lane(path, vehicles):

    
    expanded_path = LineString(path).buffer((WIDTH/2), cap_style=CAP_STYLE.square)
    min_distance_to_vehicles = np.inf

    for v in vehicles:
        d = expanded_path.distance(v.geometry)
        if d < min_distance_to_vehicles:
            min_distance_to_vehicles = d

    if min_distance_to_vehicles < 5:
        return 0

    return 1

def check_obstacles(path, obstacles):
    expanded_path = LineString(path).buffer((WIDTH/2), cap_style=CAP_STYLE.square)

    for obstacle in obstacles:
        obstacle_polygon = obstacle.geometry
        
        if expanded_path.intersects(obstacle_polygon):
            return 1

    return 0


def check_out_boundary(polyline, path):
    line = LineString(polyline).buffer((WIDTH/2), cap_style=CAP_STYLE.square)

    for edge in path:
        left, right = edge.adjacent_edges
        if (left is None and line.intersects(edge.left_boundary.linestring)) or \
            (right is None and line.intersects(edge.right_boundary.linestring)):
            return 1

    return 0


def find_top_k_paths(paths, traffic_light_data, ego_state, map_api, route_roadblock_ids, observation, k=10):
    path_costs = []

    for path in paths:
        
        cost = calculate_cost(path, traffic_light_data, ego_state, map_api, route_roadblock_ids, observation)
        
        
        path_costs.append((cost, path))
    
    
    path_costs.sort(key=lambda x: x[0])

    top_k_paths = [path for cost, path in path_costs[:k]]
    
    
    top_k_paths_array = np.array([path for path in top_k_paths])
    

    return top_k_paths_array


def extract_polygon_coordinates(ego_state, observation, best_ref_path):

    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BARRIER,
                    TrackedObjectType.CZONE_SIGN, TrackedObjectType.TRAFFIC_CONE,
                    TrackedObjectType.GENERIC_OBJECT]
    objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)
    obstacles = []
    vehicles = []
    for obj in objects:
        if obj.box.geometry.distance(ego_state.car_footprint.geometry) > 30:
            continue

        if obj.tracked_object_type == TrackedObjectType.VEHICLE:
            if obj.velocity.magnitude() < 0.01:
                obstacles.append(obj.box)
            else:
                vehicles.append(obj.box)
        else:
            obstacles.append(obj.box)
    
    all_coordinates = []
    
    for obstacle in obstacles:
        obstacle_polygon = obstacle.geometry 
        coords = np.array(obstacle_polygon.exterior.coords)  
        all_coordinates.append(coords)
    
   
    if len(all_coordinates) == 0:
        return None
    else:
        
        all_coordinates = np.stack(all_coordinates, axis=0)
        all_coordinates_ego_frame = transform_all_paths_to_ego_frame(all_coordinates, ego_state)
        
        all_frenet_obstacles = transform_obstacles_to_frenet(all_coordinates_ego_frame, best_ref_path)
        if all_frenet_obstacles is not None:

            return all_frenet_obstacles
        else:

            return None

        
    
def plot_obstacle_coordinates(all_coordinates):
    if all_coordinates is None:
        print("No obstacle coordinates to plot.")
        return

    plt.figure(figsize=(10, 8))
    

    for coords in all_coordinates:
        plt.plot(coords[:, 0], coords[:, 1], '-o', label='Obstacle')
    
    s_min = np.min(all_coordinates[:, :, 0])
    s_max = np.max(all_coordinates[:, :, 0])

    plt.xlabel('Frenet s')
    plt.ylabel('Frenet l')
    plt.title('Frenet Coordinates of Obstacles')

    
    plt.xlim([s_min, s_max])
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()



def occupancy_from_obstacles(all_frenet_obstacles, max_len=120, plot=False):

    occupancy_map = np.zeros((max_len * 10, 200))  

    
    for obstacle_coords in all_frenet_obstacles:
        
        obstacle_polygon = Polygon(obstacle_coords)

        min_s = np.clip(np.min(obstacle_coords[:, 0]), 0, max_len)
        max_s = np.clip(np.max(obstacle_coords[:, 0]), 0, max_len)
        min_l = np.clip(np.min(obstacle_coords[:, 1]), -10, 10)
        max_l = np.clip(np.max(obstacle_coords[:, 1]), -10, 10)

        
        for i in range(int(min_s * 10), int(max_s * 10)):  
            s = i / 10.0  
            for l_idx in range(200):  
                l = (l_idx - 100) * 0.1  
                
                
                if min_l <= l <= max_l:
                    
                    occupancy_map[i, l_idx] = 1  

    
    if plot:
        plt.figure(figsize=(12, 6))
        
        
        plt.imshow(occupancy_map.T, cmap='hot', interpolation='nearest', aspect='auto', extent=[0, max_len, 10, -10])
        
        
        plt.colorbar(label='Occupancy')
        plt.xlabel('Frenet s coordinate (distance along path)')
        plt.ylabel('Frenet l coordinate (lateral offset)')
        plt.title('Occupancy Map in Frenet Frame (s: 0-120m, l: -10m to 10m)')
        plt.grid(True)
        plt.show()
    
    return occupancy_map

def calculate_cost(path,  traffic_light_data, ego_state, map_api, route_roadblock_ids, observation):


    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BARRIER,
                    TrackedObjectType.CZONE_SIGN, TrackedObjectType.TRAFFIC_CONE,
                    TrackedObjectType.GENERIC_OBJECT]
    objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)
    obstacles = []
    vehicles = []
    for obj in objects:
        if obj.box.geometry.distance(ego_state.car_footprint.geometry) > 30:
            continue

        if obj.tracked_object_type == TrackedObjectType.VEHICLE:
            if obj.velocity.magnitude() < 0.01:
                obstacles.append(obj.box)
            else:
                vehicles.append(obj.box)
        else:
            obstacles.append(obj.box)

    curvature = calculate_path_curvature(path[:100])#计算弯曲程度
    curvature = np.max(curvature)


    # go to the target lane
    target = check_target_lane(path[:50], vehicles)

    # check obstacles
    obstacles = check_obstacles(path[:200], obstacles)
    
    # final cost
    #cost = 10 * obstacles + 2 * out_boundary + 1 * lane_change  + 0.1 * curvature - 5 * target
    cost = 10 * obstacles   + 0.1 * curvature - 5 * target

    return cost



def compute_frenet_differences(ref_path_frenet, sl):

    ref_s = ref_path_frenet[:, 0]
    ref_l = ref_path_frenet[:, 1]

    
    opt_s = sl[:, 0]
    opt_l = sl[:, 1]

    
    s_diff = ref_s - opt_s
    l_diff = ref_l - opt_l
    
    
    s_diff = np.sum(s_diff)
    l_diff = np.sum(l_diff)



    return s_diff, l_diff




def extend_trajectory(x, y, extend_len=10):

    last_dx = x[-1] - x[-2]
    last_dy = y[-1] - y[-2]

    for i in range(extend_len):
        x = np.append(x, x[-1] + last_dx)
        y = np.append(y, y[-1] + last_dy)

    return x, y

def calculate_yaw_and_curvature(x, y):

    
    dx = np.diff(x)
    dy = np.diff(y)

    
    ddx = np.diff(dx)
    ddy = np.diff(dy)

   
    if len(dx) < 1200:
        x_extended, y_extended = extend_trajectory(x, y, 1200 - len(dx) - 1)
        dx = np.diff(x_extended)
        dy = np.diff(y_extended)
        ddx = np.diff(dx)
        ddy = np.diff(dy)

    
    yaw = np.arctan2(dy, dx)

    
    curvature = (ddy * dx[:-1] - ddx * dy[:-1]) / ((dx[:-1] ** 2 + dy[:-1] ** 2) ** (3 / 2))

    
    yaw = np.pad(yaw, (0, 1200 - len(yaw)), 'edge')
    curvature = np.pad(curvature, (0, 1200 - len(curvature)), 'edge')

    return yaw, curvature