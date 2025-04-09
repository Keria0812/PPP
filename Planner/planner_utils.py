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
            #print(best_ref_path.shape)#(1200, 6)

            #ego_state = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading
            #print(ego_state)#(664747.3499715761, 3999453.7245545867, -2.0259242529133137) 
            '''确定采样边界'''
            frenet_paths = np.array([transform_to_Frenet(traj, best_ref_path) for traj in ref_paths])
            #print(frenet_paths.shape)#（3，1200，3）
            left_bound, right_bound = get_sampling_boundaries(frenet_paths)
            #print(left_bound)#（1200，）每个s上的最大最小l值
            #print(right_bound)

            ref_path_frenet = transform_to_Frenet(best_ref_path, best_ref_path)
            #print(ref_path_frenet.shape)#(1200,3),一开始是（0，0，0）

            lateral_sampling_frenet=lateral_sampling_in_frenet(ref_path_frenet, left_bound, right_bound, num_samples=20)#横向采样41条轨迹，输出frenet坐标系下的坐标
            #print(lateral_sampling_frenet.shape)#(41, 1200, 3)

            ego_state_frenet = ref_path_frenet[0]

            bezier_path = generate_combined_paths(ego_state_frenet , lateral_sampling_frenet)#横向采样，按10，15，20,25m采样

            #lateral_sampling = transform_all_paths_to_Cartesian(lateral_sampling_frenet, best_ref_path)#把41条轨迹转回笛卡尔坐标系
            lateral_sampling = transform_all_paths_to_Cartesian(bezier_path, best_ref_path)#把21条轨迹转回笛卡尔坐标系
            #print(lateral_sampling.shape)#(164, 1200, 2)41*4

            lateral_sampling_world_frame = transform_all_paths_to_world_frame(lateral_sampling, ego_state)#转回世界坐标系
            #print(lateral_sampling_world_frame)
            select_path_world_frame = find_top_k_paths(lateral_sampling_world_frame,  traffic_light_data, ego_state, map_api, route_roadblock_ids, observation)#选择10条cost最低的轨迹
            #print(select_path_world_frame.shape)#(10, 1200, 2)
            select_path = transform_all_paths_to_ego_frame(select_path_world_frame, ego_state)#转回自车坐标系

            #用规则方法筛选轨迹

            best_lateral_sampling = self.select_best_ref_path(plan, select_path)#用距离筛选出10条轨迹中最近的一条
            
            # 计算每个点的方向角度（yaw），假设以弧度表示
            dx = np.diff(best_lateral_sampling[:, 0])
            dy = np.diff(best_lateral_sampling[:, 1])
            yaw = np.arctan2(dy, dx)

            # 将 yaw 值的形状从 (1199,) 扩展到 (1200,) 并添加到轨迹中
            # 这里我们将最后一个 yaw 角度值重复添加到最后一个点上，以保证与其他点对齐
            yaw = np.append(yaw, yaw[-1])
            '''
            best_lateral_sampling = np.column_stack((best_lateral_sampling, yaw))
            # 选择 best_ref_path 的后三个维度
            additional_info = best_ref_path[:, 3:]

            # 将 additional_info 添加到 best_lateral_sampling 中
            best_lateral_sampling = np.concatenate((best_lateral_sampling, additional_info), axis=1)
            '''
            best_lateral_sampling_frenet = transform_to_Frenet(best_lateral_sampling, best_ref_path)
            #print(lane_error)#[0.         0.0003533  0.0007198  ... 0.20005542 0.20005542 0.20005542]
            #print(best_lateral_sampling_frenet.shape)#(1200,2)
            all_obstacles_coordinates = extract_polygon_coordinates(ego_state, observation, best_ref_path)#提取静态障碍物坐标,将静态障碍物转化为frenet，并过滤掉s<=0，s>30, l>10的点
            #print(all_obstacles_coordinates.shape)#(32,5,2)
            #plot_obstacle_coordinates(all_obstacles_coordinates)

            if all_obstacles_coordinates is not None:
                #all_frenet_obstacles = transform_obstacles_to_frenet(all_obstacles_coordinates, best_ref_path)#将静态障碍物转化为frenet，并过滤掉s<=0，s>30, l>10的点
                #if all_frenet_obstacles is not None:
                #plot_obstacle_coordinates(all_frenet_obstacles)#可视化
                #print("have_obstacles")
                occ_map_obstacles = occupancy_from_obstacles(all_obstacles_coordinates)#障碍物occ图
                #print(occ_map_obstacles.shape)#(1200,200)
                ego_state_transformed_sl = ego_state_transformed.to(self._device)
                best_lateral_sampling_frenet_sl = torch.from_numpy(best_lateral_sampling_frenet).unsqueeze(0).to(self._device)
                occ_map_obstacles = torch.from_numpy(occ_map_obstacles).unsqueeze(0).to(self._device)
                ref_path_sl = torch.from_numpy(ref_path_frenet).unsqueeze(0).to(self._device)
                left_bound = torch.from_numpy(left_bound).unsqueeze(0).to(self._device)
                right_bound = torch.from_numpy(right_bound).unsqueeze(0).to(self._device)
                yaw = torch.from_numpy(yaw).unsqueeze(0).to(self._device)
                #print(f"best_ref_path_sl shape: {ref_path_sl.shape}") 
                '''
                print(f"ego_state_transformed_sl shape: {ego_state_transformed_sl.shape}")
                print(f"best_lateral_sampling_frenet_sl shape: {best_lateral_sampling_frenet_sl.shape}")
                print(f"all_frenet_obstacles_sl shape: {all_frenet_obstacles_sl.shape}")
                print(f"best_ref_path_sl shape: {best_ref_path_sl.shape}")
                go_state_transformed_sl shape: torch.Size([1, 7])                                                                                                                                                                                                                                         
                best_lateral_sampling_frenet_sl shape: torch.Size([1, 1200, 2])
                occ_map_obstacles shape: torch.Size([1, 1200,200])
                ref_path_sl shape: torch.Size([1, 1200, 3])    
                '''
                
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

            s_diff, l_diff = compute_frenet_differences(best_ref_path, best_lateral_sampling)
            logging.info(f"s 差值: {s_diff}")  

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
            

        if debug:
            best_mode = np.argmax(scores[0, 1:].cpu().numpy(), axis=-1)
            predictions = predictions[0].cpu().numpy()
            best_predictions = [predictions[i, best_mode[i], :, :2] for i in range(predictions.shape[0])]
            best_predictions_array = np.array(best_predictions)

            self.plot(iteration, features, plan, best_predictions_array, best_lateral_sampling, select_path, path_optimized)

        return plan

    def plot(self, iteration, env_inputs, ego_future, agents_future, best_ref_path, lateral_sampling, path_optimized):
        fig = plt.gcf()
        dpi = 100
        size_inches = 800 / dpi
        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_tight_layout(True)

        # plot map
        map_lanes = env_inputs['lanes'][0]
        for i in range(map_lanes.shape[0]):
            lane = map_lanes[i].cpu().numpy()
            if lane[0, 0] != 0:
                plt.plot(lane[:, 0], lane[:, 1], color="gray", linewidth=20, zorder=1)
                plt.plot(lane[:, 0], lane[:, 1], "k--", linewidth=1, zorder=2)

        map_crosswalks = env_inputs['crosswalks'][0]
        for crosswalk in map_crosswalks:
            pts = crosswalk.cpu().numpy()
            plt.plot(pts[:, 0], pts[:, 1], 'b:', linewidth=2)

        # plot ego
        front_length = get_pacifica_parameters().front_length
        rear_length = get_pacifica_parameters().rear_length
        width = get_pacifica_parameters().width
        rect = plt.Rectangle((0 - rear_length, 0 - width/2), front_length + rear_length, width, 
                             linewidth=2, color='r', alpha=0.9, zorder=3)
        plt.gca().add_patch(rect)

        # plot agents
        agents = env_inputs['neighbor_agents_past'][0]
        #print(agents.shape)#torch.Size([20, 21, 11])   
        for agent in agents:
            agent = agent[-1].cpu().numpy()
            if agent[0] != 0:
                rect = plt.Rectangle((agent[0] - agent[6]/2, agent[1] - agent[7]/2), agent[6], agent[7],
                                      linewidth=2, color='m', alpha=0.9, zorder=3,
                                      transform=mpl.transforms.Affine2D().rotate_around(*(agent[0], agent[1]), agent[2]) + plt.gca().transData)
                plt.gca().add_patch(rect)
                                    

        # plot ego and agents future trajectories
        #ego = ego_future.cpu().numpy()
        ego = ego_future
        #agents = agents_future.cpu().numpy()
        agents = agents_future
        plt.plot(ego[:, 0], ego[:, 1], color="r", linewidth=3)
        plt.gca().add_patch(plt.Circle((ego[29, 0], ego[29, 1]), 0.5, color="r", zorder=4))
        plt.gca().add_patch(plt.Circle((ego[79, 0], ego[79, 1]), 0.5, color="r", zorder=4))
        plt.plot(best_ref_path[:, 0], best_ref_path[:, 1], color="y", linewidth=3)
        plt.plot(path_optimized[:, 0], path_optimized[:, 1], color="k", linewidth=3)

        for agent in agents:
            if np.abs(agent[0, 0]) > 1:
                #print(agent[0, 0])
                #agent = trajectory_smoothing(agent)
                plt.plot(agent[:, 0], agent[:, 1], color="m", linewidth=3)
                plt.gca().add_patch(plt.Circle((agent[29, 0], agent[29, 1]), 0.5, color="m", zorder=4))
                plt.gca().add_patch(plt.Circle((agent[79, 0], agent[79, 1]), 0.5, color="m", zorder=4))

        # plot
        plt.gca().margins(0)  
        plt.gca().set_aspect('equal')
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axis([-100, 100, -100, 100])
        plt.show()
        plot_file = os.path.join(self.save_path, f"trajectory_plot_{iteration}.png")
        plt.savefig(plot_file)
        plt.close(fig)




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
        # 调用 transform_to_Frenet 将障碍物坐标转换为 Frenet 坐标
        frenet_coords = transform_to_Frenet(obstacle_coords, ref_path)
        
        # 丢弃 s 值全为 0 的障碍物
        if np.all(frenet_coords[:, 0] == 0):
            continue
        
        # 丢弃 s 值全大于 100 的障碍物
        if np.all(frenet_coords[:, 0] > 100):
            continue
        
        # 如果 l 值的绝对值中有任何一个大于 10，则丢弃
        if np.any(np.abs(frenet_coords[:, 1]) > 10):
            continue

        # 如果障碍物通过所有检查，则添加到结果列表中
        all_frenet_obstacles.append(frenet_coords)
    
    # 如果有数据，将结果转换为 numpy 数组
    if len(all_frenet_obstacles) > 0:
        all_frenet_obstacles = np.array(all_frenet_obstacles)
    else:
        all_frenet_obstacles = None  # 如果没有有效障碍物坐标，返回 None
    
    return all_frenet_obstacles


def transform_to_Cartesian_path(path, ref_path):

    l = path[:, 1]

    cartesian_x = ref_path[:, 0] - l * np.sin(ref_path[:, 2])
    cartesian_y = ref_path[:, 1] + l * np.cos(ref_path[:, 2])
    cartesian_path = np.column_stack([cartesian_x, cartesian_y])

    return cartesian_path


def transform_all_paths_to_Cartesian(sampled_paths, ref_path):
    """
    把横向采样的41条轨迹转换成笛卡尔坐标系
    Transform all sampled paths from Frenet to Cartesian coordinates.
    
    :param sampled_paths: A numpy array of sampled paths in Frenet coordinates (shape: [41, 1200, 3]).
    :param ref_path: The reference path in Cartesian coordinates (shape: [1200, 6]).
    
    :return: A numpy array of sampled paths in Cartesian coordinates (shape: [41, 1200, 2]).
    """
    cartesian_paths = []
    
    for path in sampled_paths:
        cartesian_path = transform_to_Cartesian_path(path, ref_path)
        cartesian_paths.append(cartesian_path)
    
    return np.array(cartesian_paths)

def get_sampling_boundaries(frenet_paths):
    """ 计算采样时的左右边界 """
    # 提取所有路径的横向偏移量 (l)
    lateral_offsets = frenet_paths[:, :, 1]
    
    # 计算最小的横向偏移量和最大的横向偏移量
    left_bound = np.max(lateral_offsets, axis=(0))  # 最大的横向偏移量
    right_bound = np.min(lateral_offsets, axis=(0))  # 最小的横向偏移量
    return left_bound, right_bound

def lateral_sampling_in_frenet(frenet_ref_path, left_bound, right_bound, num_samples=20, lateral_offset=0.1):
    """ 在给定边界内进行横向采样 """
    sampled_paths = []

    # Include the original reference path first
    sampled_paths.append(frenet_ref_path.copy())

    # Perform sampling on both sides of the reference path
    for i in range(1, num_samples + 1):
        # Positive lateral offset (to the left of the reference path)
        sampled_left = frenet_ref_path.copy()
        sampled_left[:, 1] += i * lateral_offset
        sampled_left[:, 1] = np.clip(sampled_left[:, 1], right_bound, left_bound)  # 确保在边界内
        sampled_paths.append(sampled_left)

        # Negative lateral offset (to the right of the reference path)
        sampled_right = frenet_ref_path.copy()
        sampled_right[:, 1] -= i * lateral_offset
        sampled_right[:, 1] = np.clip(sampled_right[:, 1], right_bound, left_bound)  # 确保在边界内
        sampled_paths.append(sampled_right)
    sampled_paths_array = np.array(sampled_paths)

    return sampled_paths_array

def generate_combined_paths(ego_state, sampled_paths):
    #横向采样
    combined_paths = []
    
    for path in sampled_paths:
        for idx in [100, 150, 200, 250]:#按10，15，20,25m采样

            ex, ey = path[idx][:2]
            eyaw = path[idx][2]  # 计算目标点的角度
            bezier_path = compute_control_points_and_trajectory(ego_state[0], ego_state[1], ego_state[2], ex, ey, eyaw, 3, idx)[0]
            #print(bezier_path.shape)
            #print(path.shape)
            combined_path = np.concatenate([bezier_path, path[idx:, :2]])
            combined_paths.append(combined_path)
    combined_paths = np.array(combined_paths)
    #print(combined_paths.shape)#(164, 1200, 2)

    return combined_paths

def annotate_speed1(ref_path, speed_limit):
    # 初始化一个速度数组，初始值为整个路径上的默认速度限制
    speed = np.ones(len(ref_path)) * speed_limit
    
    # 找到转弯点的索引，定义为路径中曲率超过1/10的第一个点
    turning_idx = np.argmax(np.abs(ref_path[:, 3]) > 1/10)

    # 如果找到了转弯点
    if turning_idx > 0:
        # 找到曲率恢复到小于1/10的第一个点
        end_turning_idx = turning_idx + np.argmax(np.abs(ref_path[turning_idx:, 3]) < 1/10)

        # 如果没有找到恢复点（即曲率在之后一直大于等于1/10），则将end_turning_idx设置为路径末尾
        if end_turning_idx == turning_idx:
            end_turning_idx = len(ref_path)

        # 将转弯点到恢复点之间的速度限制设置为3 m/s
        speed[turning_idx:end_turning_idx] = 3

    # 返回一个列向量形式的速度数组
    return speed[:, None]

def post_process(best_lateral_sampling, ego_state, map_api, route_roadblock_ids, traffic_light_data):


    x = best_lateral_sampling[:, 0]
    y = best_lateral_sampling[:, 1]
    ryaw, rk = calculate_yaw_and_curvature(x, y)
    #print(ryaw.shape)
    spline_path = np.stack([x, y, ryaw, rk], axis=1)
    #plt.plot(spline_path[:, 0], spline_path[:, 1], color="y", linewidth=3)
    #plt.show()
    #插值以后可能不够1200个点
    if spline_path.shape[0] < MAX_LEN * 10:#如果路径长度小于MAX_LEN * 10（即1200），则通过重复最后一个点填充路径以达到预期长度。
        spline_path = np.append(spline_path, np.repeat(spline_path[np.newaxis, -1], MAX_LEN*10-len(spline_path), axis=0), axis=0)
        #print(spline_path.shape)#(1200, 4)   
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
            #print(red_light_lane.shape)#(129, 2)(132, 2)
            occupancy = annotate_occupancy(occupancy, spline_path, red_light_lane)

    target_speed = 13.0 # [m/s]  这里是否需要改成15？
    target_speed = starting_block.interior_edges[0].speed_limit_mps or target_speed
    target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
    #print(target_speed)
    max_speed = annotate_speed1(spline_path, target_speed)
    best_lateral_sampling = np.concatenate([spline_path, max_speed, occupancy], axis=-1) # [x, y, theta, k, v_max, occupancy]

    #print(best_lateral_sampling.shape)#(1200, 6)


    return best_lateral_sampling  

def calc_spline_course_refpath(x, y, ds=0.1):
    if len(x) < 2 or len(y) < 2:
        print("Not enough data points to calculate spline course. Returning None.")
        logging.info(f"Not enough data points to calculate spline course. Returning None.")
        return None, None, None, None
    #plt.plot(x, y, color="y", linewidth=3)
    #print(x)
    xy = np.column_stack((x, y))
    #print(xy)
    # 使用np.unique，但不排序，保持顺序不变
    _, unique_indices = np.unique(xy, axis=0, return_index=True)

    # 保持原始顺序的情况下获取唯一的行
    xy_unique = xy[np.sort(unique_indices)]
    #print(xy_unique)
    # 从 xy_unique 中分离出去重后的 x 和 y
    x_unique = xy_unique[:, 0]
    y_unique = xy_unique[:, 1]
    ''' '''
    sp = Spline2D(x_unique, y_unique)
    #sp = CubicSpline2D(x, y)
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
        # 调用之前定义的函数，将每一条路径从ego坐标系转换回世界坐标系
        world_path = transform_to_world_frame(path, ego_state)
        world_paths.append(world_path)
    
    # 将所有的路径合并成一个数组返回
    return np.array(world_paths)


def transform_to_world_frame(path, ego_state):
    # 获取ego车的世界坐标系下的位置和航向角
    ego_x, ego_y, ego_h = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading

    # 从ego坐标系下的路径点提取x和y坐标
    ego_path_x, ego_path_y = path[:, 0], path[:, 1]

    # 将路径点从ego坐标系转换回世界坐标系
    world_path_x = np.cos(ego_h) * ego_path_x - np.sin(ego_h) * ego_path_y + ego_x
    world_path_y = np.sin(ego_h) * ego_path_x + np.cos(ego_h) * ego_path_y + ego_y

    # 将转换后的x和y坐标组合成路径
    world_path = np.column_stack((world_path_x, world_path_y))

    return world_path


def transform_all_paths_to_ego_frame(paths, ego_state):
    ego_paths = []
    for path in paths:
        # 调用之前定义的函数，将每一条路径从ego坐标系转换回世界坐标系
        ego_path = transform_to_ego_frame(path, ego_state)
        ego_paths.append(ego_path)
    
    # 将所有的路径合并成一个数组返回
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
        #print(obstacle_polygon)
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
        # 计算每条路径的成本
        cost = calculate_cost(path, traffic_light_data, ego_state, map_api, route_roadblock_ids, observation)
        
        # 存储路径及其成本
        path_costs.append((cost, path))
    
    # 根据成本排序，按从小到大的顺序
    path_costs.sort(key=lambda x: x[0])
    #print(path_costs)
    
    # 选出成本最小的k条路径
    top_k_paths = [path for cost, path in path_costs[:k]]
    
    # 将路径列表转换为 numpy 数组，确保维度为 (k, 1200, 2)
    top_k_paths_array = np.array([path for path in top_k_paths])
    
    # 返回成本最小的路径及其对应的成本
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
        obstacle_polygon = obstacle.geometry  # 假设每个障碍物的 geometry 是一个多边形
        coords = np.array(obstacle_polygon.exterior.coords)  # 提取多边形的坐标
        all_coordinates.append(coords)
    
    # 检查是否有坐标
    if len(all_coordinates) == 0:
        return None
    else:
        # 将所有坐标堆叠成一个 (n, 5, 2) 的 numpy 数组
        all_coordinates = np.stack(all_coordinates, axis=0)
        all_coordinates_ego_frame = transform_all_paths_to_ego_frame(all_coordinates, ego_state)
        #print(all_coordinates_ego_frame)
        all_frenet_obstacles = transform_obstacles_to_frenet(all_coordinates_ego_frame, best_ref_path)#将静态障碍物转化为frenet，并过滤掉s<=0，s>30, l>10的点
        if all_frenet_obstacles is not None:

            return all_frenet_obstacles
        else:

            return None

        
    
def plot_obstacle_coordinates(all_coordinates):
    if all_coordinates is None:
        print("No obstacle coordinates to plot.")
        return

    plt.figure(figsize=(10, 8))
    
    # 遍历每个障碍物的坐标并绘制
    for coords in all_coordinates:
        plt.plot(coords[:, 0], coords[:, 1], '-o', label='Obstacle')
    
     # 获取s的最小值和最大值，设置横轴范围
    s_min = np.min(all_coordinates[:, :, 0])
    s_max = np.max(all_coordinates[:, :, 0])

    plt.xlabel('Frenet s')
    plt.ylabel('Frenet l')
    plt.title('Frenet Coordinates of Obstacles')

    # 设置横轴范围，显示负坐标
    plt.xlim([s_min, s_max])
    
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()



def occupancy_from_obstacles(all_frenet_obstacles, max_len=120, plot=False):
    """
    生成基于障碍物的占用图，精度为 0.1m。
    
    参数:
    - all_frenet_obstacles: 障碍物在 Frenet 坐标系下的坐标，形状为 (num_obstacles, 5, 2)
    - max_len: 路径的最大长度（默认为 120），用于构建占用图
    - plot: 是否可视化占用图
    
    返回:
    - occupancy_map: 基于障碍物的占用图，形状为 (1200, 200)，表示 s 和 l 的栅格图
    """
    # 创建一个初始为 0 的占用图 (1200 表示 s 的范围 [0, 120)，200 表示 l 的范围 [-10, 10))
    occupancy_map = np.zeros((max_len * 10, 200))  # 每个栅格代表 0.1m

    # 遍历每个障碍物
    for obstacle_coords in all_frenet_obstacles:
        # 构建障碍物的多边形
        obstacle_polygon = Polygon(obstacle_coords)

        # 获取该多边形在 s 和 l 方向的最小和最大范围
        min_s = np.clip(np.min(obstacle_coords[:, 0]), 0, max_len)
        max_s = np.clip(np.max(obstacle_coords[:, 0]), 0, max_len)
        min_l = np.clip(np.min(obstacle_coords[:, 1]), -10, 10)
        max_l = np.clip(np.max(obstacle_coords[:, 1]), -10, 10)

        # 由于 l 的精度更高，我们将 l 坐标的区间进一步细分为 40 个点
        for i in range(int(min_s * 10), int(max_s * 10)):  # s 方向栅格化到 [0, 1200]
            s = i / 10.0  # 将索引 i 转换为实际的 s 值
            for l_idx in range(200):  # l 方向栅格化到 [-10, 10)，每个栅格代表 0.1m
                l = (l_idx - 100) * 0.1  # 将索引 l_idx 映射到实际的 l 值
                
                # 现在，我们不使用 Polygon.contains，而是检查 s 和 l 是否在障碍物的占用范围内
                if min_l <= l <= max_l:
                    #print(l)
                    occupancy_map[i, l_idx] = 1  # 标记为占用

    # 可视化 occupancy_map
    if plot:
        plt.figure(figsize=(12, 6))
        
        # 绘制 occupancy map 的热图
        plt.imshow(occupancy_map.T, cmap='hot', interpolation='nearest', aspect='auto', extent=[0, max_len, 10, -10])
        # 交换 extent 中的纵坐标值，将正负 l 显示正确
        
        plt.colorbar(label='Occupancy')
        plt.xlabel('Frenet s coordinate (distance along path)')
        plt.ylabel('Frenet l coordinate (lateral offset)')
        plt.title('Occupancy Map in Frenet Frame (s: 0-120m, l: -10m to 10m)')
        plt.grid(True)
        plt.show()
    
    return occupancy_map

def calculate_cost(path,  traffic_light_data, ego_state, map_api, route_roadblock_ids, observation):
    '''
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
    '''

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
    #print(objects)
    #all_coordinates = extract_polygon_coordinates(obstacles)
    #print(all_coordinates.shape)
    '''
    num_edges = len(starting_block.interior_edges)  #换道损失，还没想好怎么写
    if len(traffic_light_data) > 0:
        just_stay_current = True
    elif num_edges >= 4 and ego_state.dynamic_car_state.rear_axle_velocity_2d.x <= 3:
        just_stay_current = True
    else:
        just_stay_current = False
            # lane change
    lane_change = path[1]
    if just_stay_current:
        lane_change = 5 * lane_change
    '''
    
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
    """
    计算参考轨迹和优化轨迹在 Frenet 坐标系下每个点的 s 和 l 的差值。

    参数：
    ref_path_frenet (numpy.ndarray): 参考轨迹，形状为 (1200, 3)，其中包含 s、l 和其他信息。
    sl (numpy.ndarray): 优化轨迹，形状为 (1200, 2)，其中包含 s 和 l 值。

    返回：
    s_diff (numpy.ndarray): s 差值，形状为 (1200,)
    l_diff (numpy.ndarray): l 差值，形状为 (1200,)
    """
    # 提取参考轨迹中的 s 和 l 值
    ref_s = ref_path_frenet[:, 0]
    ref_l = ref_path_frenet[:, 1]

    # 提取优化轨迹中的 s 和 l 值
    opt_s = sl[:, 0]
    opt_l = sl[:, 1]

    # 计算 s 和 l 的差值
    s_diff = ref_s - opt_s
    l_diff = ref_l - opt_l
    
    # 计算 s 和 l 的差值的和
    s_diff = np.sum(s_diff)
    l_diff = np.sum(l_diff)



    return s_diff, l_diff




def extend_trajectory(x, y, extend_len=10):
    """
    在轨迹末端进行插值，延长指定数量的点
    """
    last_dx = x[-1] - x[-2]
    last_dy = y[-1] - y[-2]

    for i in range(extend_len):
        x = np.append(x, x[-1] + last_dx)
        y = np.append(y, y[-1] + last_dy)

    return x, y

def calculate_yaw_and_curvature(x, y):
    """
    计算路径的偏航角（yaw）和曲率（curvature），保持长度为1200。
    如果导数减少了点数，通过轨迹末端插值来补齐点数。
    """
    # 计算一阶导数 dx/ds 和 dy/ds
    dx = np.diff(x)
    dy = np.diff(y)

    # 计算二阶导数 d²x/ds² 和 d²y/ds²
    ddx = np.diff(dx)
    ddy = np.diff(dy)

    # 插值扩展点数，保证计算出的点为1200
    if len(dx) < 1200:
        x_extended, y_extended = extend_trajectory(x, y, 1200 - len(dx) - 1)
        dx = np.diff(x_extended)
        dy = np.diff(y_extended)
        ddx = np.diff(dx)
        ddy = np.diff(dy)

    # 计算偏航角（yaw）
    yaw = np.arctan2(dy, dx)

    # 计算曲率（curvature）
    curvature = (ddy * dx[:-1] - ddx * dy[:-1]) / ((dx[:-1] ** 2 + dy[:-1] ** 2) ** (3 / 2))

    # 插值结果，保持长度为1200
    yaw = np.pad(yaw, (0, 1200 - len(yaw)), 'edge')
    curvature = np.pad(curvature, (0, 1200 - len(curvature)), 'edge')

    return yaw, curvature