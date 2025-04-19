import os
import scipy
import argparse
from tqdm import tqdm
from common_utils import *
from PPP.data_utils import *
from PPP.ref_path import Ref_path
import matplotlib.pyplot as plt
import math
import time
from shapely import Point, LineString
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping


class TrajectoryDataProcessor(object):
    def __init__(self, scenarios):
        self.scenarios = scenarios

        self.past_time_window = 2  # [seconds]
        self.num_past_positions = 10 * self.past_time_window
        self.future_time_window = 8  # [seconds]
        self.num_future_positions = 10 * self.future_time_window
        self.max_agents = 20

        self.map_feature_types = ['LANE', 'ROUTE_LANES', 'CROSSWALK']
        self.max_elements_per_layer = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5}
        self.max_points_per_feature = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30}
        self.query_radius = 100  # [m]
        self.interpolation_method = 'linear'  # interpolation method

    def get_ego_vehicle_state(self):
        self.anchor_ego_state = self.scenario.initial_ego_state
        
        past_ego_states = self.scenario.get_ego_past_trajectory(
            iteration=0, num_samples=self.num_past_positions, time_horizon=self.past_time_window
        )

        sampled_past_ego_states = list(past_ego_states) + [self.anchor_ego_state]
        past_ego_states_tensor = sampled_past_ego_states_to_tensor(sampled_past_ego_states)

        past_time_stamps = list(
            self.scenario.get_past_timestamps(
                iteration=0, num_samples=self.num_past_positions, time_horizon=self.past_time_window
            )
        ) + [self.scenario.start_time]

        past_time_stamps_tensor = sampled_past_timestamps_to_tensor(past_time_stamps)

        return past_ego_states_tensor, past_time_stamps_tensor
    
    def get_neighbor_agents_data(self):
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects
        past_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_past_tracked_objects(
                iteration=0, time_horizon=self.past_time_window, num_samples=self.num_past_positions
            )
        ]

        sampled_past_observations = past_tracked_objects + [present_tracked_objects]
        past_tracked_objects_tensor_list, past_tracked_objects_types = \
              tensorize_tracked_objects(sampled_past_observations)

        return past_tracked_objects_tensor_list, past_tracked_objects_types

    def get_map_data(self):        
        ego_state = self.scenario.initial_ego_state
        ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
        route_roadblock_ids = self.scenario.get_route_roadblock_ids()
        traffic_light_data = self.scenario.get_traffic_light_status_at_iteration(0)

        coords, traffic_light_data = get_surrounding_vector_set_map(
            self.map_api, self.map_feature_types, ego_coords, self.query_radius, route_roadblock_ids, traffic_light_data
        )

        vector_map = process_map_data(ego_state.rear_axle, coords, traffic_light_data, self.map_feature_types, 
                                 self.max_elements_per_layer, self.max_points_per_feature, self.interpolation_method)

        return vector_map
    
    def initialize_route_plan(self, route_roadblock_ids):
        self.route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self.route_roadblocks.append(block)

        self.candidate_lane_edge_ids = [
            edge.id for block in self.route_roadblocks if block for edge in block.interior_edges
        ]
    
    def get_reference_path(self):
        traffic_light_data = list(self.scenario.get_traffic_light_status_at_iteration(0))
        ego_state = self.scenario.initial_ego_state
        observation = self.scenario.initial_tracked_objects.tracked_objects
        
        min_target_speed = 3
        max_target_speed = 15
        cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
        closest_distance = math.inf
        starting_block = None

        for block in self.route_roadblocks:
            for edge in block.interior_edges:
                distance = edge.polygon.distance(Point(cur_point))
                if distance < closest_distance:
                    starting_block = block
                    closest_distance = distance

            if np.isclose(closest_distance, 0):
                break
        
        if closest_distance > 5:
            return None

        ref_paths = self.route_path.plan_route(ego_state, starting_block, observation, traffic_light_data)
        
        if ref_paths is None:
            return None

        processed_paths = []
        for ref_path in ref_paths:
            occupancy = np.zeros(shape=(ref_path.shape[0], 1))
            for data in traffic_light_data:
                id_ = str(data.lane_connector_id)
                if data.status == TrafficLightStatusType.RED and id_ in self.candidate_lane_edge_ids:
                    lane_conn = self.map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                    conn_path = lane_conn.baseline_path.discrete_path
                    conn_path = np.array([[p.x, p.y] for p in conn_path])
                    red_light_lane = self.transform_to_ego_frame(conn_path, ego_state)
                    occupancy = self.annotate_occupancy(occupancy, ref_path, red_light_lane)

            target_speed = starting_block.interior_edges[0].speed_limit_mps or self.target_speed
            target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
            max_speed = self.annotate_speed(ref_path, target_speed)

            ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1)
            processed_paths.append(ref_path.astype(np.float32))

        return np.stack(processed_paths, axis=0).astype(np.float32)

    def annotate_occupancy(self, occupancy, ego_path, red_light_lane):
        ego_path_red_light = scipy.spatial.distance.cdist(ego_path[:, :2], red_light_lane)
        if len(red_light_lane) >= 80:
            occupancy[np.any(ego_path_red_light < 0.5, axis=-1)] = 1
        return occupancy

    def annotate_speed(self, ref_path, speed_limit):
        speed = np.ones(len(ref_path)) * speed_limit
        turning_idx = np.argmax(np.abs(ref_path[:, 3]) > 1/10)

        if turning_idx > 0:
            speed[turning_idx:] = 3

        return speed[:, None]

    def transform_to_ego_frame(self, path, ego_state):
        ego_x, ego_y, ego_h = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading
        path_x, path_y = path[:, 0], path[:, 1]
        ego_path_x = np.cos(ego_h) * (path_x - ego_x) + np.sin(ego_h) * (path_y - ego_y)
        ego_path_y = -np.sin(ego_h) * (path_x - ego_x) + np.cos(ego_h) * (path_y - ego_y)
        return np.stack([ego_path_x, ego_path_y], axis=-1)
    

    def get_ego_agent_future(self):
        current_absolute_state = self.scenario.initial_ego_state

        trajectory_absolute_states = self.scenario.get_ego_future_trajectory(
            iteration=0, num_samples=self.num_future_positions, time_horizon=self.num_future_positions
        )

        # Get all future poses of the ego relative to the ego coordinate system
        trajectory_relative_poses = convert_absolute_to_relative_poses(
            current_absolute_state.rear_axle, [state.rear_axle for state in trajectory_absolute_states]
        )

        return trajectory_relative_poses
    
    def get_neighbor_agents_future(self, agent_index):
        current_ego_state = self.scenario.initial_ego_state
        present_tracked_objects = self.scenario.initial_tracked_objects.tracked_objects

        # Get all future poses of of other agents
        future_tracked_objects = [
            tracked_objects.tracked_objects
            for tracked_objects in self.scenario.get_future_tracked_objects(
                iteration=0, time_horizon=self.future_time_window, num_samples=self.num_future_positions
            )
        ]

        sampled_future_observations = [present_tracked_objects] + future_tracked_objects
        future_tracked_objects_tensor_list, _ = tensorize_tracked_objects(sampled_future_observations)
        agent_futures = process_future_agent_data(current_ego_state, future_tracked_objects_tensor_list, self.max_agents, agent_index)

        return agent_futures

    def plot_scenario(self, data):
        create_map_raster(data['lanes'], data['crosswalks'], data['route_lanes'], data['ref_paths'])
        create_ego_raster(data['ego_agent_past'][-1])
        create_agents_raster(data['neighbor_agents_past'][:, -1])
        draw_trajectory(data['ego_agent_past'], data['neighbor_agents_past'])
        draw_trajectory(data['ego_agent_future'], data['neighbor_agents_future'])

        plt.gca().set_aspect('equal')
        plt.tight_layout()
        plt.show()

    def save_to_disk(self, dir, data):
        np.savez(f"{dir}/{data['map_name']}_{data['token']}.npz", **data)

    def process_scenarios(self, save_dir, debug=False):
        for scenario in tqdm(self.scenarios):
            self.scenario = scenario
            self.map_api = scenario.map_api   

            self.route_roadblock_ids = self.scenario.get_route_roadblock_ids()
            self.initialize_route_plan(self.route_roadblock_ids)

            self.route_path = Ref_path(self.candidate_lane_edge_ids)
            self.target_speed = 13.0 # [m/s]

            ego_agent_past, time_stamps_past = self.get_ego_vehicle_state()
            neighbor_agents_past, neighbor_agents_types = self.get_neighbor_agents_data()
            ego_agent_past, neighbor_agents_past, neighbor_indices = \
                process_past_agent_data(ego_agent_past, time_stamps_past, neighbor_agents_past, neighbor_agents_types, self.max_agents)

            vector_map = self.get_map_data()
            ref_paths = self.get_reference_path()
            
            if ref_paths is None:
                continue

            ego_agent_future = self.get_ego_agent_future()
            neighbor_agents_future = self.get_neighbor_agents_future(neighbor_indices)

            data = {
                "map_name": scenario._map_name,
                "token": scenario.token,
                "ego_agent_past": ego_agent_past,
                "ego_agent_future": ego_agent_future,
                "neighbor_agents_past": neighbor_agents_past,
                "neighbor_agents_future": neighbor_agents_future,
                "ref_paths": ref_paths
            }
            data.update(vector_map)

            if debug:
                self.plot_scenario(data)

            self.save_to_disk(save_dir, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Data Processing')
    parser.add_argument('--data_path', type=str, help='path to raw data')
    parser.add_argument('--map_path', type=str, help='path to map data')
    parser.add_argument('--save_path', type=str, help='path to save processed data')
    parser.add_argument('--scenarios_per_type', type=int, default=100, help='number of scenarios per type')
    parser.add_argument('--total_scenarios', default=1000, help='limit total number of scenarios')
    parser.add_argument('--shuffle_scenarios', type=bool, default=False, help='shuffle scenarios')
    parser.add_argument('--debug', action="store_true", help='if visualize the data output', default=False)
    args = parser.parse_args()

    # create save folder
    os.makedirs(args.save_path, exist_ok=True)

    map_version = "nuplan-maps-v1.0"    
    sensor_root = None
    db_files = None
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, sensor_root, db_files, map_version, scenario_mapping=scenario_mapping)
    scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type, args.total_scenarios, args.shuffle_scenarios))
    worker = SingleMachineParallelExecutor(use_process_pool=True)
    scenarios = builder.get_scenarios(scenario_filter, worker)
    print(f"Total number of scenarios: {len(scenarios)}")
    
    # process data
    del worker, builder, scenario_filter, scenario_mapping
    processor = TrajectoryDataProcessor(scenarios)
    processor.process_scenarios(args.save_path, debug=args.debug)
