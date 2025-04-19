import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from nuplan.database.nuplan_db.nuplan_scenario_queries import *
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
from nuplan.planning.training.preprocessing.features.trajectory_utils import convert_absolute_to_relative_poses

from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import (
    AgentInternalIndex,
    EgoInternalIndex,
    sampled_past_ego_states_to_tensor,
    sampled_past_timestamps_to_tensor,
    compute_yaw_rate_from_state_tensors,
    filter_agents_tensor,
    pack_agents_tensor,
    pad_agent_states
)

from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import interpolate_points


def extract_agent_tensor_from_tracked_objects(tracked_objects, track_token_ids, object_types):
    """
    Extracts the relevant data from the agents present in a past detection into a tensor.
    Only objects of specified type will be transformed. Others will be ignored.
    :param tracked_objects: The tracked objects to turn into a tensor.
    :param track_token_ids: A dictionary used to assign track tokens to integer IDs.
    :param object_types: TrackedObjectType to filter agents by.
    :return: The generated tensor and the updated track_token_ids dict.
    """
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    tensor_output = torch.zeros((len(agents), AgentInternalIndex.dim()), dtype=torch.float32)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]

        tensor_output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        tensor_output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        tensor_output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        tensor_output[idx, AgentInternalIndex.heading()] = agent.center.heading
        tensor_output[idx, AgentInternalIndex.width()] = agent.box.width
        tensor_output[idx, AgentInternalIndex.length()] = agent.box.length
        tensor_output[idx, AgentInternalIndex.x()] = agent.center.x
        tensor_output[idx, AgentInternalIndex.y()] = agent.center.y
        agent_types.append(agent.tracked_object_type)

    return tensor_output, track_token_ids, agent_types


def tensorize_tracked_objects(past_tracked_objects):
    """
    Converts past tracked objects to tensorized features.
    :param past_tracked_objects: The tracked objects to tensorize.
    :return: The tensorized objects and their types.
    """
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    tensor_output = []
    object_types_output = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        tensorized_data, track_token_ids, agent_types = extract_agent_tensor_from_tracked_objects(past_tracked_objects[i], track_token_ids, object_types)
        tensor_output.append(tensorized_data)
        object_types_output.append(agent_types)

    return tensor_output, object_types_output


def convert_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * torch.cos(anchor_heading) + velocity[:, 1] * torch.sin(anchor_heading)
    velocity_y = velocity[:, 1] * torch.cos(anchor_heading) - velocity[:, 0] * torch.sin(anchor_heading)

    return torch.stack([velocity_x, velocity_y], dim=-1)


def convert_coordinates_to_relative(agent_state, ego_state, agent_type='ego'):
    """
    Converts agent coordinates and velocities from absolute to ego-relative.
    :param agent_state: The agent state to convert.
    :param ego_state: The ego state to convert.
    :param agent_type: Type of agent ('ego' for ego vehicle).
    :return: The converted agent state in relative coordinates.
    """
    ego_pose = torch.tensor(
        [
            float(ego_state[EgoInternalIndex.x()].item()),
            float(ego_state[EgoInternalIndex.y()].item()),
            float(ego_state[EgoInternalIndex.heading()].item()),
        ],
        dtype=torch.float64,
    )

    if agent_type == 'ego':
        agent_global_poses = agent_state[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        agent_state[:, EgoInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, EgoInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, EgoInternalIndex.heading()] = transformed_poses[:, 2].float()
    else:
        agent_global_poses = agent_state[:, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]]
        agent_global_velocities = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]]
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        transformed_velocities = convert_velocity_to_local(agent_global_velocities, ego_pose[-1])
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()

    return agent_state


def process_past_agent_data(past_ego_states, past_time_stamps, past_tracked_objects, tracked_object_types, num_agents):
    """
    Processes past data for the ego and agents.
    :param past_ego_states: The ego's past states.
    :param past_time_stamps: Time stamps for the past states.
    :param past_tracked_objects: Past tracked objects.
    :param tracked_object_types: Types of tracked objects.
    :param num_agents: Number of agents.
    :return: Processed ego and agent data.
    """
    agents_states_dim = Agents.agents_states_dim()
    ego_history = past_ego_states
    time_stamps = past_time_stamps
    agents = past_tracked_objects

    anchor_ego_state = ego_history[-1, :].squeeze().clone()
    ego_tensor = convert_coordinates_to_relative(ego_history, anchor_ego_state)
    agent_history = filter_agents_tensor(agents, reverse=True)
    agent_types = tracked_object_types[-1]

    if agent_history[-1].shape[0] == 0:
        agents_tensor = torch.zeros((len(agent_history), 0, agents_states_dim)).float()
    else:
        local_coords_agent_states = []
        padded_agent_states = pad_agent_states(agent_history, reverse=True)

        for agent_state in padded_agent_states:
            local_coords_agent_states.append(convert_coordinates_to_relative(agent_state, anchor_ego_state, 'agent'))

        yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
        agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)

    agents_tensor_padded = np.zeros(shape=(num_agents, agents_tensor.shape[0], agents_tensor.shape[-1]+3), dtype=np.float32)

    distance_to_ego = torch.norm(agents_tensor[-1, :, :2], dim=-1)
    indices = list(torch.argsort(distance_to_ego).numpy())[:num_agents]

    for i, j in enumerate(indices):
        agents_tensor_padded[i, :, :agents_tensor.shape[-1]] = agents_tensor[:, j, :agents_tensor.shape[-1]].numpy()
        if agent_types[j] == TrackedObjectType.VEHICLE:
            agents_tensor_padded[i, :, agents_tensor.shape[-1]:] = [1, 0, 0]
        elif agent_types[j] == TrackedObjectType.PEDESTRIAN:
            agents_tensor_padded[i, :, agents_tensor.shape[-1]:] = [0, 1, 0]
        else:
            agents_tensor_padded[i, :, agents_tensor.shape[-1]:] = [0, 0, 1]

    return ego_tensor.numpy().astype(np.float32), agents_tensor_padded, indices


def process_future_agent_data(anchor_ego_state, future_tracked_objects, num_agents, agent_index):
    """
    Processes the future agent data.
    :param anchor_ego_state: The ego vehicle state.
    :param future_tracked_objects: Future tracked agent objects.
    :param num_agents: Number of agents.
    :param agent_index: Index of the agents.
    :return: Future state of agents.
    """
    anchor_ego_state = torch.tensor([anchor_ego_state.rear_axle.x, anchor_ego_state.rear_axle.y, anchor_ego_state.rear_axle.heading, 
                                     anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                                     anchor_ego_state.dynamic_car_state.rear_axle_velocity_2d.y,
                                     anchor_ego_state.dynamic_car_state.rear_axle_acceleration_2d.x,
                                     anchor_ego_state.dynamic_car_state.rear_axle_acceleration_2d.y])

    agent_future_data = filter_agents_tensor(future_tracked_objects)
    local_coords_agent_states = []
    for agent_state in agent_future_data:
        local_coords_agent_states.append(convert_coordinates_to_relative(agent_state, anchor_ego_state, 'agent'))
    padded_agent_states = pad_agent_states_with_zeros(local_coords_agent_states)

    agent_futures = np.zeros(shape=(num_agents, padded_agent_states.shape[0]-1, 3), dtype=np.float32)
    for i, j in enumerate(agent_index):
        agent_futures[i] = padded_agent_states[1:, j, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]].numpy()

    return agent_futures


def pad_agent_states_with_zeros(agent_trajectories):
    key_frame = agent_trajectories[0]
    track_id_idx = AgentInternalIndex.track_token()
    padded_agent_trajectories = torch.zeros((len(agent_trajectories), key_frame.shape[0], key_frame.shape[1]), dtype=torch.float32)

    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]
        mapped_rows = frame[:, track_id_idx]

        for row_idx in range(key_frame.shape[0]):
            if row_idx in mapped_rows:
                padded_agent_trajectories[idx, row_idx] = frame[frame[:, track_id_idx] == row_idx]

    return padded_agent_trajectories


def process_feature_layer_to_fixed_size(ego_pose, feature_coords, feature_tl_data, max_elements, max_points,
                                         traffic_light_encoding_dim, interpolation):
    """
    Converts variable sized map features to fixed size tensors. Map elements are padded/trimmed to max_elements size.
        Points per feature are interpolated to maintain max_points size.
    :param ego_pose: the current pose of the ego vehicle.
    :param feature_coords: Vector set of coordinates for collection of elements in map layer.
        [num_elements, num_points_in_element (variable size), 2]
    :param feature_tl_data: Optional traffic light status corresponding to map elements at given index in coords.
        [num_elements, traffic_light_encoding_dim (4)]（可选）与 coords 中给定索引的地图元素相对应的交通灯状态，形状为 [num_elements, traffic_light_encoding_dim (4)]。
    :param max_elements: Number of elements to pad/trim to.
    :param max_points: Number of points to interpolate or pad/trim to.
    :param traffic_light_encoding_dim: Dimensionality of traffic light data.
    :param interpolation: Optional interpolation mode for maintaining fixed number of points per element.
        None indicates trimming and zero-padding to take place in lieu of interpolation. Interpolation options: 'linear' and 'area'.
    :return:
        coords_tensor: The converted coords tensor.
        tl_data_tensor: The converted traffic light data tensor (if available).
        avails_tensor: Availabilities tensor identifying real vs zero-padded data in coords_tensor and tl_data_tensor.
    """
    if feature_tl_data is not None and len(feature_coords) != len(feature_tl_data):
        raise ValueError(f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}")

    coords_tensor = torch.zeros((max_elements, max_points, 2), dtype=torch.float32)
    avails_tensor = torch.zeros((max_elements, max_points), dtype=torch.bool)
    tl_data_tensor = (
        torch.zeros((max_elements, max_points, traffic_light_encoding_dim), dtype=torch.float32)
        if feature_tl_data is not None else None
    )

    mapping = {}
    for i, e in enumerate(feature_coords):
        dist = torch.norm(e - ego_pose[None, :2], dim=-1).min()
        mapping[i] = dist

    sorted_mapping = sorted(mapping.items(), key=lambda item: item[1])
    selected_elements = sorted_mapping[:max_elements]

    for idx, element_idx in enumerate(selected_elements):
        element_coords = feature_coords[element_idx[0]]
        element_coords = interpolate_points(element_coords, max_points, interpolation=interpolation)
        coords_tensor[idx] = element_coords
        avails_tensor[idx] = True

        if tl_data_tensor is not None and feature_tl_data is not None:
            tl_data_tensor[idx] = feature_tl_data[element_idx[0]]

    return coords_tensor, tl_data_tensor, avails_tensor


def get_surrounding_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    route_roadblock_ids: List[str],
    traffic_light_status_data: List[TrafficLightStatusData],
) -> Tuple[Dict[str, MapObjectPolylines], Dict[str, LaneSegmentTrafficLightData]]:
    """
    Extracts map features surrounding the ego vehicle.
    :param map_api: Map API to fetch the features from.
    :param map_features: List of features to extract.
    :param point: Ego vehicle's location.
    :param radius: Search radius for map features.
    :param route_roadblock_ids: List of roadblock IDs on the route.
    :param traffic_light_status_data: Traffic light status information.
    :return: Extracted map features and traffic light data.
    """
    feature_coords = {}
    traffic_light_data = {}
    available_layers = []

    for feature_name in map_features:
        try:
            available_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(f"Feature layer for {feature_name} not available.")

    if VectorFeatureLayer.LANE in available_layers:
        lane_data = get_lane_polylines(map_api, point, radius)
        feature_coords[VectorFeatureLayer.LANE.name] = lane_data[0]

        lane_ids = lane_data[3]
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
            lane_ids, traffic_light_status_data
        )

        if VectorFeatureLayer.LEFT_BOUNDARY in available_layers:
            feature_coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(lane_data[1].polylines)
        if VectorFeatureLayer.RIGHT_BOUNDARY in available_layers:
            feature_coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(lane_data[2].polylines)

    if VectorFeatureLayer.ROUTE_LANES in available_layers:
        route_lane_data = get_route_lane_polylines_from_roadblock_ids(map_api, point, radius, route_roadblock_ids)
        feature_coords[VectorFeatureLayer.ROUTE_LANES.name] = route_lane_data

    for layer in available_layers:
        if layer in VectorFeatureLayerMapping.available_polygon_layers():
            polygons = get_map_object_polygons(
                map_api, point, radius, VectorFeatureLayerMapping.semantic_map_layer(layer)
            )
            feature_coords[layer.name] = polygons

    return feature_coords, traffic_light_data

def process_map_data(anchor_state, feature_coords, traffic_light_data, map_features, max_elements, max_points, interpolation_method):
    """
    This function processes the raw vector set map data and converts it into fixed-size tensor representations.
    :param anchor_state: The current state of the ego vehicle.
    :param feature_coords: The coordinates of different map features (lanes, crosswalks, etc.).
    :param traffic_light_data: The traffic light status data corresponding to the map features.
    :param map_features: List of map feature types to process (e.g., lanes, crosswalks).
    :param max_elements: Maximum number of map elements per feature layer.
    :param max_points: Maximum number of points per feature element.
    :param interpolation_method: The interpolation method to use for adjusting the number of points.
    :return: A dictionary containing processed map features in tensor form.
    """
    anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float32)
    tensor_data = {}

    for feature_name, coords in feature_coords.items():
        feature_coords_list = []

        # Convert coordinates to tensor list
        for coord in coords.to_vector():
            feature_coords_list.append(torch.tensor(coord, dtype=torch.float32))
        tensor_data[f"coords.{feature_name}"] = feature_coords_list

        # Add traffic light data to tensor list if it exists
        if feature_name in traffic_light_data:
            traffic_light_list = []
            for tl_data in traffic_light_data[feature_name].to_vector():
                traffic_light_list.append(torch.tensor(tl_data, dtype=torch.float32))
            tensor_data[f"traffic_light_data.{feature_name}"] = traffic_light_list

    tensor_output = {}
    traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    for feature_name in map_features:
        if f"coords.{feature_name}" in tensor_data:
            feature_coords_tensor = tensor_data[f"coords.{feature_name}"]
            traffic_light_tensor = (
                tensor_data[f"traffic_light_data.{feature_name}"]
                if f"traffic_light_data.{feature_name}" in tensor_data
                else None
            )

            # Convert to fixed size and apply interpolation if needed
            coords, tl_data, availabilities = process_feature_layer_to_fixed_size(
                anchor_state_tensor,
                feature_coords_tensor,
                traffic_light_tensor,
                max_elements[feature_name],
                max_points[feature_name],
                traffic_light_encoding_dim,
                interpolation=interpolation_method if feature_name in [
                    VectorFeatureLayer.LANE.name,
                    VectorFeatureLayer.LEFT_BOUNDARY.name,
                    VectorFeatureLayer.RIGHT_BOUNDARY.name,
                    VectorFeatureLayer.ROUTE_LANES.name,
                    VectorFeatureLayer.CROSSWALK.name
                ] else None
            )

            # Convert to ego-relative frame
            coords = vector_set_coordinates_to_local_frame(coords, availabilities, anchor_state_tensor)

            tensor_output[f"vector_set_map.coords.{feature_name}"] = coords
            tensor_output[f"vector_set_map.availabilities.{feature_name}"] = availabilities

            if tl_data is not None:
                tensor_output[f"vector_set_map.traffic_light_data.{feature_name}"] = tl_data
    
    for feature_name in map_features:
        if feature_name == "LANE":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}'].numpy()
            traffic_light_state = tensor_output[f'vector_set_map.traffic_light_data.{feature_name}'].numpy()
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}'].numpy()
            vector_map_lanes = process_polyline_data(polylines, avails, traffic_light_state)

        elif feature_name == "CROSSWALK":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}'].numpy()
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}'].numpy()
            vector_map_crosswalks = process_polyline_data(polylines, avails)

        elif feature_name == "ROUTE_LANES":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}'].numpy()
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}'].numpy()
            vector_map_route_lanes = process_polyline_data(polylines, avails)

        else:
            pass

    vector_map_output = {'lanes': vector_map_lanes, 'crosswalks': vector_map_crosswalks, 'route_lanes': vector_map_route_lanes}

    return vector_map_output




def process_polyline_data(polylines, availabilities, traffic_light=None):
    """
    This function processes polyline data for map features, including adding heading and optionally traffic light data.
    :param polylines: The polylines data for the feature.
    :param availabilities: Availability of the feature in the map.
    :param traffic_light: Optional traffic light data corresponding to the polylines.
    :return: Processed polylines with added heading and traffic light data.
    """
    dim = 3 if traffic_light is None else 7
    processed_polylines = np.zeros(shape=(polylines.shape[0], polylines.shape[1], dim), dtype=np.float32)

    for i in range(polylines.shape[0]):
        if availabilities[i][0]:
            polyline = polylines[i]
            # Compute heading for each polyline segment
            polyline_heading = wrap_angle_to_pi(np.arctan2(polyline[1:, 1] - polyline[:-1, 1], polyline[1:, 0] - polyline[:-1, 0]))
            polyline_heading = np.insert(polyline_heading, -1, polyline_heading[-1])[:, np.newaxis]
            if traffic_light is None:
                processed_polylines[i] = np.concatenate([polyline, polyline_heading], axis=-1)
            else:
                processed_polylines[i] = np.concatenate([polyline, polyline_heading, traffic_light[i]], axis=-1)

    return processed_polylines


def wrap_angle_to_pi(theta):
    """
    Wraps angle to the range [-pi, pi].
    :param theta: The angle to wrap.
    :return: The angle wrapped to the range [-pi, pi].
    """
    return (theta + np.pi) % (2 * np.pi) - np.pi

def create_ego_raster(vehicle_state):

    vehicle_parameters = get_pacifica_parameters()
    ego_width = vehicle_parameters.width
    ego_front_length = vehicle_parameters.front_length
    ego_rear_length = vehicle_parameters.rear_length

    # Extract ego vehicle state
    x_center, y_center, heading = vehicle_state[0], vehicle_state[1], vehicle_state[2]
    ego_bottom_right = (x_center - ego_rear_length, y_center - ego_width/2)

    # Paint the rectangle
    rect = plt.Rectangle(ego_bottom_right, ego_front_length+ego_rear_length, ego_width, linewidth=2, color='r', alpha=0.6, zorder=3,
                        transform=mpl.transforms.Affine2D().rotate_around(*(x_center, y_center), heading) + plt.gca().transData)
    plt.gca().add_patch(rect)


def create_agents_raster(agents):
    for i in range(agents.shape[0]):
        if agents[i, 0] != 0:
            x_center, y_center, heading = agents[i, 0], agents[i, 1], agents[i, 2]
            agent_length, agent_width = agents[i, 6],  agents[i, 7]
            agent_bottom_right = (x_center - agent_length/2, y_center - agent_width/2)

            rect = plt.Rectangle(agent_bottom_right, agent_length, agent_width, linewidth=2, color='m', alpha=0.6, zorder=3,
                                transform=mpl.transforms.Affine2D().rotate_around(*(x_center, y_center), heading) + plt.gca().transData)
            plt.gca().add_patch(rect)


def create_map_raster(lanes, crosswalks, route_lanes,ref_paths):
    for i in range(lanes.shape[0]):
        lane = lanes[i]
        if lane[0][0] != 0:
            plt.plot(lane[:, 0], lane[:, 1], 'c', linewidth=3) # plot centerline

    for j in range(crosswalks.shape[0]):
        crosswalk = crosswalks[j]
        if crosswalk[0][0] != 0:
            plt.plot(crosswalk[:, 0], crosswalk[:, 1], 'b', linewidth=4) # plot crosswalk

    for k in range(route_lanes.shape[0]):
        route_lane = route_lanes[k]
        if route_lane[0][0] != 0:
            plt.plot(route_lane[:, 0], route_lane[:, 1], 'g', linewidth=4) # plot route_lanes

    for k in range(ref_paths.shape[0]):
        ref_path = ref_paths[k]

        plt.plot(ref_path[:, 0], ref_path[:, 1], 'y', linewidth=4) # plot route_lanes

def draw_trajectory(ego_trajectory, agent_trajectories):
    # plot ego 
    plt.plot(ego_trajectory[:, 0], ego_trajectory[:, 1], 'r', linewidth=3, zorder=3)

    # plot others
    for i in range(agent_trajectories.shape[0]):
        if agent_trajectories[i, -1, 0] != 0:
            trajectory = agent_trajectories[i]
            plt.plot(trajectory[:, 0], trajectory[:, 1], 'm', linewidth=3, zorder=3)
