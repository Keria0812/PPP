import torch,math
import scipy
from shapely import Point, LineString
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.training.preprocessing.features.agents import Agents
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.geometry.torch_geometry import global_state_se2_tensor_to_local
from nuplan.planning.training.preprocessing.utils.vector_preprocessing import interpolate_points
from nuplan.common.geometry.torch_geometry import vector_set_coordinates_to_local_frame
from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import *
from PPP.ref_path import Ref_path
from PPP.data_utils import get_surrounding_vector_set_map,process_map_data
from common_utils import *


def observation_adapter_addref(history_buffer, ego_state, observation, traffic_light_data, map_api, route_roadblock_ids, device='cpu'):
    num_agents = 20
    map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
    max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
    max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
    radius = 100 # [m] query radius scope relative to the current pose.
    interpolation_method = 'linear'

    ego_state_buffer = history_buffer.ego_state_buffer # Past ego state including the current
    observation_buffer = history_buffer.observation_buffer # Past observations including the current

    ego_agent_past = sampled_past_ego_states_to_tensor(ego_state_buffer)
    past_tracked_objects_tensor_list, past_tracked_objects_types = sampled_tracked_objects_to_tensor_list(observation_buffer)
    time_stamps_past = sampled_past_timestamps_to_tensor([state.time_point for state in ego_state_buffer])
    ego_state = history_buffer.current_state[0]
    ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)


    ref_path = get_reference_path(ego_state, traffic_light_data, observation, map_api, route_roadblock_ids)
    #print(ref_path.shape)#(2, 1200, 6)
    if ref_path is None:
        ref_path = np.zeros((1, 1200, 6), dtype=np.float32)

    coords, traffic_light_data = get_surrounding_vector_set_map(
        map_api, map_features, ego_coords, radius, route_roadblock_ids, traffic_light_data
    )

    ego_agent_past, neighbor_agents_past = agent_past_process(
        ego_agent_past, time_stamps_past, past_tracked_objects_tensor_list, past_tracked_objects_types, num_agents
    )



    vector_map = process_map_data(ego_state.rear_axle, coords, traffic_light_data, map_features, 
                             max_elements, max_points, interpolation_method)

    data = {"ego_agent_past": ego_agent_past[1:], 
            "neighbor_agents_past": neighbor_agents_past[:, 1:],
            "ref_paths":ref_path}
    data.update(vector_map)
    data = convert_to_model_inputs(data, device)

    return data


def convert_to_model_inputs(data, device):
    tensor_data = {}
    for k, v in data.items():
        # Convert numpy arrays to torch tensors
        if isinstance(v, np.ndarray):
            v = torch.tensor(v)
        tensor_data[k] = v.float().unsqueeze(0).to(device)

    return tensor_data

def initialize_route_plan( map_api, route_roadblock_ids):

    route_roadblocks = []

    for id_ in route_roadblock_ids:
        block = map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
        block = block or map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        route_roadblocks.append(block)

    candidate_lane_edge_ids = [
        edge.id for block in route_roadblocks if block for edge in block.interior_edges
    ]
    return candidate_lane_edge_ids, route_roadblocks
    


def get_reference_path(ego_state, traffic_light_data, observation, map_api, route_roadblock_ids):
    # Get starting block
    starting_block = None
    min_target_speed = 3
    max_target_speed = 15
    cur_point = (ego_state.rear_axle.x, ego_state.rear_axle.y)
    closest_distance = math.inf

    candidate_lane_edge_ids,route_roadblocks = initialize_route_plan(map_api, route_roadblock_ids)
    for block in route_roadblocks:
        for edge in block.interior_edges:
            distance = edge.polygon.distance(Point(cur_point))
            if distance < closest_distance:
                starting_block = block
                closest_distance = distance

        if np.isclose(closest_distance, 0):
            break
        
    # In case the ego vehicle is not on the route, return None
    if closest_distance > 5:
        return None

    # Get reference path, handle exception
    try:
        path_planner = Ref_path(candidate_lane_edge_ids)
        ref_paths = path_planner.plan_route(ego_state, starting_block, observation, traffic_light_data)
    except:
        ref_paths = None

    if ref_paths is None:
        return None

    # Initialize list to store processed paths
    processed_paths = []

    for ref_path in ref_paths:

        # Annotate red light to occupancy
        occupancy = np.zeros(shape=(ref_path.shape[0], 1))
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in candidate_lane_edge_ids:
                lane_conn = map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                conn_path = lane_conn.baseline_path.discrete_path
                conn_path = np.array([[p.x, p.y] for p in conn_path])
                red_light_lane = transform_to_ego_frame(conn_path, ego_state)
                #print(red_light_lane.shape)#(129, 2)(132, 2)
                occupancy = annotate_occupancy(occupancy, ref_path, red_light_lane)

        # Annotate max speed along the reference path
        target_speed = 13.0 # [m/s]  这里是否需要改成15？
        target_speed = starting_block.interior_edges[0].speed_limit_mps or target_speed
        target_speed = np.clip(target_speed, min_target_speed, max_target_speed)
        max_speed = annotate_speed(ref_path, target_speed)

        # Finalize reference path
        ref_path = np.concatenate([ref_path, max_speed, occupancy], axis=-1) # [x, y, theta, k, v_max, occupancy]


        processed_paths.append(ref_path.astype(np.float32))

    # Stack all processed paths into a single array
    if processed_paths == None:
        array_zeros = np.zeros((1, 1200, 6), dtype=np.float32)
        return array_zeros

    stacked_paths = np.stack(processed_paths, axis=0) 
    
    return stacked_paths.astype(np.float32)




def extract_agent_tensor(tracked_objects, track_token_ids, object_types):
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    output = torch.zeros((len(agents), AgentInternalIndex.dim()), dtype=torch.float32)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]

        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y
        agent_types.append(agent.tracked_object_type)

    return output, track_token_ids, agent_types


def sampled_tracked_objects_to_tensor_list(past_tracked_objects):
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    output = []
    output_types = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        tensorized, track_token_ids, agent_types = extract_agent_tensor(past_tracked_objects[i].tracked_objects, track_token_ids, object_types)
        output.append(tensorized)
        output_types.append(agent_types)

    return output, output_types


def convert_feature_layer_to_fixed_size(ego_pose, feature_coords, feature_tl_data, max_elements, max_points,
                                         traffic_light_encoding_dim, interpolation):
    if feature_tl_data is not None and len(feature_coords) != len(feature_tl_data):
        raise ValueError(f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}")

    # trim or zero-pad elements to maintain fixed size
    coords_tensor = torch.zeros((max_elements, max_points, 2), dtype=torch.float32)
    avails_tensor = torch.zeros((max_elements, max_points), dtype=torch.bool)
    tl_data_tensor = (
        torch.zeros((max_elements, max_points, traffic_light_encoding_dim), dtype=torch.float32)
        if feature_tl_data is not None else None
    )

    # get elements according to the mean distance to the ego pose
    mapping = {}
    for i, e in enumerate(feature_coords):
        dist = torch.norm(e - ego_pose[None, :2], dim=-1).min()
        mapping[i] = dist

    mapping = sorted(mapping.items(), key=lambda item: item[1])
    sorted_elements = mapping[:max_elements]

    # pad or trim waypoints in a map element
    for idx, element_idx in enumerate(sorted_elements):
        element_coords = feature_coords[element_idx[0]]
    
        # interpolate to maintain fixed size if the number of points is not enough
        element_coords = interpolate_points(element_coords, max_points, interpolation=interpolation)
        coords_tensor[idx] = element_coords
        avails_tensor[idx] = True  # specify real vs zero-padded data

        if tl_data_tensor is not None and feature_tl_data is not None:
            tl_data_tensor[idx] = feature_tl_data[element_idx[0]]

    return coords_tensor, tl_data_tensor, avails_tensor


def global_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * torch.cos(anchor_heading) + velocity[:, 1] * torch.sin(anchor_heading)
    velocity_y = velocity[:, 1] * torch.cos(anchor_heading) - velocity[:, 0] * torch.sin(anchor_heading)

    return torch.stack([velocity_x, velocity_y], dim=-1)


def convert_absolute_quantities_to_relative(agent_state, ego_state, agent_type='ego'):
    """
    Converts the agent' poses and relative velocities from absolute to ego-relative coordinates.
    :param agent_state: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
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
        transformed_velocities = global_velocity_to_local(agent_global_velocities, ego_pose[-1])
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()

    return agent_state


def agent_past_process(past_ego_states, past_time_stamps, past_tracked_objects, tracked_objects_types, num_agents):
    agents_states_dim = Agents.agents_states_dim()
    ego_history = past_ego_states
    time_stamps = past_time_stamps
    agents = past_tracked_objects

    anchor_ego_state = ego_history[-1, :].squeeze().clone()
    ego_tensor = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)
    agent_history = filter_agents_tensor(agents, reverse=True)
    agent_types = tracked_objects_types[-1]

    if agent_history[-1].shape[0] == 0:
        # Return zero tensor when there are no agents in the scene
        agents_tensor = torch.zeros((len(agent_history), 0, agents_states_dim)).float()
    else:
        local_coords_agent_states = []
        padded_agent_states = pad_agent_states(agent_history, reverse=True)

        for agent_state in padded_agent_states:
            local_coords_agent_states.append(convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))
    
        # Calculate yaw rate
        yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
    
        agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)

    agents = torch.zeros((num_agents, agents_tensor.shape[0], agents_tensor.shape[-1]+3), dtype=torch.float32)

    # sort agents according to distance to ego
    distance_to_ego = torch.norm(agents_tensor[-1, :, :2], dim=-1)
    indices = list(torch.argsort(distance_to_ego).numpy())[:num_agents]

    # fill agent features into the array
    added_agents = 0
    for i in indices:
        if added_agents >= num_agents:
            break
        
        if agents_tensor[-1, i, 0] < -6.0:
            continue

        agents[added_agents, :, :agents_tensor.shape[-1]] = agents_tensor[:, i, :agents_tensor.shape[-1]]

        if agent_types[i] == TrackedObjectType.VEHICLE:
            agents[added_agents, :, agents_tensor.shape[-1]:] = torch.tensor([1, 0, 0])
        elif agent_types[i] == TrackedObjectType.PEDESTRIAN:
            agents[added_agents, :, agents_tensor.shape[-1]:] = torch.tensor([0, 1, 0])
        else:
            agents[added_agents, :, agents_tensor.shape[-1]:] = torch.tensor([0, 0, 1])

        added_agents += 1

    return ego_tensor, agents



def annotate_occupancy(occupancy, ego_path, red_light_lane):
    #print(ego_path.shape)
    #print(red_light_lane.shape)
    ego_path_red_light = scipy.spatial.distance.cdist(ego_path[:, :2], red_light_lane)

    if len(red_light_lane) < 80:
        pass
    else:
        occupancy[np.any(ego_path_red_light < 0.5, axis=-1)] = 1

    return occupancy


def annotate_speed(ref_path, speed_limit):
    # 初始化一个速度数组，初始值为整个路径上的默认速度限制
    speed = np.ones(len(ref_path)) * speed_limit
    
    # 找到转弯点的索引，定义为路径中曲率超过1/10的第一个点
    turning_idx = np.argmax(np.abs(ref_path[:, 3]) > 1/10)

    # 如果找到了转弯点，将转弯点及其之后的速度限制设置为3 m/s
    if turning_idx > 0:
        speed[turning_idx:] = 3

    # 返回一个列向量形式的速度数组
    return speed[:, None]





def transform_to_ego_frame(path, ego_state):
    ego_x, ego_y, ego_h = ego_state.rear_axle.x, ego_state.rear_axle.y, ego_state.rear_axle.heading
    path_x, path_y = path[:, 0], path[:, 1]
    ego_path_x = np.cos(ego_h) * (path_x - ego_x) + np.sin(ego_h) * (path_y - ego_y)
    ego_path_y = -np.sin(ego_h) * (path_x - ego_x) + np.cos(ego_h) * (path_y - ego_y)
    ego_path = np.stack([ego_path_x, ego_path_y], axis=-1)

    return ego_path