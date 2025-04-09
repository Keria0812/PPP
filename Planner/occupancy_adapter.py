import scipy
import numpy as np
import matplotlib.pyplot as plt
from common_utils import *
from nuplan.planning.metrics.utils.expert_comparisons import principal_value


def occupancy_grid(predictions, scores, neighbor_data, reference_path, plot=False):

    best_mode = np.argmax(scores.cpu().numpy(), axis=-1)
    predictions = predictions.cpu().numpy()
    neighbor_data = neighbor_data.cpu().numpy()
    
    best_predictions = [predictions[i, best_mode[i], :, :2] for i in range(predictions.shape[0])]
    transformed_predictions = [transform_to_frenet_coordinates(a, reference_path) for a in best_predictions]    
    path_length = reference_path.shape[0]
    
    # Initialize time occupancy based on reference path
    time_occupancy = np.stack(T * 10 * [reference_path[:, -1]], axis=0) 

    for t in range(10 * T):  # Assuming T is defined elsewhere
        for n, traj in enumerate(transformed_predictions):
            if neighbor_data[n][0] == 0:
                continue

            if traj[0][0] <= 0:
                continue
            
            # Calculate the threshold for occupancy
            agent_width = neighbor_data[n][7]
            threshold = agent_width * 0.5 + WIDTH * 0.5 + 0.3  # Assuming WIDTH is predefined

            if traj[t][0] > 0 and np.abs(traj[t][1]) < threshold:
                agent_length = neighbor_data[n][6]
                backward = 0.5 * agent_length + 3
                forward = 0.5 * agent_length
                start_pos = np.clip(traj[t][0] - backward, 0, MAX_LEN)  # Assuming MAX_LEN is predefined
                end_pos = np.clip(traj[t][0] + forward, 0, MAX_LEN)
                time_occupancy[t][int(start_pos*10):int(end_pos*10)] = 1

        if path_length < MAX_LEN * 10:
            time_occupancy[t][path_length:] = 1

    # Reshape the time occupancy and calculate the maximum
    time_occupancy = np.reshape(time_occupancy, (10 * T, -1, 10))  # Assuming T is defined
    time_occupancy = np.max(time_occupancy, axis=-1)
    
    if plot:
        visualize_time_occupancy(time_occupancy)

    return time_occupancy


def visualize_time_occupancy(time_occupancy):
    """
    Visualizes the time occupancy map as a heatmap.
    
    :param time_occupancy: Time occupancy map with shape (T * 10, max_len).
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(time_occupancy, cmap='hot', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Occupancy')
    plt.xlabel('Frenet s coordinate (distance along path)')
    plt.ylabel('Time step')
    plt.title('Time-dependent Occupancy Map')
    plt.show()


def transform_to_frenet_coordinates(trajectory, reference_path):
    """
    Transforms trajectory coordinates into Frenet coordinates based on a reference path.
    
    :param trajectory: The trajectory to transform, shape (N, 2 or 3).
    :param reference_path: The reference path to which the trajectory is transformed.
    :return: Frenet coordinates as an array.
    """
    distance_to_ref_path = scipy.spatial.distance.cdist(trajectory[:, :2], reference_path[:, :2])
    frenet_idx = np.argmin(distance_to_ref_path, axis=-1)
    reference_points = reference_path[frenet_idx]
    interval = 0.1

    frenet_s = interval * frenet_idx
    sign_factor = np.sign((trajectory[:, 1] - reference_points[:, 1]) * np.cos(reference_points[:, 2]) - 
                          (trajectory[:, 0] - reference_points[:, 0]) * np.sin(reference_points[:, 2]))
    frenet_l = np.linalg.norm(trajectory[:, :2] - reference_points[:, :2], axis=-1) * sign_factor 

    if trajectory.shape[-1] == 3:
        frenet_heading = principal_value(reference_points[:, 2] - trajectory[:, 2])
        frenet_trajectory = np.column_stack([frenet_s, frenet_l, frenet_heading])
    else:
        frenet_trajectory = np.column_stack([frenet_s, frenet_l])

    return frenet_trajectory
