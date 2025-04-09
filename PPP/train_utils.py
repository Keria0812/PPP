import torch
import logging
import glob
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F
import math
import time
import matplotlib.pyplot as plt
from shapely import Point, LineString
from Planner.planner_utils import *
from PPP.predictor import PPP


from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring

def initLogging(log_file: str, level: str = "INFO"):
    logging.basicConfig(filename=log_file, filemode='w',
                        level=getattr(logging, level, None),
                        format='[%(levelname)s %(asctime)s] %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler())


def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DrivingData(Dataset):
    def __init__(self, data_dir, n_neighbors):
        self.data_list = glob.glob(data_dir)
        self._n_neighbors = n_neighbors

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        inputs= dict()
        inputs['ego_agent_past'] = data['ego_agent_past']
        inputs['neighbor_agents_past'] = data['neighbor_agents_past']
        inputs['route_lanes'] = data['route_lanes'] 
        inputs['map_lanes'] = data['lanes']
        inputs['map_crosswalks'] = data['crosswalks']
        inputs['ref_paths'] = data['ref_paths']
        inputs['ego_agent_future'] = data['ego_agent_future']
        inputs['neighbor_agents_future'] = data['neighbor_agents_future'][:self._n_neighbors]

        return inputs
    

def NLL_Loss(gmm_output, ground_truth):
    """
    Compute the Negative Log Likelihood (NLL) loss based on the Gaussian Mixture Model (GMM) predictions.
    """
    B, N = gmm_output.shape[0], gmm_output.shape[1]
    
    # Extract means and ground truth positions
    predicted_means = gmm_output[..., :2]
    ground_truth_position = ground_truth[:, :, None, :, :2]

    # Calculate pairwise distances
    distances = torch.norm(predicted_means - ground_truth_position, dim=-1)
    
    # Get the index of the best mode (minimum distance)
    best_mode_indices = torch.argmin(distances.mean(dim=-1), dim=-1)

    # Step 2: Extract the means for the best predicted mode
    means = predicted_means
    selected_means = means[torch.arange(B).unsqueeze(1), torch.arange(N).unsqueeze(0), best_mode_indices]
    
    # Calculate the deltas (differences) for x and y coordinates
    delta_x = ground_truth[..., 0] - selected_means[..., 0]
    delta_y = ground_truth[..., 1] - selected_means[..., 1]

    # Step 3: Extract and process the covariance matrices
    covariances = gmm_output[..., 2:]
    selected_covariance = covariances[torch.arange(B).unsqueeze(1), torch.arange(N).unsqueeze(0), best_mode_indices]
    
    # Apply log transformations and exponentiate to get standard deviations
    log_std_x = torch.clamp(selected_covariance[..., 0], min=-2, max=2)
    log_std_y = torch.clamp(selected_covariance[..., 1], min=-2, max=2)
    std_x = torch.exp(log_std_x)
    std_y = torch.exp(log_std_y)

    # Calculate the NLL loss
    nll_loss = log_std_x + log_std_y + 0.5 * (torch.square(delta_x / std_x) + torch.square(delta_y / std_y))
    nll_loss = nll_loss.mean()

    return nll_loss, selected_means


def Cross_Entropy_Loss(scores, best_mode, ground_truth):
    """
    Compute the cross-entropy loss based on the predicted scores and the best mode indices.
    """
    # Cross-entropy loss with label smoothing
    cross_entropy_loss = F.cross_entropy(scores.permute(0, 2, 1), best_mode, label_smoothing=0.2, reduction='none')
    
    # Apply the mask to ignore ground truth with zeros
    cross_entropy_loss = cross_entropy_loss * (ground_truth[:, :, 0, 0]!= 0)
    cross_entropy_loss = cross_entropy_loss.mean()

    return cross_entropy_loss




def l1_loss(plan, ego_future):
    loss = F.smooth_l1_loss(plan, ego_future)
    loss += F.smooth_l1_loss(plan[:, -1], ego_future[:, -1])

    return loss

def l2_loss(plan, ego_future):
    loss = F.mse_loss(plan, ego_future)

    loss += F.mse_loss(plan[:, -1], ego_future[:, -1])

    return loss






def motion_metrics(plan_trajectory, prediction_trajectories, ego_future, neighbors_future, neighbors_future_valid):
    prediction_trajectories = prediction_trajectories * neighbors_future_valid
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)
    heading_error = torch.abs(torch.fmod(plan_trajectory[:, :, 2] - ego_future[:, :, 2] + np.pi, 2 * np.pi) - np.pi)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])
    plannerAHE = torch.mean(heading_error)
    plannerFHE = torch.mean(heading_error[:, -1])
    
    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, neighbors_future_valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, neighbors_future_valid[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), plannerAHE.item(), plannerFHE.item(), predictorADE.item(), predictorFDE.item()
