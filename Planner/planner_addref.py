import math
import time
import matplotlib.pyplot as plt
from shapely import Point, LineString
from .planner_utils import *
from .observation_addref import *
from PPP.predictor import PPP
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner, PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring


class Planner(AbstractPlanner):
    def __init__(self, model_path, device=None):
        self._max_path_length = MAX_LEN # [m]
        self._future_horizon = T # [s] 
        self._step_interval = DT # [s]
        self._target_speed = 13.0 # [m/s]
        self._N_points = int(T/DT)
        self._model_path = model_path

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        self._device = device
    
    def name(self) -> str:
        return "PPP Planner"
    
    def observation_type(self):
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization):
        self._map_api = initialization.map_api
        self._goal = initialization.mission_goal
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = TrajectoryPlanner()


    def _initialize_model(self):
        # The parameters of the model should be the same as the one used in training
        self._model = PPP()
        
        # Load trained model
        self._model.load_state_dict(torch.load(self._model_path, map_location=self._device))
        self._model.to(self._device)
        self._model.eval()
        
    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            self._route_roadblocks.append(block)

        self._candidate_lane_edge_ids = [
            edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        ]

    def _get_prediction(self, features):
        predictions, plan = self._model(features)
        final_predictions = predictions['agents_pred'][:, 1:]
        final_scores = predictions['scores']
        ego_current = features['ego_agent_past'][:, -1]
        neighbors_current = features['neighbor_agents_past'][:, :, -1]

        return plan, final_predictions, final_scores, ego_current, neighbors_current
    
    def _plan(self, ego_state, history, traffic_light_data, observation, iteration):
        # Construct input features
        features = observation_adapter_addref(history, ego_state, observation, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        # Get reference path
        ref_path = get_reference_path(ego_state, traffic_light_data, observation, self._map_api, self._route_roadblock_ids)

        # Infer prediction model
        with torch.no_grad():
            plan, predictions, scores, ego_state_transformed, neighbors_state_transformed = self._get_prediction(features)


        # Trajectory refinement
        with torch.no_grad():
            plan = self._trajectory_planner.plan(ego_state, ego_state_transformed, neighbors_state_transformed, 
                                                 predictions, plan, scores, ref_path, observation, features, iteration, self._map_api, self._route_roadblock_ids, traffic_light_data)
            
        states = transform_predictions_to_states(plan, history.ego_states, self._future_horizon, DT)
        trajectory = InterpolatedTrajectory(states)

        return trajectory
    
    def compute_planner_trajectory(self, current_input: PlannerInput):
        s = time.time()
        iteration = current_input.iteration.index
        history = current_input.history
        traffic_light_data = list(current_input.traffic_light_data)
        ego_state, observation = history.current_state
        trajectory = self._plan(ego_state, history, traffic_light_data, observation, iteration)
        print(f'Iteration {iteration}: {time.time() - s:.3f} s')

        return trajectory


