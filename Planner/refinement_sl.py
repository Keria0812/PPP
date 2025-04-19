import torch
import theseus as th
import matplotlib.pyplot as plt
from common_utils import *


def curvature_constraint(optim_vars, aux_vars, max_curvature=0.2):
    sl = optim_vars[0].tensor.view(-1, 1200, 2)
    s = sl[:, :, 0]
    l = sl[:, :, 1]

    ds = torch.diff(s, dim=1)
    dl = torch.diff(l, dim=1)

    dl_ds = dl / ds

    d2l_ds2 = torch.diff(dl_ds, dim=1) / torch.diff(s[:, :-1], dim=1)

    curvature = torch.abs(d2l_ds2) / (1 + dl_ds[:, :-1]**2)**(3/2)

    curvature_violation = torch.clamp(curvature - max_curvature, min=0)

    curvature_cost = curvature_violation**2

    return curvature_cost

def smoothness_cost(optim_vars, aux_vars):
    sl = optim_vars[0].tensor.view(-1, 1200, 2)
    s = sl[:, :, 0]
    l = sl[:, :, 1]

    ds = torch.diff(s, dim=1)
    dl = torch.diff(l, dim=1)
    dl_ds = dl / ds
    d2l_ds2 = torch.diff(dl_ds, dim=1) / torch.diff(s[:, :-1], dim=1)

    smoothness_cost = d2l_ds2**2

    return smoothness_cost

def speed_target(optim_vars, aux_vars):
    ds = optim_vars[0].tensor.view(-1, 1200, 2)
    speed_limit = aux_vars[0].tensor
    s = aux_vars[1].tensor
    s = (s * 10).long().clip(0, speed_limit.shape[1]-1)
    speed_limit = speed_limit[torch.arange(s.shape[0])[:, None], s]
    speed_error = ds - speed_limit

    return speed_error

def jerk(optim_vars, aux_vars):
    sl = optim_vars[0].tensor.view(-1, 1200, 2)

    s = sl[:, :, 0]
    l = sl[:, :, 1]

    ds = torch.diff(s, dim=1)
    dl = torch.diff(l, dim=1)
    dl_ds = dl / ds
    d2l_ds2 = torch.diff(dl_ds, dim=1) / torch.diff(s[:, :-1], dim=1)
    d3l_ds3 = torch.diff(d2l_ds2, dim=1) / torch.diff(s[:, :-2], dim=1)

    jerk = d3l_ds3**2

    return jerk

def end_condition(optim_vars, aux_vars):
    sl = optim_vars[0].tensor.view(-1, 1200, 2)
    ref_line = aux_vars[0].tensor

    traj_end_s = sl[:, -1:, 0]
    traj_end_l = sl[:, -1:, 1]
    
    ref_end_s = ref_line[:, -1:, 0]
    ref_end_l = ref_line[:, -1:, 1]

    s_diff = traj_end_s - ref_end_s
    l_diff = traj_end_l - ref_end_l

    end_condition = s_diff ** 2 + l_diff ** 2

    return end_condition

def bound_check_cost(optim_vars, aux_vars):
    l = optim_vars[0].tensor.view(-1, 1200, 2)[:, :, 1]

    left_bound = aux_vars[0].tensor
    right_bound = aux_vars[1].tensor
    yaw = aux_vars[2].tensor

    nonzero_left_mask = left_bound != 0
    nonzero_right_mask = right_bound != 0

    nonzero_left_mask = nonzero_left_mask.float()
    nonzero_right_mask = nonzero_right_mask.float()

    delta_s = LENGTH

    delta_l_left = WIDTH / 2
    delta_l_right = -WIDTH / 2

    delta_l_front_left = delta_s * torch.sin(yaw) + delta_l_left * torch.cos(yaw)
    delta_l_front_right = delta_s * torch.sin(yaw) + delta_l_right * torch.cos(yaw)

    l_front_left = l + delta_l_front_left
    l_front_right = l + delta_l_front_right

    left_violation = torch.clamp(l_front_left - left_bound, min=0)
    right_violation = torch.clamp(right_bound - l_front_right, min=0)

    left_cost_masked = torch.where(nonzero_left_mask.bool(), left_violation, torch.zeros_like(left_violation))
    right_cost_masked = torch.where(nonzero_right_mask.bool(), right_violation, torch.zeros_like(right_violation))

    left_cost = torch.sum(left_cost_masked ** 2, dim=1, keepdim=True)
    right_cost = torch.sum(right_cost_masked ** 2, dim=1, keepdim=True)

    bound_check = torch.cat([left_cost, right_cost], dim=1)

    return bound_check

def lane_error(optim_vars, aux_vars):
    lane_error = optim_vars[0].tensor.view(-1, 1200, 2)[:, :, 1]

    return lane_error

def safety(optim_vars, aux_vars):
    sl = optim_vars[0].tensor.view(-1, 1200, 2)
    
    occupancy = aux_vars[1].tensor
    
    s = sl[:, :, 0]
    l = sl[:, :, 1]

    safety_cost_list = []

    s_front = s + LENGTH
    l_left = l - WIDTH / 2
    l_right = l + WIDTH / 2

    l_left_current = torch.clamp((l_left * 10).long() + 100, 0, 199)
    l_right_current = torch.clamp((l_right * 10).long() + 100, 0, 199)
    s_current = torch.clamp(s_front.long(), 0, 1199)

    l_range = torch.arange(0, 200, device=occupancy.device).unsqueeze(0)

    for i in range(500):
        s_idx = s_current[:, i]
        occ_at_s = occupancy[:, s_idx].squeeze(1)

        mask = (l_range >= l_left_current[0, i]) & (l_range <= l_right_current[0, i])

        masked_occ = occ_at_s * mask.float()

        safety_cost_list.append(torch.sum(masked_occ))

    safety_cost = torch.stack(safety_cost_list).unsqueeze(0)

    return safety_cost

class RefinementPlanner_sl:
    def __init__(self, device):
        self._device = device
        self.N = int(T/DT) # trajectory points (ds/dt)
        self.gains = {
            "speed": 1.0,
            "accel": 10.0,
            "jerk": 5.0,
            "end": 40.0,
            "soft_constraint": 30.0,
            "hard_constraint": 100.0
        }
        self.build_optimizer()

    def build_optimizer(self):
        control_variables = th.Vector(dof=2400, name="sl")
        weights = {k: th.ScaleCostWeight(th.Variable(torch.tensor(v), name=f'gain_{k}')) for k, v in self.gains.items()}
        ego_state = th.Variable(torch.empty(1, 7), name="ego_state")
        ref_line_info = th.Variable(torch.empty(1, MAX_LEN*10, 3), name="ref_line_info")
        occ_map_obstacles = th.Variable(torch.empty(1, MAX_LEN*10,200), name="occupancy")
        left_bound = th.Variable(torch.empty(1, MAX_LEN*10), name = "left_bound")
        right_bound = th.Variable(torch.empty(1, MAX_LEN*10), name = "right_bound")
        yaw = th.Variable(torch.empty(1, MAX_LEN*10), name = "yaw")

        objective = th.Objective()
        self.objective = self.build_cost_function(objective, control_variables, ego_state, ref_line_info, 
                                                   occ_map_obstacles, left_bound, right_bound, yaw, weights)
        self.optimizer = th.GaussNewton(objective, th.CholeskyDenseSolver, 
                                        max_iterations=50, 
                                        step_size=0.3,
                                        rel_err_tolerance=1e-3)
        
        self.layer = th.TheseusLayer(self.optimizer, vectorize=False)
        self.layer.to(device=self._device)

    def build_cost_function(self, objective, control_variables, ego_state, ref_line_info, occ_map_obstacles, left_bound, right_bound, yaw, weights, vectorize=True):

        curvature_cost = th.AutoDiffCostFunction([control_variables], curvature_constraint, 1198, weights['accel'], #0.1
                                             autograd_vectorize=vectorize, name="curvature_cost")
        objective.add(curvature_cost)


        smo_cost = th.AutoDiffCostFunction([control_variables], smoothness_cost, 1198, weights['jerk'],#0.3
                                            autograd_vectorize=vectorize, name="smoothness_cost")
        objective.add(smo_cost)



        jerk_cost = th.AutoDiffCostFunction([control_variables], jerk, 1197, weights['jerk'],#0.3
                                            aux_vars=[ego_state], autograd_vectorize=vectorize, name="jerk")
        objective.add(jerk_cost)
        

        safety_cost = th.AutoDiffCostFunction([control_variables], safety, 500, weights['hard_constraint'], 
                                               aux_vars=[ego_state, occ_map_obstacles], autograd_vectorize=vectorize, name="safety")
        objective.add(safety_cost)


        end_cost = th.AutoDiffCostFunction([control_variables], end_condition, 1, weights['end'],
                                           aux_vars=[ref_line_info], autograd_vectorize=vectorize, name="end_condition")
        objective.add(end_cost)

        bound_cost = th.AutoDiffCostFunction([control_variables], bound_check_cost, 2, weights['hard_constraint'],
                                           aux_vars=[left_bound, right_bound, yaw], autograd_vectorize=vectorize, name="bound_check")
        objective.add(bound_cost)       

        l_cost = th.AutoDiffCostFunction([control_variables], lane_error, 1200, weights['soft_constraint'],
                                            autograd_vectorize=vectorize, name="lane_error")
        objective.add(l_cost)       

        return objective


    def plan(self, ego_state, best_lateral_sampling_frenet, occ_map_obstacles, best_ref_path, left_bound, right_bound, yaw):
        # initial plan

        sl = best_lateral_sampling_frenet.to(torch.float32)

        occ_map_obstacles = occ_map_obstacles.to(torch.float32)
        left_bound= left_bound.to(torch.float32)
        right_bound = right_bound.to(torch.float32)
        ego_state = ego_state.to(torch.float32)
        best_ref_path = best_ref_path.to(torch.float32)
        yaw = yaw.to(torch.float32)


        # update planner inputs
        planner_inputs = {
            "sl": sl.view(-1,2400),
            "occupancy": occ_map_obstacles,
            "ego_state": ego_state,
            "ref_line_info": best_ref_path,
            "left_bound": left_bound,
            "right_bound": right_bound,
            "yaw": yaw
        }
        
        # plan
        _, info = self.layer.forward(planner_inputs, optimizer_kwargs={'track_best_solution': True})
        sl = info.best_solution['sl'].view(-1, 1200, 2).to(self._device)
        
        return sl

