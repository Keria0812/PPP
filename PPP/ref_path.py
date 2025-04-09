import scipy
import numpy as np
import matplotlib.pyplot as plt
from common_utils import *
from shapely import Point, LineString
from shapely.geometry.base import CAP_STYLE
from Planner.bezier_path import compute_control_points_and_trajectory
from Planner.cubic_spline import compute_spline_course
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from nuplan.planning.metrics.utils.expert_comparisons import principal_value

MAX_PATH_LENGTH = 120

class Ref_path:
    def __init__(self, route_ids, max_len=MAX_PATH_LENGTH):
        self.max_path_len = max_len
        self.route_edge_ids = route_ids
        self.max_len = max_len

    def find_candidate_paths(self, edges):
        '''Retrieve candidate paths through depth-first search'''
        paths = []
        for edge in edges:
            paths.extend(self.depth_first_search(edge))

        path_candidates = {}

        # Extract path polyline
        for i, path in enumerate(paths):
            path_polyline = []
            for edge in path:
                path_polyline.extend(edge.baseline_path.discrete_path)

            path_polyline = self.optimize_path(np.array(path_to_linestring(path_polyline).coords))
            dist_to_vehicle = scipy.spatial.distance.cdist([self.vehicle_position], path_polyline)
            path_polyline = path_polyline[dist_to_vehicle.argmin():]

            if len(path_polyline) < 3:
                continue

            path_length = len(path_polyline) * 0.25
            polyline_headings = self.compute_path_headings(path_polyline)
            path_polyline = np.stack([path_polyline[:, 0], path_polyline[:, 1], polyline_headings], axis=1)
            path_candidates[i] = (path_length, dist_to_vehicle.min(), path, path_polyline)

        if len(path_candidates) == 0:
            return None

        # Filter paths by length
        self.path_len = max([v[0] for v in path_candidates.values()])
        acceptable_path_len = MAX_PATH_LENGTH * 0.2 if self.path_len > MAX_PATH_LENGTH * 0.2 else self.path_len
        path_candidates = {k: v for k, v in path_candidates.items() if v[0] >= acceptable_path_len}

        # Sort paths by proximity to vehicle
        path_candidates = sorted(path_candidates.items(), key=lambda x: x[1][1])

        return path_candidates

    def identify_candidate_edges(self, starting_block, vehicle_state):
        '''Identify candidate edges from the initial block'''
        candidate_edges = []
        edge_distances = []
        self.vehicle_position = (vehicle_state.rear_axle.x, vehicle_state.rear_axle.y)
        self.num_edges = len(starting_block.interior_edges)

        for edge in starting_block.interior_edges:
            edge_distances.append(edge.polygon.distance(Point(self.vehicle_position)))
            if edge.polygon.distance(Point(self.vehicle_position)) < 4:
                candidate_edges.append(edge)

        # If no edge is close to the vehicle, pick the closest edge
        if len(candidate_edges) == 0:
            candidate_edges.append(starting_block.interior_edges[np.argmin(edge_distances)])

        return candidate_edges

    def plan_route(self, vehicle_state, starting_block, observation, traffic_data):
        # Get candidate paths
        edges = self.identify_candidate_edges(starting_block, vehicle_state)
        candidate_paths = self.find_candidate_paths(edges)
        #print(candidate_paths)
        if candidate_paths is None:
            return None

        # Extract obstacles
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BARRIER,
                        TrackedObjectType.CZONE_SIGN, TrackedObjectType.TRAFFIC_CONE,
                        TrackedObjectType.GENERIC_OBJECT]
        objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)
        #print(objects)

        obstacles = []
        vehicles = []
        for obj in objects:
            if obj.box.geometry.distance(vehicle_state.car_footprint.geometry) > 30:
                continue

            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                if obj.velocity.magnitude() < 0.01:
                    obstacles.append(obj.box)
                else:
                    vehicles.append(obj.box)
            else:
                obstacles.append(obj.box)

        # Generate paths using state lattice
        paths = self.create_paths(vehicle_state, candidate_paths)
        #print(paths)

        # Avoid lane changes in large intersections
        if len(traffic_data) > 0:
            self.prevent_lane_change = True
        elif self.num_edges >= 4 and vehicle_state.dynamic_car_state.rear_axle_velocity_2d.x <= 3:
            self.prevent_lane_change = True
        else:
            self.prevent_lane_change = False

        optimal_paths = []
        for i in range(0, len(paths), 4):
            min_cost = np.inf
            optimal_path = None
            for path in paths[i:i + 4]:
                cost = self.compute_path_cost(path, obstacles, vehicles)
                if cost < min_cost:
                    min_cost = cost
                    optimal_path = path[0]
            refined_path = self.refine_path(optimal_path, vehicle_state)
            #print(refined_path.shape)

            if refined_path is None:
                continue

            if refined_path.shape[0] < MAX_PATH_LENGTH * 10:
                refined_path = np.append(refined_path, np.repeat(refined_path[np.newaxis, -1], MAX_PATH_LENGTH * 10 - len(refined_path), axis=0), axis=0)

            optimal_paths.append(refined_path[:MAX_PATH_LENGTH * 10])

        if len(optimal_paths) > 0:
            stacked_paths = np.stack(optimal_paths)
            return stacked_paths
        else:
            return None

    def create_paths(self, vehicle_state, paths):
        '''Create paths using state lattice'''
        generated_paths = []
        vehicle_state_position = vehicle_state.rear_axle.x, vehicle_state.rear_axle.y, vehicle_state.rear_axle.heading

        for _, (path_len, dist, path, path_polyline) in paths:
            if len(path_polyline) > 81:
                sample_indices = np.array([5, 10, 15, 20]) * 4
            elif len(path_polyline) > 61:
                sample_indices = np.array([5, 10, 15]) * 4
            elif len(path_polyline) > 41:
                sample_indices = np.array([5, 10]) * 4
            elif len(path_polyline) > 21:
                sample_indices = [20]
            else:
                sample_indices = [1]

            target_states = path_polyline[sample_indices].tolist()
            for j, state in enumerate(target_states):
                first_stage_path = compute_control_points_and_trajectory(vehicle_state_position[0], vehicle_state_position[1], vehicle_state_position[2],
                                                            state[0], state[1], state[2], 3, sample_indices[j])[0]
                second_stage_path = path_polyline[sample_indices[j] + 1:, :2]
                path_polyline = np.concatenate([first_stage_path, second_stage_path], axis=0)
                generated_paths.append((path_polyline, dist, path, path_len))

        return generated_paths

    def compute_path_cost(self, path, obstacles, vehicles):
        # path curvature
        path_curvature = self.calculate_path_curvature(path[0][:100])
        max_curvature = np.max(path_curvature)

        # lane change penalty
        lane_change = path[1]
        if self.prevent_lane_change:
            lane_change = 5 * lane_change

        # Target lane alignment
        target_alignment = self.check_target_lane(path[0][:50], path[3], vehicles)

        # Obstacle checking
        obstacle_check = self.check_obstacles(path[0][:100], obstacles)

        # Boundary crossing check
        boundary_check = self.check_out_boundary(path[0][:100], path[2])

        # Final cost calculation
        total_cost = 10 * obstacle_check + 2 * boundary_check + 1 * lane_change + 0.1 * max_curvature - 5 * target_alignment

        return total_cost

    def refine_path(self, path, vehicle_state):
        index = np.arange(0, len(path), 10)
        x = path[:, 0][index]
        y = path[:, 1][index]
        rx, ry, ryaw, rk = compute_spline_course(x, y)
        if rx is None:
            return None

        spline_path = np.stack([rx, ry, ryaw, rk], axis=1)
        refined_path = self.transform_to_vehicle_frame(spline_path, vehicle_state)
        refined_path = refined_path[:self.max_len * 10]

        return refined_path

    def depth_first_search(self, starting_edge, depth=0):
        if depth >= self.max_len:
            return [[starting_edge]]
        else:
            traversed_edges = []
            child_edges = [edge for edge in starting_edge.outgoing_edges if edge.id in self.route_edge_ids]

            if child_edges:
                for child in child_edges:
                    edge_len = len(child.baseline_path.discrete_path) * 0.25
                    traversed_edges.extend(self.depth_first_search(child, depth + edge_len))

            if len(traversed_edges) == 0:
                return [[starting_edge]]

            edges_to_return = []
            for edge_seq in traversed_edges:
                edges_to_return.append([starting_edge] + edge_seq)
            return edges_to_return


    def check_target_lane(self,path, path_len, vehicles):
        if np.abs(path_len - self.path_len) > 5:
            return 0

        expanded_path = LineString(path).buffer((WIDTH / 2), cap_style=CAP_STYLE.square)
        min_distance = np.inf

        for vehicle in vehicles:
            distance = expanded_path.distance(vehicle.geometry)
            if distance < min_distance:
                min_distance = distance

        if min_distance < 5:
            return 0

        return 1

    @staticmethod
    def optimize_path(path):
        refined_path = [path[0]]

        for i in range(1, path.shape[0]):
            if np.linalg.norm(path[i] - path[i-1]) < 0.1:
                continue
            else:
                refined_path.append(path[i])

        line = np.array(refined_path)

        return line

    @staticmethod
    def compute_path_headings(path):
        headings = np.arctan2(path[1:, 1] - path[:-1, 1], path[1:, 0] - path[:-1, 0])
        headings = np.append(headings, headings[-1])

        return headings

    @staticmethod
    def check_obstacles(path, obstacles):
        expanded_path = LineString(path).buffer((WIDTH / 2), cap_style=CAP_STYLE.square)

        for obstacle in obstacles:
            obstacle_polygon = obstacle.geometry
            if expanded_path.intersects(obstacle_polygon):
                return 1

        return 0

    @staticmethod
    def check_out_boundary(polyline, path):
        line = LineString(polyline).buffer((WIDTH / 2), cap_style=CAP_STYLE.square)

        for edge in path:
            left, right = edge.adjacent_edges
            if (left is None and line.intersects(edge.left_boundary.linestring)) or \
                (right is None and line.intersects(edge.right_boundary.linestring)):
                return 1

        return 0

    @staticmethod
    def calculate_path_curvature(path):
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**(3/2)

        return curvature

    @staticmethod
    def transform_to_vehicle_frame(path, vehicle_state):
        vehicle_x, vehicle_y, vehicle_heading = vehicle_state.rear_axle.x, vehicle_state.rear_axle.y, vehicle_state.rear_axle.heading
        path_x, path_y, path_heading, path_curvature = path[:, 0], path[:, 1], path[:, 2], path[:, 3]
        vehicle_frame_x = np.cos(vehicle_heading) * (path_x - vehicle_x) + np.sin(vehicle_heading) * (path_y - vehicle_y)
        vehicle_frame_y = -np.sin(vehicle_heading) * (path_x - vehicle_x) + np.cos(vehicle_heading) * (path_y - vehicle_y)
        vehicle_frame_heading = principal_value(path_heading - vehicle_heading)
        vehicle_frame = np.stack([vehicle_frame_x, vehicle_frame_y, vehicle_frame_heading, path_curvature], axis=-1)

        return vehicle_frame
