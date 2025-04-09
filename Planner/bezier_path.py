import numpy as np
import scipy.special

def compute_control_points_and_trajectory(start_x, start_y, start_angle, end_x, end_y, end_angle, curve_factor, num_points=100):
    """
    Compute control points and trajectory based on start and end points.

    :param start_x: (float) x-coordinate of the start point
    :param start_y: (float) y-coordinate of the start point
    :param start_angle: (float) starting direction angle
    :param end_x: (float) x-coordinate of the end point
    :param end_y: (float) y-coordinate of the end point
    :param end_angle: (float) ending direction angle
    :param curve_factor: (float) factor controlling the curve tightness
    :return: (numpy array, numpy array) - trajectory and control points
    """
    path_distance = np.hypot(start_x - end_x, start_y - end_y) / curve_factor
    control_points = np.array(
        [[start_x, start_y],
         [start_x + path_distance * np.cos(start_angle), start_y + path_distance * np.sin(start_angle)],
         [end_x - path_distance * np.cos(end_angle), end_y - path_distance * np.sin(end_angle)],
         [end_x, end_y]])

    trajectory = generate_trajectory(control_points, num_points=num_points)

    return trajectory, control_points


def generate_trajectory(control_points, num_points=100):
    """
    Generate trajectory using control points.

    :param control_points: (numpy array) - set of control points
    :param num_points: (int) - number of points in the trajectory
    :return: (numpy array) - calculated trajectory
    """
    trajectory = []
    for t in np.linspace(0, 1, num_points):
        trajectory.append(compute_point_on_curve(t, control_points))

    return np.array(trajectory)


def compute_polynomial(degree, index, t):
    """
    Compute Bernstein polynomial value.

    :param degree: (int) degree of the polynomial
    :param index: (int) index of the polynomial term
    :param t: (float) parameter ranging from 0 to 1
    :return: (float) - polynomial value
    """
    return scipy.special.comb(degree, index) * t ** index * (1 - t) ** (degree - index)


def compute_point_on_curve(t, control_points):
    """
    Compute a point on the curve at parameter t using control points.

    :param t: (float) parameter in [0, 1]
    :param control_points: (numpy array) - control points
    :return: (numpy array) - calculated point coordinates
    """
    degree = len(control_points) - 1
    return np.sum([compute_polynomial(degree, i, t) * control_points[i] for i in range(degree + 1)], axis=0)


def compute_derivatives_of_curve(control_points, num_derivatives):
    """
    Compute control points for successive derivatives of the curve.

    :param control_points: (numpy array) - control points
    :param num_derivatives: (int) - number of derivatives to compute
    :return: ([numpy array]) - control points for each derivative
    """
    results = {0: control_points}
    for i in range(num_derivatives):
        n = len(results[i])
        results[i + 1] = np.array([(n - 1) * (results[i][j + 1] - results[i][j])
                                 for j in range(n - 1)])
    return results


def calculate_curvature(dx, dy, ddx, ddy):
    """
    Compute curvature at a point given first and second derivatives.

    :param dx: (float) first derivative with respect to x
    :param dy: (float) first derivative with respect to y
    :param ddx: (float) second derivative with respect to x
    :param ddy: (float) second derivative with respect to y
    :return: (float) - curvature value
    """
    return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)
