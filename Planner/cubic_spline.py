import math
import numpy as np
import bisect
import logging

class Spline1D:
    """
    1D Spline Interpolation class, utilizing cubic spline.
    """

    def __init__(self, x_points, y_points):
        self.x = x_points
        self.y = y_points
        self.num_points = len(x_points)

        # Validate that x_points are sorted
        if np.any(np.diff(x_points) < 0):
            raise ValueError("x points must be sorted in ascending order.")

        self.coefficients_a, self.coefficients_b, self.coefficients_c, self.coefficients_d = [], [], [], []

        # Coefficients for the spline
        self.coefficients_a = y_points
        h = np.diff(x_points)

        # Calculate coefficients for c, b, d
        A_matrix = self._compute_A_matrix(h)
        B_vector = self._compute_B_vector(h, self.coefficients_a)
        self.coefficients_c = np.linalg.solve(A_matrix, B_vector)

        for i in range(self.num_points - 1):
            dx = h[i]
            self.coefficients_b.append((self.coefficients_a[i + 1] - self.coefficients_a[i]) / dx - dx * (2 * self.coefficients_c[i] + self.coefficients_c[i + 1]) / 3)
            self.coefficients_d.append((self.coefficients_c[i + 1] - self.coefficients_c[i]) / (3 * dx))

    def _compute_A_matrix(self, h):
        """
        Computes matrix A for spline coefficient c.
        """
        A_matrix = np.zeros((self.num_points, self.num_points))
        A_matrix[0, 0] = 1
        for i in range(1, self.num_points - 1):
            A_matrix[i, i - 1] = h[i - 1]
            A_matrix[i, i] = 2 * (h[i - 1] + h[i])
            A_matrix[i, i + 1] = h[i]
        A_matrix[self.num_points - 1, self.num_points - 1] = 1
        return A_matrix

    def _compute_B_vector(self, h, a):
        """
        Computes matrix B for spline coefficient c.
        """
        B_vector = np.zeros(self.num_points)
        for i in range(1, self.num_points - 1):
            B_vector[i] = 3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1]
        return B_vector

    def _find_segment_index(self, x):
        """
        Finds the segment where x belongs to in the data.
        """
        return bisect.bisect(self.x, x) - 1

    def calculate_position(self, x):
        """
        Calculate the y position for a given x.
        """
        if x < self.x[0] or x > self.x[-1]:
            return None

        idx = self._find_segment_index(x)
        dx = x - self.x[idx]
        return self.coefficients_a[idx] + self.coefficients_b[idx] * dx + self.coefficients_c[idx] * dx ** 2 + self.coefficients_d[idx] * dx ** 3

    def calculate_first_derivative(self, x):
        """
        Calculate the first derivative (slope) at a given x.
        """
        if x < self.x[0] or x > self.x[-1]:
            return None

        idx = self._find_segment_index(x)
        dx = x - self.x[idx]
        return self.coefficients_b[idx] + 2 * self.coefficients_c[idx] * dx + 3 * self.coefficients_d[idx] * dx ** 2

    def calculate_second_derivative(self, x):
        """
        Calculate the second derivative at a given x.
        """
        if x < self.x[0] or x > self.x[-1]:
            return None

        idx = self._find_segment_index(x)
        dx = x - self.x[idx]
        return 2 * self.coefficients_c[idx] + 6 * self.coefficients_d[idx] * dx


class Spline2D:
    """
    2D Spline Interpolation class, utilizing cubic spline for both X and Y axes.
    """

    def __init__(self, x_points, y_points):
        self.s = self._calculate_s(x_points, y_points)
        self.sx = Spline1D(self.s, x_points)
        self.sy = Spline1D(self.s, y_points)

    def _calculate_s(self, x_points, y_points):
        dx = np.diff(x_points)
        dy = np.diff(y_points)
        distances = np.hypot(dx, dy)
        s = np.concatenate(([0], np.cumsum(distances)))
        return s

    def calculate_position(self, s):
        """
        Calculate the 2D position (x, y) at a given s (arc length).
        """
        x = self.sx.calculate_position(s)
        y = self.sy.calculate_position(s)
        return x, y

    def calculate_curvature(self, s):
        """
        Calculate curvature at a given s (arc length).
        """
        dx = self.sx.calculate_first_derivative(s)
        ddx = self.sx.calculate_second_derivative(s)
        dy = self.sy.calculate_first_derivative(s)
        ddy = self.sy.calculate_second_derivative(s)
        return (ddy * dx - ddx * dy) / ((dx**2 + dy**2) ** 1.5)

    def calculate_yaw(self, s):
        """
        Calculate yaw (direction) at a given s (arc length).
        """
        dx = self.sx.calculate_first_derivative(s)
        dy = self.sy.calculate_first_derivative(s)
        return math.atan2(dy, dx)


def compute_spline_course(x_points, y_points, ds=0.1):
    """
    Compute the cubic spline course for a set of points, including position, yaw, and curvature.
    """
    if len(x_points) < 2 or len(y_points) < 2:
        logging.error("Not enough data points to calculate spline course.")
        return None, None, None, None

    spline_2d = Spline2D(x_points, y_points)
    s_values = np.arange(0, spline_2d.s[-1], ds)
    rx, ry, ryaw, rk = [], [], [], []

    for s in s_values:
        x, y = spline_2d.calculate_position(s)
        rx.append(x)
        ry.append(y)
        ryaw.append(spline_2d.calculate_yaw(s))
        rk.append(spline_2d.calculate_curvature(s))

    return rx, ry, ryaw, rk
