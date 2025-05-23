import numpy as np
from project_code.geometry.domain import Domain
import matplotlib.pyplot as plt

'''
Class representing a square in 2D space implementing the Domain superclass.
'''

class Square(Domain):
    """
    Class representing a square in 2D space.
    """

    def __init__(self, side_length):
        """
        Initialize the square with a given side length.

        Parameters
        ----------
        side_length : The length of the sides of the square.
        type: float
        """
        self.side_length = side_length

    def distance(self, x, y):
        """
        Calculate the distance between two points in the square.

        Parameters
        ----------
        x : The first point.
        type: list
        y : The second point.
        type: list

        Returns
        -------
        float
            The distance between the two points.
        """
        return np.linalg.norm(np.array(x) - np.array(y))

    def project(self, x):
        """
        Project a point onto the square.
        The square is centered at the origin.

        Parameters
        ----------
        x : The point to project, as a NumPy array or list.
        type: np.ndarray or list

        Returns
        -------
        np.ndarray
            The projected point.
        """
        x = np.asarray(x, dtype=float)
        half_side = self.side_length / 2.0
        # np.clip works element-wise, suitable for single points (1D array) or multiple points (2D array)
        return np.clip(x, -half_side, half_side)

    def _make_cloud(self, n, rng):
        """
        Generate a cloud of n random points uniformly distributed within the square.
        The square is centered at the origin.

        Parameters
        ----------
        n : Number of points to generate.
        type: int
        rng : NumPy random number generator instance.
        type: np.random.Generator

        Returns
        -------
        np.ndarray
            An array of shape (n, 2) representing the cloud of points.
        """
        half_side = self.side_length / 2.0
        # Generate random x and y coordinates in the range [-half_side, half_side]
        xs = rng.uniform(-half_side, half_side, n)
        ys = rng.uniform(-half_side, half_side, n)
        return np.column_stack([xs, ys])

    def draw_boundary(self, ax):
        """
        Draw the boundary of the square on the given axis.
        The square is centered at the origin.

        Parameters
        ----------
        ax : The axis on which to draw the boundary.
        type: matplotlib.axes.Axes
        """
        half_side = self.side_length / 2.0
        square_shape = plt.Rectangle((-half_side, -half_side), self.side_length, self.side_length,
                                     edgecolor='gray', facecolor='none', fill=False)
        ax.add_patch(square_shape)
        ax.set_xlim(-half_side * 1.1, half_side * 1.1)
        ax.set_ylim(-half_side * 1.1, half_side * 1.1)
        ax.set_aspect('equal', adjustable='box')