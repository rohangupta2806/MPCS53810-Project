import numpy as np
from project_code.geometry.domain import Domain

'''
Class representing a circle in 2D space implementing the Domain superclass.
'''

class Circle(Domain):
    """
    Class representing a circle in 2D space.
    """

    def __init__(self, radius):
        """
        Initialize the circle with a given radius.

        Parameters
        ----------
        radius : The radius of the circle.
        type: float
        """
        self.radius = radius

    def distance(self, x, y):
        """
        Calculate the distance between two points in the circle.

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
        Project a point onto the circle.

        Parameters
        ----------
        x : The point to project.
        type: list

        Returns
        -------
        list
            The projected point.
        """
        norm = np.linalg.norm(x)
        if norm > self.radius:
            return (x / norm) * self.radius
        else:
            return x