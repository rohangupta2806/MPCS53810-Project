import numpy as np
from project_code.geometry.domain import Domain
from scipy.spatial import Voronoi, voronoi_plot_2d
from shapely.geometry import Polygon, Point, LineString
import matplotlib.pyplot as plt

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

    def voronoi_payoffs(self, positions):
        """
        Calculate the Voronoi payoffs for a given set of positions.

        Parameters
        ----------
        positions : The positions of the players.
        type: list

        Returns
        -------
        list
            The Voronoi payoffs for each player.
        """
        # Calculate the Voronoi payoffs for each player
        payoffs = []
        vor = Voronoi(positions)
        regions, vertices = self.voronoi_finite_polygons_2d(vor, radius=5*self.radius)
        circle = Point(0, 0).buffer(self.radius)
        # Clip the Voronoi regions to the circle
        for reg in regions:
            poly = Polygon(vertices[reg]).intersection(circle)
            payoffs.append(poly.area / (np.pi * self.radius**2))
        return payoffs

    def draw_boundary(self, ax):
        """
        Draw the boundary of the circle on the given axis.

        Parameters
        ----------
        ax : The axis on which to draw the boundary.
        type: matplotlib.axes.Axes
        """
        circle = plt.Circle((0, 0), self.radius, color='gray', fill=False)
        ax.add_artist(circle)
        ax.set_xlim(-self.radius, self.radius)
        ax.set_ylim(-self.radius, self.radius)
        ax.set_aspect('equal')

