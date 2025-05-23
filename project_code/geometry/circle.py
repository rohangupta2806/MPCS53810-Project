import numpy as np
from project_code.geometry.domain import Domain
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

    def compute_payoff(domain, positions, *, cloud=None, n_samples=50_000):
        """
        Monte-Carlo pay-off vector.  If `cloud` is supplied we reuse it,
        otherwise we allocate one ad hoc (slow path).
        """
        pts = np.asarray(positions, float).reshape(len(positions), -1)

        if cloud is None:                        # slow one-off call
            rng   = np.random.default_rng()
            cloud = domain._make_cloud(n_samples, rng)

        return domain.mc_shares(pts, cloud)

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

