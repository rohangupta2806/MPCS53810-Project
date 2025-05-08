import numpy as np
from project_code.geometry.domain import Domain

'''
Class representing the surface of a sphere in 3-dimensional space implementing the Domain superclass.
'''

class Sphere(Domain):
    """
    Class representing the surface of a sphere in 3-dimensional space.
    """

    def __init__(self, radius):
        """
        Initialize the sphere with a given radius.

        Parameters
        ----------
        radius : The radius of the sphere.
        type: float
        """
        self.radius = radius

    def distance(self, x, y):
        """
        Calculate the distance between two points on the sphere's surface.

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

        # Project points onto the sphere's surface
        x_proj = self.project(x)
        y_proj = self.project(y)

        cosine = np.dot(x_proj, y_proj) / self.radius**2
        angle = np.arccos(np.clip(cosine, -1.0, 1.0))  # Clip to avoid numerical errors
        return angle * self.radius

    def project(self, x):
        """
        Project a point onto the sphere's surface.

        Parameters
        ----------
        x : The point to project.
        type: list

        Returns
        -------
        list
            The projected point on the sphere's surface.
        """

        x = np.asarray(x, dtype=float)
        norm = np.linalg.norm(x, axis = -1, keepdims=True)
        return self.radius * (x / norm) if norm > 0 else x

    @staticmethod
    def from_angles(theta, phi, radius=1):
        '''
        Create a point on the sphere's surface from spherical coordinates.

        Parameters
        ----------
        theta : The polar angle (inclination) in radians.
        type: float
        phi : The azimuthal angle (longitude) in radians.
        type: float

        Returns
        -------
        list
            The Cartesian coordinates of the point on the sphere's surface.
        '''

        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)

        return radius * np.array([x, y, z])