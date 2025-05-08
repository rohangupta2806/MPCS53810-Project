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
        # Calculate the great-circle distance using the haversine formula
        x = np.array(x)
        y = np.array(y)

        dlat = np.radians(y[0] - x[0])
        dlon = np.radians(y[1] - x[1])

        a = np.sin(dlat / 2)**2 + np.cos(np.radians(x[0])) * np.cos(np.radians(y[0])) * np.sin(dlon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return self.radius * c

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
        norm = np.linalg.norm(x)
        if norm > self.radius:
            return (x / norm) * self.radius
        else:
            return x

    @staticmethod
    def from_angles(theta, phi):
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

        return [x, y, z]

