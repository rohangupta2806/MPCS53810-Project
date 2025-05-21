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
        """
        x = np.asarray(x, dtype=float)
        # Handle both single points and arrays
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            return self.radius * (x / norm) if norm > 0 else x
        else:
            norm = np.linalg.norm(x, axis=-1, keepdims=True)
            return self.radius * (x / (norm + 1e-10))  # Add small constant to avoid division by zero

    @staticmethod
    def from_angles(theta, phi, radius=1):
        '''
        Create a point on the sphere's surface from spherical coordinates.

        Parameters
        ----------
        theta : The polar angle (inclination) in radians.
        type: float or np.ndarray
        phi : The azimuthal angle (longitude) in radians.
        type: float or np.ndarray

        Returns
        -------
        np.ndarray
            The Cartesian coordinates of the point on the sphere's surface.
            Returns a 1D array (shape (3,)) if theta and phi are scalar.
            Returns a 2D array (shape (n,3)) if theta and phi are 1D arrays.
        '''
        # Check if the original inputs were scalar before converting them to numpy arrays
        theta_is_scalar = np.isscalar(theta)
        phi_is_scalar = np.isscalar(phi)

        theta_arr = np.asarray(theta)
        phi_arr = np.asarray(phi)

        x = np.sin(theta_arr) * np.cos(phi_arr)
        y = np.sin(theta_arr) * np.sin(phi_arr)
        z = np.cos(theta_arr)

        if theta_is_scalar and phi_is_scalar:
            # For single point, ensure x, y, z are scalars before creating the 1D array
            return radius * np.array([x.item(), y.item(), z.item()])
        else:
            # For multiple points, return a 2D array with shape (n, 3)
            return radius * np.column_stack([x, y, z])

    def _make_cloud(self, n, rng):
        '''
        Generate a random cloud of points on the sphere's surface.

        Parameters
        ----------
        n : The number of points to generate.
        type: int
        rng : The random number generator.
        type: np.random.Generator

        Returns
        -------
        np.ndarray
            An array of shape (n, 3) containing the Cartesian coordinates of the points.
        '''

        theta = np.arccos(2 * rng.random(n) - 1)
        phi = 2 * np.pi * rng.random(n)
        return self.from_angles(theta, phi, self.radius)

    def mc_shares(self, pts, cloud):
        """
        Calculate the Monte Carlo shares for a given set of points on the sphere's surface.

        Parameters
        ----------
        pts : The points to calculate shares for.
        type: list
            A list of 1D numpy arrays (each of shape (3,)), representing player positions.
        cloud : The random cloud of points on the sphere's surface.
        type: np.ndarray

        Returns
        -------
        np.ndarray
            An array of shares for each player.
        """
        # If from_angles is fixed, pts (the input list) will contain 1D (3,) arrays.
        # np.asarray can directly convert this list of 1D arrays into a 2D (k,3) array.
        pts_array = np.asarray(pts, dtype=float)

        # For each point in cloud, find the nearest player point
        # cloud shape: (n, 3), pts_array shape: (k, 3)
        # cloud[:, None, :] shape: (n, 1, 3)
        # pts_array[None, :, :] shape: (1, k, 3)
        # diff shape: (n, k, 3)
        diff = cloud[:, None, :] - pts_array[None, :, :]

        # Use squared Euclidean distance (works the same as geodesic for finding closest)
        dist_sq = np.sum(diff**2, axis=2)                   # (n,k)

        # Find index of closest player for each cloud point
        owner = np.argmin(dist_sq, axis=1)                  # (n,)

        # Count how many points each player owns
        counts = np.bincount(owner, minlength=len(pts_array))

        # Return normalized counts (shares)
        return counts / len(cloud)

    def compute_payoffs(self, positions, n_samples=50_000):
        """
        Compute the payoffs for a given set of positions on the sphere's surface.

        Parameters
        ----------
        positions : The positions of the players.
        type: list
        n_samples : Number of Monte Carlo samples to use
        type: int, optional

        Returns
        -------
        np.ndarray
            The payoffs for each player.
        """
        # Project positions to ensure they're on the sphere
        positions = [self.project(pos) for pos in positions]

        # Generate a random cloud of points
        rng = np.random.default_rng()
        cloud = self._make_cloud(n_samples, rng)

        # Calculate shares using mc_shares
        return self.mc_shares(np.asarray(positions), cloud)