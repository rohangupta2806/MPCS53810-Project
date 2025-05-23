import numpy as np

'''
Domain interface for the simulation of Hotelling's model in N dimensions.
'''

class Domain:
    """
    Abstract base class for a domain in which to run the simulation.
    """

    def distance(self, x, y):
        """
        Calculate the distance between two points in the domain.
        This might be Euclidean or geodesic depending on the domain.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def project(self, x):
        """
        Project a point onto the domain.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def draw_boundary(self, ax):
        """
        Draw the boundary of the domain on the given axis.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def _make_cloud(self, n_points, rng):
        """
        Generate a cloud of points within/on the domain.
        Must be implemented by subclasses to ensure correct distribution.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def mc_shares(self, pts_array, cloud):
        """
        Monte-Carlo shares for the k players in `pts_array`, using a *given*
        cloud of demand points. Returns length-k array summing to 1.
        `pts_array` is assumed to be a 2D NumPy array of shape (k, num_dimensions).
        `cloud` is assumed to be a 2D NumPy array of shape (n, num_dimensions).
        Player positions in `pts_array` should already be projected onto the domain.
        """
        if pts_array.shape[0] == 0: # No players
            return np.array([])
        if cloud.shape[0] == 0: # No cloud points
            return np.zeros(pts_array.shape[0])

        # diff will have shape (n, k, num_dimensions)
        diff  = cloud[:, None, :] - pts_array[None, :, :]
        # (diff**2).sum(axis=-1) gives squared Euclidean distances, shape (n, k)
        # owner will have shape (n,)
        owner = np.argmin((diff**2).sum(axis=-1), axis=1)
        # counts will have shape (k,)
        counts = np.bincount(owner, minlength=pts_array.shape[0])
        return counts / cloud.shape[0]

    def compute_payoff(self, positions, *, cloud=None, n_samples=50_000):
        """
        Monte-Carlo pay-off vector. If `cloud` is supplied, we reuse it;
        otherwise, we allocate one ad hoc.
        Positions are projected onto the domain before calculating payoffs.
        """
        if not positions: # No players
            return np.array([])

        # Project all positions onto the domain using the subclass's project method
        projected_positions = [self.project(pos) for pos in positions]

        # Ensure pts_array is a 2D numpy array.
        # The number of dimensions will be inferred from the first projected point.
        if not projected_positions: # Should not happen if initial positions list was not empty
             return np.array([])

        # Attempt to convert to a robust 2D array
        try:
            pts_array = np.array(projected_positions, dtype=float)
            if pts_array.ndim == 1 and len(projected_positions) == 1: # Single player, ensure it's 2D
                pts_array = pts_array.reshape(1, -1)
            elif pts_array.ndim == 0: # Should not happen
                 return np.array([])
        except ValueError as e:
            # Handle cases where positions might be jagged if project returns different types/shapes
            # This is less likely if project is consistent
            raise ValueError(f"Could not convert projected_positions to a uniform NumPy array: {projected_positions}. Error: {e}")


        if cloud is None: # slow one-off call
            print("Warning: No cloud provided, generating a new one.")
            rng   = np.random.default_rng()
            cloud = self._make_cloud(n_samples, rng) # Uses subclass's _make_cloud

        return self.mc_shares(pts_array, cloud) # Calls the base class mc_shares



