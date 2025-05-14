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
        raise NotImplementedError("Subclasses must implement this method.")

    def project(self, x):
        """
        Project a point onto the domain.

        Parameters
        ----------
        x : The point to project.
        type: list

        Returns
        -------
        list
            The projected point.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def voronoi_payoffs(self, positions):
        """
        Calculate the Voronoi payoffs for a given set of positions, or
        return a not implemented error if the domain does not support Voronoi payoffs.

        Parameters
        ----------
        positions : The positions of the players.
        type: list

        Returns
        -------
        list
            The Voronoi payoffs for each player.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def draw_boundary(self, ax):
        """
        Draw the boundary of the domain on the given axis.

        Parameters
        ----------
        ax : The axis on which to draw the boundary.
        type: matplotlib.axes.Axes
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def voronoi_finite_polygons_2d(self, vor, radius=10):
        """
        Reconstruct finite polygons from Voronoi regions.

        Returns
        -------
        regions  : list of lists of vertices indices
        vertices : ndarray of vertices coords
        """
        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2-D input")
        new_regions, new_vertices = [], vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()*2

        # Map ridge vertices to ridges
        all_ridges = {}
        for (p,q), (v1,v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p, []).append((q, v1, v2))
            all_ridges.setdefault(q, []).append((p, v1, v2))

        # Reconstruct each finite region
        for p, reg_idx in enumerate(vor.point_region):
            verts = vor.regions[reg_idx]
            if -1 not in verts:
                new_regions.append(verts)
                continue

            # Local ridge vertices
            ridges = all_ridges[p]
            new_region = [v for v in verts if v != -1]

            for q, v1, v2 in ridges:
                if v2 < 0: v1, v2 = v2, v1
                if v1 >= 0 and v2 >= 0:
                    # finite ridge: already in region
                    continue

                # Compute a new vertex at "far away"
                t = vor.points[q] - vor.points[p]     # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])           # normal
                midpoint = vor.points[[p,q]].mean(axis=0)
                direction = np.sign(np.dot(midpoint-center, n)) * n
                faraway = vor.vertices[v2] + direction*radius

                new_vertices.append(faraway.tolist())
                new_region.append(len(new_vertices)-1)

            new_regions.append(new_region)

        return new_regions, np.asarray(new_vertices)