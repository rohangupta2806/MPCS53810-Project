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




