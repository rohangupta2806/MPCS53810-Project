import scipy
import os
import sys
from project_code.geometry.circle import Circle
from project_code.geometry.sphere import Sphere


def sim(domain=None, num_players=None, start_positions=None, tol=None, max_iter=None):
    """
    Simulate Hotelling in N dimensions with the given parameters.and

    Parameters
    ----------
    domain : The domain in which to run the simulation.
    type: Domain
    num_players : The number of players in the simulation.
    type: int
    start_positions : The starting positions of the players. Fed as a list of vectors of R^N.
    type: list
    tol : The tolerance for convergence.
    type: float
    max_iter : The maximum number of iterations to run the simulation.
    type: int
    """
    if domain is None:
        raise ValueError("Domain must be provided.")
    if num_players is None:
        raise ValueError("Number of players must be provided.")
    if start_positions is None:
        raise ValueError("Starting positions must be provided.")
    if tol is None:
        raise ValueError("Tolerance must be provided.")
    if max_iter is None:
        raise ValueError("Maximum iterations must be provided.")

    if domain not in ["circle", "sphere"]:
        raise ValueError("Domain must be either 'circle' or 'square'.")
    if num_players <= 0:
        raise ValueError("Number of players must be greater than 0.")
    if tol <= 0:
        raise ValueError("Tolerance must be greater than 0.")
    if max_iter <= 0:
        raise ValueError("Maximum iterations must be greater than 0.")
    if len(start_positions) != num_players:
        raise ValueError("Starting positions must be a list of length equal to the number of players.")

    if domain == "circle":
        domain = Circle(radius=1)
    elif domain == "sphere":
        domain = Sphere(radius=1)


if __name__ == "__main__":
    # Feed in the file name from the command line

    file = sys.argv[1]

    # Check if the file exists
    if not os.path.exists(file):
        print(f"File {file} does not exist.")
        sys.exit(1)

    # If file exists, read the file and run the simulation
    with open(file, 'r') as f:
        lines = f.readlines()
        domain = lines[0].strip()
        num_players = int(lines[1].strip())
        start_positions = [list(map(float, line.strip().split())) for line in lines[2:2+num_players]]
        tol = float(lines[2+num_players].strip())
        max_iter = int(lines[3+num_players].strip())

    # Run the simulation
    print("Running simulation with the following parameters:")
    print(f"Domain: {domain}")
    print(f"Number of players: {num_players}")
    print(f"Starting positions: {start_positions}")
    print(f"Tolerance: {tol}")
    print(f"Max iterations: {max_iter}")
    sim(domain=domain, num_players=num_players, start_positions=start_positions, tol=tol, max_iter=max_iter)