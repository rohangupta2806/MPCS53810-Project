import scipy
import numpy as np
import os
import sys
from project_code.geometry.circle import Circle
from project_code.geometry.sphere import Sphere
from matplotlib import pyplot as plt
from itertools import cycle
COLOR_CYCLE = cycle(plt.get_cmap("tab20").colors)   # 20 bright hues
import pathlib
from project_code.vis.plotting import plot_circle, plot_sphere, create_animation

def compute_payoff(domain, positions, *, cloud=None, n_samples=50_000):
    """
    Monte-Carlo pay-off vector (demand shares).

    Parameters
    ----------
    domain      : the Domain instance (Circle or Sphere)
    positions   : list/array of k player positions, shape (k, dim)
    cloud       : (n, dim) ndarray of demand points **reused** inside one
                  outer iteration; if None a fresh cloud is drawn ad hoc.
    n_samples   : how many points to draw for the ad-hoc cloud

    Returns
    -------
    numpy.ndarray, length k, summing to 1
    """
    # Convert positions to a uniform numpy array more carefully
    positions_array = []
    for pos in positions:
        # Ensure each position is a numpy array
        pos_array = np.asarray(pos, dtype=float)
        positions_array.append(pos_array)

    # make a cloud only when caller didn't supply one (slow path)
    if cloud is None:
        rng = np.random.default_rng()
        cloud = domain._make_cloud(n_samples, rng)

    return domain.mc_shares(positions_array, cloud)

def sim(domain=None, num_players=None, start_positions=None, tol=None, max_iter=None, samples_per_iter = None, regularization=None):
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
        raise ValueError("Maximum number of iterations to run the simulation.")
    if samples_per_iter is None:
        raise ValueError("Samples per iteration must be provided.")
    if regularization is None:
        raise ValueError("Regularization must be provided.")

    if domain not in ["circle", "sphere"]:
        raise ValueError("Domain must be either 'circle' or 'sphere'.")
    if num_players <= 0:
        raise ValueError("Number of players must be greater than 0.")
    if tol <= 0:
        raise ValueError("Tolerance must be greater than 0.")
    if max_iter <= 0:
        raise ValueError("Maximum iterations must be greater than 0.")
    if len(start_positions) != num_players:
        raise ValueError("Starting positions must be a list of length equal to the number of players.")

    outdir = pathlib.Path("frames")
    outdir.mkdir(parents=True, exist_ok=True)
    # Initialize player positions
    positions = start_positions

    if domain == "circle":
        domain = Circle(radius=1)
        # Ensure all positions are numpy arrays
        positions = [np.array(pos, dtype=float) for pos in start_positions]
    elif domain == "sphere":
        domain = Sphere(radius=1)
        # Convert start from angle to Cartesian coordinates
        positions = [Sphere.from_angles(float(theta), float(phi)) for theta, phi in start_positions]
        # Ensure all positions are numpy arrays with the same shape
        positions = [np.array(pos, dtype=float) for pos in positions]

    # Use scipy minimizer to implement best response dynamics
    # Payoff is the area of the circle/ sphere that is nearest to the player

    player_colors = [next(COLOR_CYCLE) for _ in range(num_players)]
    for iteration in range(max_iter):
        print(f"Iteration {iteration+1}/{max_iter}")

        # one shared cloud for this iteration
        rng   = np.random.default_rng(iteration)
        cloud = domain._make_cloud(samples_per_iter, rng)

        old_positions = positions.copy()
        new_positions = positions.copy()

        previous_payoff = compute_payoff(domain, old_positions, cloud=cloud)

        # --------------- best-response search --------------------------
        for i in range(num_players):
            def objective(x):
                # Ensure x has proper shape for projection
                x_reshaped = np.asarray(x, dtype=float)
                cand = domain.project(x_reshaped)
                pts = old_positions.copy()
                pts[i] = cand
                share_i = compute_payoff(domain, pts, cloud=cloud)[i]
                movement_cost = regularization * np.linalg.norm(np.array(cand) - np.array(old_positions[i]))
                return -share_i + movement_cost  # maximise

            # Ensure initial position is 1D
            initial_position = np.array(old_positions[i]).flatten()

            res = scipy.optimize.minimize(
                    objective,
                    initial_position,
                    method='Nelder-Mead')

            new_positions[i] = domain.project(res.x)

        positions = new_positions

        from scipy.spatial.distance import pdist
        print("min distance:", pdist(positions).min())

        payoffs = compute_payoff(domain, positions, cloud=cloud)
        for idx, p in enumerate(payoffs, 1):
            print(f" Player {idx}: share {p:.4f}")

        current_payoff = payoffs # payoffs are the shares

        # Replace the plotting code with calls to the new functions
        if isinstance(domain, Circle):
            plot_circle(domain, positions, player_colors, outdir, iteration)
        elif isinstance(domain, Sphere):
            plot_sphere(domain, positions, current_payoff, player_colors, outdir, iteration) # Pass current_payoff as shares

        # Check for convergence
        if np.linalg.norm(np.array(positions) - np.array(old_positions)) < tol\
                and np.linalg.norm(np.array(current_payoff) - np.array(previous_payoff)) < tol:
            print(f"Converged after {iteration} iterations.")
            break

    # Replace the animation code with a call to the new function
    create_animation(outdir, "hotelling.gif", fps=2)

if __name__ == "__main__":
    # Feed in the file name from the command line

    file = sys.argv[1]

    # Check if the file exists
    if not os.path.exists(file):
        print(f"File {file} does not exist.")
        sys.exit(1)

    def clean(line):
        """
        Strip the line of # comments
        """
        return line.split("#")[0]
    # If file exists, read the file and run the simulation
    with open(file, 'r') as f:
        lines = f.readlines()
        lines = [clean(line) for line in lines]

    domain = lines[0].strip()
    num_players = int(lines[1].strip())
    start_positions = [list(map(float, line.strip().split())) for line in lines[2:2+num_players]]
    tol = float(lines[2+num_players].strip())
    max_iter = int(lines[3+num_players].strip())
    samples_per_iter = int(lines[4+num_players].strip())
    regularization = float(lines[5+num_players].strip())

    # Run the simulation
    print("Running simulation with the following parameters:")
    print(f"Domain: {domain}")
    print(f"Number of players: {num_players}")
    print(f"Starting positions: {start_positions}")
    print(f"Tolerance: {tol}")
    print(f"Max iterations: {max_iter}")
    print(f"Samples per iteration: {samples_per_iter}")
    print(f"Regularization: {regularization}")
    sim(domain=domain, num_players=num_players, start_positions=start_positions,
        tol=tol, max_iter=max_iter, samples_per_iter=samples_per_iter, regularization=regularization)