import scipy # Make sure scipy is imported
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

    # Convert to numpy array properly if it's a list of 1D arrays
    if positions_array and isinstance(positions_array[0], np.ndarray):
        try:
            pts_np_array = np.stack(positions_array)
        except ValueError: # Fallback for potentially inconsistent shapes before stacking
            pts_np_array = positions_array # Keep as list if stacking fails, mc_shares should handle
    else:
        pts_np_array = np.asarray(positions_array, dtype=float)


    # make a cloud only when caller didn't supply one (slow path)
    if cloud is None:
        rng = np.random.default_rng()
        cloud = domain._make_cloud(n_samples, rng)

    return domain.mc_shares(pts_np_array, cloud)

def sim(domain_name_str=None, num_players=None, start_positions_config=None, tol=None, max_iter=None, samples_per_iter = None, regularization=None): # Renamed params for clarity
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
    if domain_name_str is None: raise ValueError("Domain must be provided.")
    if num_players is None:
        raise ValueError("Number of players must be provided.")
    if start_positions_config is None:
        raise ValueError("Starting positions must be provided.")
    if tol is None:
        raise ValueError("Tolerance must be provided.")
    if max_iter is None:
        raise ValueError("Maximum number of iterations to run the simulation.")
    if samples_per_iter is None:
        raise ValueError("Samples per iteration must be provided.")
    if regularization is None:
        raise ValueError("Regularization must be provided.")

    if domain_name_str not in ["circle", "sphere"]:
        raise ValueError("Domain must be either 'circle' or 'sphere'.")
    if num_players <= 0:
        raise ValueError("Number of players must be greater than 0.")
    if tol <= 0:
        raise ValueError("Tolerance must be greater than 0.")
    if max_iter <= 0:
        raise ValueError("Maximum iterations must be greater than 0.")
    if len(start_positions_config) != num_players:
        raise ValueError("Starting positions must be a list of length equal to the number of players.")

    outdir = pathlib.Path("frames")
    outdir.mkdir(parents=True, exist_ok=True)
    # Initialize player positions
    positions = start_positions_config

    if domain_name_str == "circle":
        domain_obj = Circle(radius=1)
        # Ensure all positions are numpy arrays
        positions = [np.array(pos, dtype=float) for pos in start_positions_config]
    elif domain_name_str == "sphere":
        domain_obj = Sphere(radius=1)
        # Convert start from angle to Cartesian coordinates
        positions = [Sphere.from_angles(float(theta), float(phi)) for theta, phi in start_positions_config]
        # Ensure all positions are numpy arrays with the same shape
        positions = [np.array(pos, dtype=float) for pos in positions]
    else:
        raise ValueError("Domain string must be 'circle' or 'sphere'.")

    # Frame directory setup
    outdir = pathlib.Path("frames")
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("frame_*.png"): # Clear old frames
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error removing file {f}: {e}")

    player_colors = [next(COLOR_CYCLE) for _ in range(num_players)]

    # History of all player positions at each iteration
    # all_positions_history[iter_num] = [player0_pos_at_iter_num, player1_pos_at_iter_num, ...]
    all_positions_history = [[p.copy() for p in positions]] # Store initial state

    previous_payoff_for_convergence = compute_payoff(domain_obj, positions, n_samples=samples_per_iter)


    for iteration in range(max_iter):
        print(f"Iteration {iteration+1}/{max_iter}")

        rng = np.random.default_rng(iteration)
        cloud = domain_obj._make_cloud(samples_per_iter, rng)

        # Positions at the start of this iteration's best-response calculations
        positions_at_iter_start = [p.copy() for p in positions]

        # This will hold positions after each player makes their best response in this iteration
        positions_after_br_this_iter = [p.copy() for p in positions]

        for i in range(num_players):
            def objective(x_candidate):
                x_reshaped = np.asarray(x_candidate, dtype=float).flatten() # Ensure 1D for project
                cand_pos_i = domain_obj.project(x_reshaped)

                pts_for_objective = [p.copy() for p in positions_at_iter_start]
                pts_for_objective[i] = cand_pos_i

                share_i = compute_payoff(domain_obj, pts_for_objective, cloud=cloud)[i]
                movement_cost = regularization * np.linalg.norm(cand_pos_i - positions_at_iter_start[i])
                return -share_i + movement_cost

            initial_guess_for_player_i = np.array(positions_at_iter_start[i]).flatten()

            res = scipy.optimize.minimize(
                    objective,
                    initial_guess_for_player_i,
                    method='Nelder-Mead',
                    tol=1e-7) # Added optimizer tolerance

            positions_after_br_this_iter[i] = domain_obj.project(res.x)

        # Update current_positions to the state after all players have moved
        positions = [p.copy() for p in positions_after_br_this_iter]
        all_positions_history.append([p.copy() for p in positions])

        # Payoffs calculated at the end of this iteration
        current_payoffs_for_plot_and_convergence = compute_payoff(domain_obj, positions, cloud=cloud)

        # Prepare player_paths for plotting
        player_paths_for_plot = [[] for _ in range(num_players)]
        for hist_state in all_positions_history: # Iterate through each past state
            for player_idx, pos_array in enumerate(hist_state):
                player_paths_for_plot[player_idx].append(pos_array)

        # Plotting
        if isinstance(domain_obj, Circle):
            plot_circle(domain_obj, positions, player_colors, outdir, iteration, player_paths_for_plot)
        elif isinstance(domain_obj, Sphere):
            plot_sphere(domain_obj, positions, current_payoffs_for_plot_and_convergence,
                        player_colors, outdir, iteration, player_paths_for_plot)

        # Convergence Check
        position_change = np.linalg.norm(np.array(positions) - np.array(positions_at_iter_start))
        payoff_change = np.linalg.norm(current_payoffs_for_plot_and_convergence - previous_payoff_for_convergence)

        print(f"  Position change: {position_change:.3e}, Payoff change: {payoff_change:.3e}")
        if iteration > 0 and position_change < tol and payoff_change < tol :
            print(f"Converged after {iteration+1} iterations.")
            break

        previous_payoff_for_convergence = current_payoffs_for_plot_and_convergence.copy()
        if iteration == max_iter - 1:
            print(f"Reached max_iter ({max_iter}) without full convergence criteria met.")

    create_animation(str(outdir), "hotelling.gif", fps=2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hotelling.py <config_file_path>")
        sys.exit(1)
    file = sys.argv[1]

    if not os.path.exists(file):
        print(f"File {file} does not exist.")
        sys.exit(1)

    def clean(line):
        return line.split("#")[0].strip()

    with open(file, 'r') as f:
        lines = [clean(line) for line in f.readlines() if clean(line)] # Filter empty lines

    domain_name_from_config = lines[0]
    num_players_from_config = int(lines[1])

    # Correctly parse start_positions based on domain
    # For sphere, positions are (theta, phi) pairs; for circle, (x, y) pairs
    parsed_start_positions = [list(map(float, line.split())) for line in lines[2:2+num_players_from_config]]

    tol_from_config = float(lines[2+num_players_from_config])
    max_iter_from_config = int(lines[3+num_players_from_config])
    samples_per_iter_from_config = int(lines[4+num_players_from_config])
    regularization_from_config = float(lines[5+num_players_from_config])

    print("Running simulation with the following parameters:")
    print(f"Domain: {domain_name_from_config}")
    print(f"Number of players: {num_players_from_config}")
    print(f"Starting positions (from config): {parsed_start_positions}")
    print(f"Tolerance: {tol_from_config}")
    print(f"Max iterations: {max_iter_from_config}")
    print(f"Samples per iteration: {samples_per_iter_from_config}")
    print(f"Regularization: {regularization_from_config}")
    sim(domain_name_str=domain_name_from_config,
        num_players=num_players_from_config,
        start_positions_config=parsed_start_positions,
        tol=tol_from_config,
        max_iter=max_iter_from_config,
        samples_per_iter=samples_per_iter_from_config,
        regularization=regularization_from_config)