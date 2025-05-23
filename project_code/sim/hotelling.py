import scipy
import numpy as np
import os
import sys
from project_code.geometry.circle import Circle
from project_code.geometry.sphere import Sphere
from project_code.geometry.square import Square
from matplotlib import pyplot as plt
from itertools import cycle
COLOR_CYCLE = cycle(plt.get_cmap("tab20").colors)
import pathlib
from project_code.vis.plotting import plot_circle, plot_sphere, plot_square, create_animation

def sim(domain_name_str=None, num_players=None, start_positions_config=None, tol=None, max_iter=None, samples_per_iter = None, regularization=None):
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

    if domain_name_str not in ["circle", "sphere", "square"]:
        raise ValueError("Domain must be 'circle', 'sphere', or 'square'.")
    if num_players <= 0:
        raise ValueError("Number of players must be greater than 0.")
    if tol <= 0:
        raise ValueError("Tolerance must be greater than 0.")
    if max_iter <= 0:
        raise ValueError("Maximum iterations must be greater than 0.")
    if len(start_positions_config) != num_players:
        raise ValueError("Starting positions must be a list of length equal to the number of players.")


    if domain_name_str == "circle":
        domain_obj = Circle(radius=1)
        positions = [np.array(pos, dtype=float) for pos in start_positions_config]
    elif domain_name_str == "sphere":
        domain_obj = Sphere(radius=1)
        positions = [Sphere.from_angles(float(theta), float(phi)) for theta, phi in start_positions_config]
        positions = [np.array(pos, dtype=float) for pos in positions]
    elif domain_name_str == "square":
        domain_obj = Square(side_length=2)
        positions = [np.array(pos, dtype=float) for pos in start_positions_config]
    else:
        raise ValueError("Domain string must be 'circle', 'sphere', or 'square'.")

    outdir = pathlib.Path("frames")
    outdir.mkdir(parents=True, exist_ok=True)
    for f in outdir.glob("frame_*.png"):
        try:
            os.remove(f)
        except OSError as e:
            print(f"Error removing file {f}: {e}")

    player_colors = [next(COLOR_CYCLE) for _ in range(num_players)]
    all_positions_history = [[p.copy() for p in positions]]

    # Use the method from domain_obj
    previous_payoff_for_convergence = domain_obj.compute_payoff(positions, n_samples=samples_per_iter)

    for iteration in range(max_iter):
        print(f"Iteration {iteration+1}/{max_iter}")
        rng = np.random.default_rng(iteration)
        cloud = domain_obj._make_cloud(samples_per_iter, rng)
        positions_at_iter_start = [p.copy() for p in positions]
        positions_after_br_this_iter = [p.copy() for p in positions]

        for i in range(num_players):
            def objective(x_candidate):
                x_reshaped = np.asarray(x_candidate, dtype=float).flatten()
                cand_pos_i = domain_obj.project(x_reshaped)
                pts_for_objective = [p.copy() for p in positions_at_iter_start]
                pts_for_objective[i] = cand_pos_i
                # Use the method from domain_obj
                share_i = domain_obj.compute_payoff(pts_for_objective, cloud=cloud)[i]
                movement_cost = regularization * np.linalg.norm(cand_pos_i - positions_at_iter_start[i])
                return -share_i + movement_cost

            initial_guess_for_player_i = np.array(positions_at_iter_start[i]).flatten()
            res = scipy.optimize.minimize(
                objective,
                initial_guess_for_player_i,
                method='Nelder-Mead',
                tol=1e-7
            )
            positions_after_br_this_iter[i] = domain_obj.project(res.x)

        positions = [p.copy() for p in positions_after_br_this_iter]
        all_positions_history.append([p.copy() for p in positions])

        # Use the method from domain_obj
        current_payoffs_for_plot_and_convergence = domain_obj.compute_payoff(positions, cloud=cloud)

        player_paths_for_plot = [[] for _ in range(num_players)]
        for hist_state in all_positions_history:
            for player_idx, pos_array in enumerate(hist_state):
                player_paths_for_plot[player_idx].append(pos_array)

        if isinstance(domain_obj, Circle):
            plot_circle(domain_obj, positions, player_colors, outdir, iteration, player_paths_for_plot, max_iter)
        elif isinstance(domain_obj, Sphere):
            plot_sphere(domain_obj, positions, current_payoffs_for_plot_and_convergence,
                        player_colors, outdir, iteration, player_paths_for_plot, max_iter)
        elif isinstance(domain_obj, Square):
            plot_square(domain_obj, positions, player_colors, outdir, iteration, player_paths_for_plot, max_iter)

        position_change = np.linalg.norm(np.array(positions) - np.array(positions_at_iter_start))
        payoff_change = np.linalg.norm(current_payoffs_for_plot_and_convergence - previous_payoff_for_convergence)
        print(f"  Position change: {position_change:.3e}, Payoff change: {payoff_change:.3e}")

        if iteration > 0 and position_change < tol and payoff_change < tol:
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
        lines = [clean(line) for line in f.readlines() if clean(line)]

    domain_name_from_config = lines[0]
    num_players_from_config = int(lines[1])
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