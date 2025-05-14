import scipy
import numpy as np
import os
import sys
from project_code.geometry.circle import Circle
from project_code.geometry.sphere import Sphere
from project_code.vis.plotting import animate
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import pyplot as plt
from itertools import cycle
COLOR_CYCLE = cycle(plt.get_cmap("tab20").colors)   # 20 bright hues
from scipy.spatial import Voronoi, voronoi_plot_2d

def compute_payoff(domain, positions, n_samples=1000):
    """
    Compute the payoff for each player in the simulation.

    Parameters
    ----------
    domain : The domain in which to run the simulation.
    type: Domain
    positions : The positions of the players.
    type: list
    n_samples : The number of samples to use for computing the payoff.
    type: int

    Returns
    -------
    list
        The payoffs for each player.
    """
    positions = np.asarray(positions, dtype=float).reshape(len(positions), -1)

    eps = 1e-12
    positions += eps * np.random.randn(*positions.shape)  # add some noise to avoid singularities

    try:
        # Compute the Voronoi payoffs for the given positions
        payoffs = domain.voronoi_payoffs(positions)
    except NotImplementedError:
        print("Voronoi payoffs not implemented for this domain. Monte Carlo not yet implemented.")

    return payoffs

def sim(domain=None, num_players=None, start_positions=None, tol=None, max_iter=None, samples_per_iter = None):
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
        raise ValueError("Domain must be either 'circle' or 'sphere'.")
    if num_players <= 0:
        raise ValueError("Number of players must be greater than 0.")
    if tol <= 0:
        raise ValueError("Tolerance must be greater than 0.")
    if max_iter <= 0:
        raise ValueError("Maximum iterations must be greater than 0.")
    if len(start_positions) != num_players:
        raise ValueError("Starting positions must be a list of length equal to the number of players.")

    # Initialize player positions
    positions = start_positions

    if domain == "circle":
        domain = Circle(radius=1)
    elif domain == "sphere":
        domain = Sphere(radius=1)
        # Convert start from angle to Cartesian coordinates
        positions = [Sphere.from_angles(theta, phi) for theta, phi in start_positions]

    # Use scipy minimizer to implement best response dynamics
    # Payoff is the area of the circle/ sphere that is nearest to the player

    history = []
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}")
        payoffs = compute_payoff(domain, positions)
        history.append((positions.copy(), payoffs.copy()))

        old_positions = positions.copy()
        new_positions = positions.copy()
        repulsion_weight = 1e-3
        for i in range(num_players):
            def objective_function(x):
                # Compute the payoff for the player at position x
                candidate = domain.project(x)
                tmp = old_positions.copy()
                tmp[i] = candidate
                payoff = compute_payoff(domain, tmp)
                dmin = np.min([domain.distance(candidate, q) for j, q in enumerate(tmp) if j != i]) + 1e-12
                # The objective is to maximize the payoff minus a repulsion term
                return -(payoff[i] - repulsion_weight / dmin)

            res = scipy.optimize.minimize(objective_function, old_positions[i], method='Nelder-Mead')
            new_positions[i] = domain.project(res.x)
            print(f"Player {i + 1}: {new_positions[i]} -> {res.x} (Payoff: {-res.fun})")

        positions = new_positions

        # Check for convergence
        if np.linalg.norm(np.array(positions) - np.array(old_positions)) < tol:
            print(f"Converged after {iteration} iterations.")
            break

    if isinstance(domain, Circle):
        # ------------- coloured Voronoi on a disk ----------------
        from shapely.geometry import Polygon, Point
        from project_code.geometry.domain import Domain
        R = domain.radius

        vor = Voronoi(positions)
        regions, verts = domain.voronoi_finite_polygons_2d(vor, radius=1000*R)
        disk = Point(0, 0).buffer(R, 256)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.add_patch(plt.Circle((0, 0), R, fill=False, lw=2, color="k"))

        for reg, colour in zip(regions, COLOR_CYCLE):
            poly = Polygon(verts[reg]).intersection(disk)
            if poly.is_empty:
                continue
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, facecolor=colour, edgecolor="k", alpha=0.35)

        pts = np.asarray(positions)
        for idx, (x, y) in enumerate(pts):
            ax.scatter(x, y, s=80, c="k", zorder=3)
            ax.text(x, y, f"P{idx+1}", fontsize=9, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2",
                            fc="white", ec="none", alpha=0.8), zorder=4)

        ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title("Final configuration (disk)")
        plt.show()

    elif isinstance(domain, Sphere):
        # --------- same scatter as before, but label points --------
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        u = np.linspace(0, 2*np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        x = domain.radius * np.outer(np.cos(u), np.sin(v))
        y = domain.radius * np.outer(np.sin(u), np.sin(v))
        z = domain.radius * np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(x, y, z, color="lightgray", alpha=0.4)

        pts = np.asarray(positions)
        for idx, (x, y, z) in enumerate(pts):
            ax.scatter(x, y, z, c="k", s=60, depthshade=True)
            ax.text(x, y, z, f"P{idx+1}", fontsize=8,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2",
                            fc="white", ec="none", alpha=0.75))

        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.set_title("Final configuration (sphere)")
        plt.show()

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

    # Run the simulation
    print("Running simulation with the following parameters:")
    print(f"Domain: {domain}")
    print(f"Number of players: {num_players}")
    print(f"Starting positions: {start_positions}")
    print(f"Tolerance: {tol}")
    print(f"Max iterations: {max_iter}")
    print(f"Samples per iteration: {samples_per_iter}")
    sim(domain=domain, num_players=num_players, start_positions=start_positions, tol=tol, max_iter=max_iter)