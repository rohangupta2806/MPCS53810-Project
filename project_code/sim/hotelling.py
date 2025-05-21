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
import imageio.v2 as imageio

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
    pts = np.asarray(positions, float).reshape(len(positions), -1)

    # make a cloud only when caller didn’t supply one (slow path)
    if cloud is None:
        rng   = np.random.default_rng()
        cloud = domain._make_cloud(n_samples, rng)   # helper you added

    return domain.mc_shares(pts, cloud)              # helper you added

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
        raise ValueError("Maximum number of iterations to run the simulation.")
    if samples_per_iter is None:
        raise ValueError("Samples per iteration must be provided.")

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
    elif domain == "sphere":
        domain = Sphere(radius=1)
        # Convert start from angle to Cartesian coordinates
        positions = [Sphere.from_angles(theta, phi) for theta, phi in start_positions]

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
                cand = domain.project(x)
                pts  = old_positions.copy()
                pts[i] = cand
                share_i = compute_payoff(domain, pts, cloud=cloud)[i]
                return -share_i                              # maximise

            res = scipy.optimize.minimize(
                    objective,
                    old_positions[i],
                    method='Nelder-Mead')

            new_positions[i] = domain.project(res.x)

        positions = new_positions

        from scipy.spatial.distance import pdist
        print("min distance:", pdist(positions).min())

        payoffs = compute_payoff(domain, positions, cloud=cloud)
        for idx, p in enumerate(payoffs, 1):
            print(f" Player {idx}: share {p:.4f}")

        current_payoff = payoffs

        if isinstance(domain, Circle):
            R   = domain.radius
            pts = np.asarray(positions, float)          # (k,2)
            k   = len(pts)

            # ------------- raster parameters -----------------
            RES   = 400                  # pixels in one dimension
            PAD   = 1.05 * R             # a whisker larger than disk
            xs    = np.linspace(-PAD, PAD, RES)
            ys    = np.linspace(-PAD, PAD, RES)
            X, Y  = np.meshgrid(xs, ys)
            mask  = X*X + Y*Y <= R*R     # inside disk
            # -------------------------------------------------

            # distance^2 from every pixel to every player
            #   grid  (RES,RES,1)  vs pts (1,1,k,2)  ->  (RES,RES,k)
            diff  = np.stack([X[...,None], Y[...,None]], axis=-1) - pts[None,None,:,:]
            dist2 = (diff**2).sum(axis=-1)

            owner = dist2.argmin(axis=-1)          # (RES,RES) int map
            owner[~mask] = k                       # outside disk sentinels

            # build an RGBA image
            img = np.zeros((RES, RES, 3))
            for pid in range(k):
                img[owner == pid] = player_colors[pid][:3]  # ignore alpha
            img[owner == k] = (1,1,1)                       # white outside

            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(img, extent=(-PAD, PAD, -PAD, PAD), origin='lower')
            ax.add_patch(plt.Circle((0,0), R, fc='none', ec='k', lw=2))

            # scatter players & labels
            for i,(x,y) in enumerate(pts):
                ax.scatter(x,y,c='k',s=60,zorder=3)
                ax.text(x,y,f"P{i+1}",ha='center',va='center',fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.2',fc='white',ec='none'))

            ax.set_xlim(-R,R); ax.set_ylim(-R,R)
            ax.set_aspect('equal'); ax.set_axis_off()
            fig.tight_layout()
            fig.savefig(outdir / f"frame_{iteration:03d}.png", dpi=300)
            plt.close(fig)

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

        # Check for convergence
        if np.linalg.norm(np.array(positions) - np.array(old_positions)) < tol\
                and np.linalg.norm(np.array(current_payoff) - np.array(previous_payoff)) < tol:
            print(f"Converged after {iteration} iterations.")
            break

    with imageio.get_writer("hotelling.gif", mode='I', duration=0.5) as writer:
        for png in sorted(outdir.glob("frame_*.png")):
            image = imageio.imread(png)
            writer.append_data(image)

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
    sim(domain=domain, num_players=num_players, start_positions=start_positions,
        tol=tol, max_iter=max_iter, samples_per_iter=samples_per_iter)