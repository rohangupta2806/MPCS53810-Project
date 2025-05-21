import numpy as np
import pathlib
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib.lines as mlines # For legend proxy artists

def plot_circle(domain, positions, player_colors, outdir, iteration):
    """
    Create and save a visualization of player positions on a circle domain.

    Parameters
    ----------
    domain : Circle
        The circle domain object
    positions : list
        List of player positions
    player_colors : list
        List of colors for each player
    outdir : pathlib.Path
        Directory to save the output image
    iteration : int
        Current iteration number for filename

    Returns
    -------
    str
        Path to the saved image
    """
    R = domain.radius
    pts = np.asarray(positions, float)  # (k,2)
    k = len(pts)

    # ------------- raster parameters -----------------
    RES = 400                  # pixels in one dimension
    PAD = 1.05 * R             # a whisker larger than disk
    xs = np.linspace(-PAD, PAD, RES)
    ys = np.linspace(-PAD, PAD, RES)
    X, Y = np.meshgrid(xs, ys)
    mask = X*X + Y*Y <= R*R     # inside disk
    # -------------------------------------------------

    # distance^2 from every pixel to every player
    #   grid  (RES,RES,1)  vs pts (1,1,k,2)  ->  (RES,RES,k)
    diff = np.stack([X[...,None], Y[...,None]], axis=-1) - pts[None,None,:,:]
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

    ax.set_xlim(-R,R)
    ax.set_ylim(-R,R)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()

    out_path = outdir / f"frame_{iteration:03d}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    return str(out_path)

def plot_sphere(domain, positions, shares, player_colors, outdir, iteration): # Added 'shares' parameter
    """
    Create and save a visualization of player positions on a sphere domain,
    showing two views side-by-side, with a legend including shares. Player tags are on top.

    Parameters
    ----------
    domain : Sphere
        The sphere domain object
    positions : list
        List of player positions
    shares : list or np.ndarray
        List/array of player shares, corresponding to positions
    player_colors : list
        List of colors for each player
    outdir : pathlib.Path
        Directory to save the output image
    iteration : int
        Current iteration number for filename

    Returns
    -------
    str
        Path to the saved image
    """
    fig = plt.figure(figsize=(17, 8))  # Slightly wider for legend

    # --- Common data generation ---
    n_samples_territory = 25000
    rng = np.random.default_rng(42)
    cloud = domain._make_cloud(n_samples_territory, rng)

    pts = np.asarray(positions)
    k = len(pts)

    diff = cloud[:, None, :] - pts[None, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    owner = np.argmin(dist_sq, axis=1)

    # Sphere surface data
    u_sphere = np.linspace(0, 2 * np.pi, 100)
    v_sphere = np.linspace(0, np.pi, 50)
    x_s = domain.radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_s = domain.radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_s = domain.radius * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))
    # --- End of common data generation ---

    view_configs = [
        {"elev": 30, "azim": iteration * 3, "subplot_idx": 1, "title": "View 1"},
        {"elev": 30, "azim": (iteration * 3) + 180, "subplot_idx": 2, "title": "View 2 (Opposite)"}
    ]

    for config in view_configs:
        ax = fig.add_subplot(1, 2, config["subplot_idx"], projection="3d")

        # 1. Plot sphere surface (bottom layer)
        ax.plot_surface(x_s, y_s, z_s, color='lightgray', alpha=0.1,
                        rstride=3, cstride=3, shade=True, zorder=0)

        # 2. Plot territory points
        for player_idx in range(k):
            owned_points = cloud[owner == player_idx]
            if len(owned_points) > 0:
                ax.scatter(
                    owned_points[:, 0], owned_points[:, 1], owned_points[:, 2],
                    c=[player_colors[player_idx][:3]], # Use player color for territory
                    s=10,
                    alpha=0.35,
                    depthshade=True, # Keep depthshade for territories
                    zorder=1 # Territories above surface
                )

        # 3. Plot player markers and then labels (top layers)
        for idx, (x_pos, y_pos, z_pos) in enumerate(pts):
            # Player marker
            ax.scatter(x_pos, y_pos, z_pos, c='black', s=120,
                       depthshade=False, # Markers should not be depthshaded if on top
                       zorder=10) # High zorder for marker
            # Player label
            ax.text(x_pos, y_pos, z_pos, f"P{idx+1}", fontsize=9,
                    ha="center", va="center", color="white", zorder=11, # Highest zorder for text
                    bbox=dict(boxstyle="round,pad=0.15",
                              fc=player_colors[idx][:3], ec="black", alpha=0.95)) # Slightly more opaque bbox

        # Configure view
        ax.set_box_aspect([1, 1, 1])
        ax.set_axis_off()
        ax.view_init(elev=config["elev"], azim=config["azim"])
        ax.set_title(config["title"], fontsize=10)

        # Add Legend to each subplot including shares
        legend_handles = []
        for i in range(k):
            share_text = f"P{i+1}: {shares[i]:.3f}" # Format share to 3 decimal places
            legend_handles.append(mlines.Line2D([], [], color=player_colors[i][:3], marker='o', linestyle='None',
                                               markersize=7, label=share_text))
        ax.legend(handles=legend_handles, loc='upper right', fontsize='x-small', frameon=True, facecolor='white', framealpha=0.7)


    fig.suptitle(f"Iteration {iteration+1}", fontsize=16, y=0.98) # Adjust y for suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.95])


    out_path = outdir / f"frame_{iteration:03d}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    return str(out_path)

def create_animation(frame_dir, output_path="hotelling.gif", fps=2):
    """
    Create an animation from a directory of frames

    Parameters
    ----------
    frame_dir : str or pathlib.Path
        Directory containing the frames
    output_path : str
        Path to save the output animation
    fps : int
        Frames per second for the animation
    """

    frame_dir = pathlib.Path(frame_dir)
    with imageio.get_writer(output_path, mode='I', duration=1/fps) as writer:
        for png in sorted(frame_dir.glob("frame_*.png")):
            image = imageio.imread(png)
            writer.append_data(image)