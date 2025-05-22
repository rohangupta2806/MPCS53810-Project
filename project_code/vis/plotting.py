import numpy as np
import pathlib
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import matplotlib.lines as mlines
import os

def plot_circle(domain, current_positions_list, player_colors, outdir, iteration, player_paths_history):
    R = domain.radius
    pts_current = np.asarray(current_positions_list, float)
    k = len(pts_current)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6)) # Adjusted figsize
    ax_territory = axes[0]
    ax_paths = axes[1]
    fig.suptitle(f"Circle Simulation - Iteration {iteration+1}", fontsize=14)

    # --- Plot 1: Territories (Left Subplot) ---
    RES = 250 # Reduced for speed
    PAD = 1.05 * R
    xs, ys = np.linspace(-PAD, PAD, RES), np.linspace(-PAD, PAD, RES)
    X, Y = np.meshgrid(xs, ys)
    mask = X*X + Y*Y <= R*R

    diff = np.stack([X[...,None], Y[...,None]], axis=-1) - pts_current[None,None,:,:]
    dist2 = (diff**2).sum(axis=-1)
    owner = dist2.argmin(axis=-1)
    owner[~mask] = k

    img = np.zeros((RES, RES, 3))
    for pid in range(k): img[owner == pid] = player_colors[pid][:3]
    img[owner == k] = (1,1,1)

    ax_territory.imshow(img, extent=(-PAD, PAD, -PAD, PAD), origin='lower', zorder=0)
    ax_territory.add_patch(plt.Circle((0,0), R, fc='none', ec='k', lw=1, zorder=1))

    for i,(x,y) in enumerate(pts_current):
        ax_territory.scatter(x,y,c='k',s=60,zorder=4)
        ax_territory.scatter(x,y,c=player_colors[i][:3], s=30, zorder=5)
        ax_territory.text(x,y,f"P{i+1}",ha='center',va='center',fontsize=7, zorder=6,
                          bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none', alpha=0.7))

    ax_territory.set_xlim(-R*1.05, R*1.05); ax_territory.set_ylim(-R*1.05, R*1.05)
    ax_territory.set_aspect('equal'); ax_territory.set_axis_off()
    ax_territory.set_title(f"Territories", fontsize=10)

    # --- Plot 2: Paths (Right Subplot) ---
    ax_paths.add_patch(plt.Circle((0,0), R, fc='whitesmoke', ec='k', lw=1, alpha=0.5, zorder=0))
    for player_idx, path_list_of_arrays in enumerate(player_paths_history):
        if len(path_list_of_arrays) > 1:
            path_arr = np.array(path_list_of_arrays) # Converts list of (2,) arrays to (N,2)
            ax_paths.plot(path_arr[:, 0], path_arr[:, 1], color=player_colors[player_idx],
                          linestyle='-', linewidth=1.2, alpha=0.7, zorder=1)
            # Mark current position on path plot
            ax_paths.scatter(path_arr[-1, 0], path_arr[-1, 1], color=player_colors[player_idx],
                             s=40, edgecolor='k', zorder=2, linewidth=0.5)
            ax_paths.text(path_arr[-1, 0], path_arr[-1, 1] + R*0.06, f"P{player_idx+1}",
                          ha='center', va='bottom', fontsize=6, color='black', zorder=3)
        elif len(path_list_of_arrays) == 1: # Plot initial point if no path yet
             ax_paths.scatter(path_list_of_arrays[0][0], path_list_of_arrays[0][1], color=player_colors[player_idx],
                             s=40, edgecolor='k', zorder=2, linewidth=0.5)
             ax_paths.text(path_list_of_arrays[0][0], path_list_of_arrays[0][1] + R*0.06, f"P{player_idx+1}",
                          ha='center', va='bottom', fontsize=6, color='black', zorder=3)


    ax_paths.set_xlim(-R*1.05, R*1.05); ax_paths.set_ylim(-R*1.05, R*1.05)
    ax_paths.set_aspect('equal'); ax_paths.set_axis_off()
    ax_paths.set_title(f"Player Paths", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
    out_path = outdir / f"frame_{iteration:03d}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)

def plot_sphere(domain, current_positions_list, shares, player_colors, outdir, iteration, player_paths_history):
    fig = plt.figure(figsize=(13, 11)) # Adjusted for 2x2
    fig.suptitle(f"Sphere Simulation - Iteration {iteration+1}", fontsize=14)

    pts_current = np.asarray(current_positions_list)
    k = len(pts_current)

    # --- Common data for territories ---
    n_samples_territory = 8000 # Reduced
    rng_cloud = np.random.default_rng(42) # Fixed seed for territory cloud for consistency across frames
    cloud_territory = domain._make_cloud(n_samples_territory, rng_cloud)
    diff_territory = cloud_territory[:, None, :] - pts_current[None, :, :]
    dist_sq_territory = np.sum(diff_territory**2, axis=2)
    owner_territory = np.argmin(dist_sq_territory, axis=1)

    # --- Common data for sphere surface ---
    u_sphere, v_sphere = np.linspace(0, 2*np.pi, 30), np.linspace(0, np.pi, 15) # Reduced
    x_s = domain.radius * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_s = domain.radius * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_s = domain.radius * np.outer(np.ones_like(u_sphere), np.cos(v_sphere))

    base_azim = iteration * 1.0 # Slower rotation

    subplot_defs = [
        {"pos": (2,2,1), "type": "territory", "view_elev": 30, "view_azim_offset": 0, "title": "Territories - View 1"},
        {"pos": (2,2,2), "type": "territory", "view_elev": 30, "view_azim_offset": 180, "title": "Territories - View 2"},
        {"pos": (2,2,3), "type": "path",      "view_elev": 30, "view_azim_offset": 0, "title": "Paths - View 1"},
        {"pos": (2,2,4), "type": "path",      "view_elev": 30, "view_azim_offset": 180, "title": "Paths - View 2"}
    ]

    for s_def in subplot_defs:
        ax = fig.add_subplot(*s_def["pos"], projection="3d")
        ax.plot_surface(x_s, y_s, z_s, color='lightgray', alpha=0.05, rstride=2, cstride=2, shade=True, zorder=0)

        if s_def["type"] == "territory":
            for player_idx in range(k):
                owned_pts = cloud_territory[owner_territory == player_idx]
                if len(owned_pts) > 0:
                    ax.scatter(owned_pts[:,0], owned_pts[:,1], owned_pts[:,2], c=[player_colors[player_idx][:3]], s=3, alpha=0.2, zorder=1)

            for idx, pos_xyz in enumerate(pts_current):
                ax.scatter(pos_xyz[0], pos_xyz[1], pos_xyz[2], c='k', s=80, depthshade=False, zorder=10)
                ax.scatter(pos_xyz[0], pos_xyz[1], pos_xyz[2], c=player_colors[idx][:3], s=40, depthshade=False, zorder=11)
                ax.text(pos_xyz[0], pos_xyz[1], pos_xyz[2] + domain.radius*0.06, f"P{idx+1}", fontsize=7, ha='center', va='bottom',
                        color='k', zorder=12, bbox=dict(boxstyle="round,pad=0.1", fc=player_colors[idx][:3], ec='none', alpha=0.6))

            legend_handles = [mlines.Line2D([],[],color=player_colors[i][:3],marker='o',ls='None',ms=4,label=f"P{i+1}:{shares[i]:.2f}") for i in range(k)]
            ax.legend(handles=legend_handles, loc='upper right', fontsize='xx-small', frameon=True, facecolor='white', framealpha=0.5)

        elif s_def["type"] == "path":
            for player_idx, path_list_of_arrays in enumerate(player_paths_history):
                if len(path_list_of_arrays) > 1:
                    path_arr = np.array(path_list_of_arrays)
                    ax.plot(path_arr[:,0], path_arr[:,1], path_arr[:,2], color=player_colors[player_idx], lw=1.2, alpha=0.6, zorder=5)
                    curr_pos = path_arr[-1]
                    ax.scatter(curr_pos[0], curr_pos[1], curr_pos[2], color=player_colors[player_idx], s=50, edgecolor='k', depthshade=False, zorder=10, linewidth=0.5)
                    ax.text(curr_pos[0], curr_pos[1], curr_pos[2] + domain.radius*0.06, f"P{player_idx+1}", fontsize=6, ha='center', va='bottom', color='k', zorder=11)
                elif len(path_list_of_arrays) == 1: # Initial point
                    curr_pos = path_list_of_arrays[0]
                    ax.scatter(curr_pos[0], curr_pos[1], curr_pos[2], color=player_colors[player_idx], s=50, edgecolor='k', depthshade=False, zorder=10, linewidth=0.5)
                    ax.text(curr_pos[0], curr_pos[1], curr_pos[2] + domain.radius*0.06, f"P{player_idx+1}", fontsize=6, ha='center', va='bottom', color='k', zorder=11)


        ax.set_box_aspect([1,1,1]); ax.set_axis_off()
        ax.view_init(elev=s_def["view_elev"], azim=base_azim + s_def["view_azim_offset"])
        ax.set_title(s_def["title"], fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust for suptitle
    out_path = outdir / f"frame_{iteration:03d}.png"
    fig.savefig(out_path, dpi=100) # Reduced DPI
    plt.close(fig)
    return str(out_path)

def create_animation(frame_dir, output_path="hotelling.gif", fps=2):
    frame_dir = pathlib.Path(frame_dir)
    print(f"Looking for frames in: {frame_dir.resolve()}")
    png_files = sorted(frame_dir.glob("frame_*.png"))

    if not png_files:
        print(f"No PNG files found in {frame_dir}. Animation not created.")
        return

    print(f"Found {len(png_files)} PNG files. Attempting to create GIF: {output_path}")
    with imageio.get_writer(output_path, mode='I', duration=int(1000/fps), loop=0) as writer:
        frames_processed = 0
        for png_path in png_files:
            try:
                if os.path.getsize(png_path) == 0:
                    print(f"Skipping empty file: {png_path}")
                    continue
                image = imageio.imread(png_path)
                writer.append_data(image)
                frames_processed += 1
            except Exception as e:
                print(f"Could not read or process file {png_path}: {e}. Skipping.")

    if frames_processed > 0:
        print(f"Animation saved to {output_path} with {frames_processed} frames.")
    else:
        print(f"No valid frames were processed. Animation {output_path} might be empty or not created.")