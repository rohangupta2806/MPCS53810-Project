from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_frame_2d(ax, players, regions, domain):
    ax.clear()
    # ask the domain to draw its boundary
    domain.draw_boundary(ax)
    # draw the cells
    ax.add_collection(PolyCollection(regions, facecolors='none', edgecolors='gray'))
    # draw the players
    pts = players
    ax.scatter(pts[:,0], pts[:,1], c='C0', s=60, edgecolor='k')
    ax.set_aspect('equal')

def draw_frame_3d(ax, players, regions, domain):
    ax.clear()
    domain.draw_surface(ax)
    for verts in regions:
        ax.add_collection(Poly3DCollection([verts], facecolors='none', edgecolors='k'))
    ax.scatter(players[:,0], players[:,1], players[:,2], c='r', s=60)

def animate(history, domain, filename='out.gif', fps=5):
    # pick the right drawer
    is3d = hasattr(domain, 'draw_surface')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') if is3d else fig.add_subplot(111)
    drawer = draw_frame_3d if is3d else draw_frame_2d

    def update(i):
        pts, regions = history[i]
        drawer(ax, pts, regions, domain)
        ax.set_title(f"Step {i}")

    anim = FuncAnimation(fig, update, frames=len(history), interval=1000/fps)
    anim.save(filename, writer='imagemagick')