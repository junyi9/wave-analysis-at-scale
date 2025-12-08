import math
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

def cubic(p0, p1, p2, p3, x):
    return p1 + 0.5 * x*(p2 - p0 + x*(2.0*p0 - 5.0*p1 + 4.0*p2 - p3 + x*(3.0*(p1 - p2) + p3 - p0)))

if False:
    pts = [2, 4, 2, 3, 5, 1, 4]
    xa = np.linspace(1, 4.999, 100)
    line = []

    for x in xa:
        
        i = math.floor(x)
        it = cubic(pts[i-1], pts[i+0], pts[i+1], pts[i+2], x-i)
        line.append(it)

    plt.plot(pts, 'o')
    plt.plot(xa, line, '.-')
    plt.show()
    exit()

def get_speed_nn(mat, x, t):
    """Nearest-neighbour lookup of the speed field."""

    return mat[round(x), round(t)]

# linear interpolation (e.g. https://en.wikipedia.org/wiki/Bilinear_interpolation#Example)
def get_speed_lin(mat, x, t):
    
    # integer part
    x0 = math.floor(x)
    t0 = math.floor(t)
    x1 = x0+1
    t1 = t0+1

    # fractional part
    dx = x-x0
    dt = t-t0
    
    it0 = (1-dx)*mat[x0, t0] + dx*mat[x1, t0]
    it1 = (1-dx)*mat[x0, t1] + dx*mat[x1, t1]
    
    it = (1-dt)*it0 + dt*it1
    
    if it<0:
        
        print('NEGATIVE speed!')
        print(mat[x0, t0], mat[x1, t0])
        print(mat[x0, t1], mat[x1, t1])
        print(it0, it1, it)
        print(dx, dt)
        it = 0.0
    
    return it

# bicubic interpolation (e.g. https://www.paulinternet.nl/?page=bicubic)
def get_speed_bc(mat, x, t):
    
    # integer part
    x0 = math.floor(x)
    t0 = math.floor(t)

    # Ensure indices are within valid range for bicubic interpolation
    if (x0 - 1 < 0 or x0 + 2 >= mat.shape[0] or
        t0 - 1 < 0 or t0 + 2 >= mat.shape[1]):
        # Fallback to linear interpolation at the borders
        return get_speed_lin(mat, x, t)

    # fractional part
    dx = x-x0
    dt = t-t0
    
    # do interplation along time first..
    it0 = cubic(mat[x0-1, t0-1], mat[x0-1, t0-0], mat[x0-1, t0+1], mat[x0-1, t0+2], dt)
    it1 = cubic(mat[x0-0, t0-1], mat[x0-0, t0-0], mat[x0-0, t0+1], mat[x0-0, t0+2], dt)
    it2 = cubic(mat[x0+1, t0-1], mat[x0+1, t0-0], mat[x0+1, t0+1], mat[x0+1, t0+2], dt)
    it3 = cubic(mat[x0+2, t0-1], mat[x0+2, t0-0], mat[x0+2, t0+1], mat[x0+2, t0+2], dt)
    
    # ... then over space...
    it = cubic(it0, it1, it2, it3, dx)
    
    return it

INTERPOLATION_METHODS = {
    "nearest": get_speed_nn,
    "linear": get_speed_lin,
    "bicubic": get_speed_bc,
}


def gen_VT(
    x_start: float,
    t_start: float,
    sf: np.ndarray,
    *,
    t_step: float,
    x_scale: float,
    t_scale: float,
    speed_lookup: Callable[[np.ndarray, float, float], float],
):
    """Generate a single virtual trajectory via repeated interpolation lookups."""

    path = []
    path_speed = []

    x = x_start
    t = t_start
    while (x > 1 and x < sf.shape[0] - 2) and (t < sf.shape[1] - 1):

        path.append([x, t])

        speed = speed_lookup(sf, x, t)

        path_speed.append(speed)
        # Integrate through the grid (mph to miles-per-step, then into grid units)
        x += t_step * speed / 3600 / x_scale
        t += t_step / t_scale

    path = np.array(path, dtype=float)
    path_speed = np.array(path_speed, dtype=float)

    return path, path_speed


def generate_virtual_trajectory(
    smooth_field: np.ndarray,
    x_start: float,
    t_start: float,
    *,
    x_len: float = 4.0,
    t_len: float = 4.5 * 3600,
    t_step: float = 0.1,
    interpolation: str = "bicubic",
    westbound: bool = True,
    convert_units: bool = True,
) -> np.ndarray:
    """Generate a virtual trajectory for a pre-loaded smoothed speed field."""

    if smooth_field.ndim != 2:
        raise ValueError("smooth_field must be a 2D array")

    try:
        speed_lookup = INTERPOLATION_METHODS[interpolation.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported interpolation '{interpolation}'") from exc

    seconds_per_column = t_len / smooth_field.shape[1]
    miles_per_row = x_len / smooth_field.shape[0]
    grid_x_scale = -miles_per_row if westbound else miles_per_row

    path, speeds = gen_VT(
        x_start,
        t_start,
        smooth_field,
        t_step=t_step,
        x_scale=grid_x_scale,
        t_scale=seconds_per_column,
        speed_lookup=speed_lookup,
    )

    if path.size == 0:
        return np.empty((0, 3))

    if convert_units:
        path = path.copy()
        path[:, 1] *= seconds_per_column
        if westbound:
            path[:, 0] = miles_per_row * (smooth_field.shape[0] - path[:, 0])
        else:
            path[:, 0] = miles_per_row * path[:, 0]

    traj = np.column_stack((path, speeds))

    return traj


if __name__ == "__main__":
    # as a demo
    x_len = 4.0  # miles
    t_len = 4.5 * 3600  # seconds
    t_step = 0.1  # simulation step in seconds

    sf = np.load('smooth/66477985b476f991aef3d7f0_WB_L2_smooth.npy')
    sf = sf[:200,]

    x_start = 197 # index
    t_start = 600 # index
    traj = generate_virtual_trajectory(
        sf,
        x_start,
        t_start,
        x_len=x_len,
        t_len=t_len,
        t_step=t_step,
        interpolation="bicubic",
        westbound=True,
        convert_units=True,
    )

    plt.scatter(traj[:, 1], traj[:, 2], c=traj[:, 2], s=1, cmap='jet_r')
    plt.savefig('vt.png', dpi=300)
    plt.close()
    np.savetxt('vt.csv', traj, delimiter=',', header='space,time,speed', comments='')