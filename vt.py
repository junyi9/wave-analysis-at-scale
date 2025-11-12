
import numpy as np
import matplotlib.pyplot as plt
import math

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


# sf = np.load('68c1a0829478cdbd7a829301_WB_L2_smooth.npy')
sf = np.load('smooth/66477985b476f991aef3d7f0_WB_L2_smooth.npy')

print('Original:', sf.shape)

sf = sf[:200,]

print(sf.shape)

x_len = 4.0 # miles
t_len = 4.5*3600 #4.5*3600 # seconds
t_step = 0.1 # simulation step in seconds

x_scale = x_len / sf.shape[0]
print("x_scale:" + str(x_scale))

t_scale = t_len / sf.shape[1]
print("t_scale:" + str(t_scale))
print(x_scale, t_scale)

x_scale = -x_scale # WB.. negative speed

# units in matrix grid/pixels!
x_start = 195
t_start = 1200/5

path = []
path_speed = []

x_hits = []
t_hits = []

# get speed directly (nearest neighbour)
def get_speed_nn(mat, x, t):
    
    x_hits.append(round(x))
    t_hits.append(round(t))
    
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

get_speed = get_speed_bc
    
def gen_VT(x_start, t_start, sf, t_step = 0.1, x_scale = x_scale, t_scale = t_scale):
    """Generate a single virtual trajectory via repeated interpolation lookups."""
    
    path = []
    path_speed = []
    
    x = x_start
    t = t_start
    while (x>1 and x < sf.shape[0]-2) and (t<sf.shape[1]-1):
        
        path.append([x, t])
        
        speed = get_speed(sf, x, t)
        
        path_speed.append(speed)
        # print(x, speed, t_step, x_scale)
        # print(t_step * speed * x_scale)
        x += t_step * speed / 3600 / x_scale # mph/h
        # x += t_step * speed / x_scale # mile per s
        t += t_step / t_scale
        print(x)
        print(t)
        # break
        # print(x, t, speed)
        
    # save the data for later use
        
    path = np.array(path)
    path_speed = np.array(path_speed)
    # add unit to the path, path first column is space and it should be in 0.02*(200-index)
    path[:,0] = 0.02*(200-path[:,0])
    path[:,1] = path[:,1]*5
    traj = np.column_stack((path, path_speed))
    
    return traj

traj = gen_VT(x_start, t_start, sf, t_step=t_step, x_scale=x_scale, t_scale=t_scale)
# print(traj.shape)
np.savetxt('vt.csv', traj, delimiter=',', header='space,time,speed', comments='')