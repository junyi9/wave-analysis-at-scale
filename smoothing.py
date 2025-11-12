import numpy as np
from ASM import AdaptiveSmoothing
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# read all the files in the folder 'data'
import os
files = os.listdir('data')
if not os.path.exists('smooth'):
    os.makedirs('smooth')
files = [f for f in files if f.endswith('.npy')]
files = sorted(files)

dx = 0.02
dt = 4.0
kernel_time_window = 3600 * dt
kernel_space_window = 200 * dx
model = AdaptiveSmoothing(kernel_time_window,kernel_space_window, dx, dt).to(device)
model.eval()

for file in files:
    # print the percent of file being processed
    print(f'Processing {file}...')
    print(f'Progress: {files.index(file)+1}/{len(files)}')
    rsf = np.load(f'data/{file}')
    rsf = rsf.astype(np.float32)
    rsf[rsf < 0] = np.nan
    raw = torch.from_numpy(3600*rsf/5280).to(device) # convert from ft/s to mi/h
    with torch.no_grad():
        ssf = model(raw)
    sm = ssf[0].cpu().numpy()
    sm[sm < 0] = np.nan
    # interpolate nan values by linear interpolation
    nans, x = np.isnan(sm), lambda z: z.nonzero()[0]
    # sm[nans] = np.interp(x(nans), x(~nans), sm[~nans])
    sm = np.clip(sm, 0, 100)*5280/3600
    np.save(f'smooth/{file[:-8]}_smooth.npy', sm)