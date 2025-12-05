import pandas as pd

def vt_read_data(vt_number, file_root):
    # note: this is for the FISTS v0.1 raw_data (prelease version, latter version is conbined as one csv)
    data = pd.read_csv(f'{file_root}/VT_{vt_number}.csv')
    return data


def decompose_trajectory(vt_sample):
    mean_speed = (vt_sample.space.max() - vt_sample.space.min())/(vt_sample.time.max() - vt_sample.time.min())
    vt_sample['average_speed'] = mean_speed
    initial_space = vt_sample['space'].max()
    vt_sample['nominal_space'] = initial_space - mean_speed * (vt_sample['time'] - vt_sample['time'].min())
    vt_sample['oscillation_space'] = -(vt_sample['space'] - vt_sample['nominal_space'])
    return vt_sample

def decompose_trajectory_fixed(vt_sample, fixed_speed = 30):
    mean_speed = fixed_speed/3600
    vt_sample['average_speed'] = mean_speed
    initial_space = vt_sample['space'].max()
    vt_sample['nominal_space'] = initial_space - mean_speed * (vt_sample['time'] - vt_sample['time'].min())
    vt_sample['oscillation_space'] = -(vt_sample['space'] - vt_sample['nominal_space'])
    return vt_sample

def vt_read_and_decompose_data(vt_number, file_root):
    # note: this is for the FISTS v0.1 raw_data (prelease version, latter version is conbined as one csv)
    data = pd.read_csv(f'{file_root}/VT_{vt_number}.csv')
    vt_sample = data
    mean_speed = (vt_sample.space.max() - vt_sample.space.min())/(vt_sample.time.max() - vt_sample.time.min())
    vt_sample['average_speed'] = mean_speed
    initial_space = vt_sample['space'].max()
    vt_sample['nominal_space'] = initial_space - mean_speed * (vt_sample['time'] - vt_sample['time'].min())
    vt_sample['oscillation_space'] = -(vt_sample['space'] - vt_sample['nominal_space'])
    return vt_sample


def decompose_trajectory_ARED(vt_sample):
    mean_speed = (vt_sample.space.max() - vt_sample.space.min())/(vt_sample.time.max() - vt_sample.time.min())
    vt_sample['average_speed'] = mean_speed
    initial_space = vt_sample['space'].min()
    vt_sample['nominal_space'] = initial_space + mean_speed * (vt_sample['time'] - vt_sample['time'].min())
    vt_sample['oscillation_space'] = (vt_sample['space'] - vt_sample['nominal_space'])
    return vt_sample