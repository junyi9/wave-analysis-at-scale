# Large-Scale Traffic Wave Analysis Tools

This repository contains tools for analyzing traffic wave patterns and vehicle trajectories for hundreds of days.

## Statement
To be transparent, this README was generated automatically by GitHub Copilot. Some details may be incorrect. I’ll add a review badge after I finish verifying it.

## Input Data Structure

```
vt_data/
├── {RUNID_1}/
│   ├── WB_L1/
│   │   └── vt_all.csv
│   ├── WB_L2/
│   │   └── vt_all.csv
│   ├── WB_L3/
│   │   └── vt_all.csv
│   └── WB_L4/
│       └── vt_all.csv
├── {RUNID_2}/
│   └── ...
```

Each `vt_all.csv` must contain columns: `vehicle_index`, `time`, `space`, `speed`.

## Modules

### `sgwcc_utils.py` - Stop-and-Go Wave Connected Components

Utilities for identifying, tracking, and analyzing stop-and-go wave patterns in traffic flow data.

## Usage Examples

### Basic Wave Analysis (Single File)

```python
from sgwcc_utils import sgwcc_process_file, sgwcc_stitch_components

# Process a single run ID/lane combination
run_id = "66477985b476f991aef3d7f0"
lane_number = 2
file_root_local = "results"
file_path = "vt_data"
seg_speed = 15  # mph
speed_tolerance = 5.0  # mph

# Step 1: Extract and track waves
sgwcc_process_file(
    file_base=run_id,
    lane_file=f"WB_L{lane_number}/vt_all.csv",
    file_root_local=file_root_local,
    file_path=file_path,
    seg_speed=seg_speed,
    speed_tolerance=speed_tolerance,
)

# Step 2: Identify connected components (clusters)
sgwcc_stitch_components(
    file_root_local=file_root_local,
    file_base=run_id,
    lane_file=f"WB_L{lane_number}/vt_all.csv",
    seg_speed=seg_speed,
    file_path=file_path,
)
```

### Batch Processing with Multiprocessing

Use the included `sgwcc_scale.py` script:

```bash
# Edit sgwcc_scale.py to configure your run IDs, lanes, and speeds
python sgwcc_scale.py
```

Or use the batch processing function directly:

```python
from sgwcc_scale import batch_process

# Process all run IDs automatically
batch_process(
    file_path="vt_data",
    file_root_local="results",
    seg_speeds=[15],
    run_ids=None,  # Auto-detect all
    lane_numbers=[1, 2, 3, 4],
    speed_tolerance=5.0,
)

# Process specific run IDs with multiple speeds
batch_process(
    file_path="vt_data",
    file_root_local="results",
    seg_speeds=[10, 15, 20, 25],
    run_ids=["66477985b476f991aef3d7f0", "664b852bb476f991aef3d7f4"],
    lane_numbers=[2, 3],
    speed_tolerance=7.0,
)
```

### Advanced: Custom Speed Tolerance

```python
from sgwcc_utils import sgwcc_extract_wave_points
import pandas as pd

# Load trajectory data
data = pd.read_csv("vt_data/66477985b476f991aef3d7f0/WB_L2/vt_all.csv")

# Extract wave points with custom tolerance
wave_front, wave_tail = sgwcc_extract_wave_points(
    data,
    seg_speed=15,
    speed_tolerance=7.0  # More permissive filtering
)

print(f"Found {len(wave_front)} wave fronts and {len(wave_tail)} wave tails")
```

## Output Structure

```
results/
└── wave_cluster/
    └── 66477985b476f991aef3d7f0/
        └── lane_2/
            ├── wave_front_15.csv           # Detected wave fronts
            ├── wave_tail_15.csv            # Detected wave tails
            ├── pair_data_15.csv            # Paired observations
            ├── wave_front_traj_15.pdf      # Trajectory visualization
            ├── wave_tail_traj_15.pdf       # Trajectory visualization
            ├── file_with_all_info_w_speed_15.csv  # Complete data with trace IDs
            ├── CC_15.csv                   # Connected components
            └── analysis_15_lane_2.pdf      # Component visualization
```

## Key Parameters

- `seg_speed`: Critical segmentation speed in mph (typically 10-25 mph for congestion analysis)
- `speed_tolerance`: Tolerance around critical speed for peak detection (default: 5.0 mph)
- `max_time_delta`: Maximum time gap for trajectory linking (default: 15 seconds)
- `max_space_delta`: Maximum space gap for trajectory linking (default: 0.05 miles)

## Function Reference

### `sgwcc_process_file()`
Main orchestration function that:
1. Extracts wave fronts and tails from trajectory data
2. Pairs wave observations per vehicle
3. Traces wave propagation across vehicles
4. Links traces to create complete wave trajectories

### `sgwcc_stitch_components()`
Identifies connected components by:
1. Finding groups of wave fronts/tails connected through vehicle detections
2. Visualizing components on time-space diagrams
3. Outputting cluster analysis results

### `sgwcc_extract_wave_points()`
Low-level function to extract individual wave observations with custom parameters.

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- scipy
- tqdm
- pathlib (standard library)





